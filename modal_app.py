"""
Modal app for running InstantSplat++ on GPU.
InstantSplat++: Sparse-view Gaussian Splatting in Seconds
Project page: https://instantsplat.github.io/
Code: https://github.com/phai-lab/InstantSplatPP

Deploy with: modal deploy modal_app.py
"""

import modal

# Create the Modal app
app = modal.App("instantplus")

# InstantSplat++ requires Python 3.10, PyTorch 2.1.2 and CUDA.
# We use the official PyTorch image (matching InstantSplat++'s own Dockerfile)
# which has PyTorch, CUDA dev tools, and the correct compiler already set up.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel",
        add_python=None,
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0",
            "DEBIAN_FRONTEND": "noninteractive",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
        }
    )
    .run_commands(
        # System dependencies
        "apt-get update && apt-get install -y "
        "git wget ffmpeg libgl1-mesa-glx libglib2.0-0 "
        "build-essential ninja-build cmake "
        "&& rm -rf /var/lib/apt/lists/*",

        # ── Phase 1: Ensure pip, wheel and setuptools are compatible
        # Pin setuptools < 75 to keep pkg_resources (needed by torch cpp_extension)
        "pip install --no-cache-dir -U pip wheel 'setuptools<75'",

        # ── Phase 2: Clone InstantSplat++ (recursive for submodules)
        "git clone --recursive https://github.com/phai-lab/InstantSplatPP.git /opt/instantsplat",

        # ── Phase 3: Download MASt3R pre-trained checkpoint (~600 MB)
        "mkdir -p /opt/instantsplat/mast3r/checkpoints && "
        "wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth "
        "-P /opt/instantsplat/mast3r/checkpoints/",

        # ── Phase 4: Install Python dependencies
        "cd /opt/instantsplat && pip install --no-cache-dir -r requirements.txt",

        # ── Phase 5: Force NumPy < 2 and headless OpenCV (avoid ABI issues)
        "pip uninstall -y numpy opencv-python 2>/dev/null || true",
        "pip install --no-cache-dir 'numpy<2' 'opencv-python-headless<4.12'",

        # ── Phase 6: Build & install CUDA submodules
        "cd /opt/instantsplat && pip install -v --no-build-isolation ./submodules/simple-knn",
        "cd /opt/instantsplat && pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization",
        "cd /opt/instantsplat && pip install -v --no-build-isolation ./submodules/fused-ssim",

        # ── Phase 7: Compile RoPE CUDA kernels (CroCo v2)
        # The curope module is inside the MASt3R → DUSt3R → CroCo submodule chain
        "bash -c 'CUROPE=$(find /opt/instantsplat -name \"curope\" -path \"*/croco/models/curope\" -type d | head -1); "
        "if [ -n \"$CUROPE\" ]; then cd \"$CUROPE\" && python setup.py build_ext --inplace; "
        "else echo \"WARNING: curope not found, skipping RoPE CUDA kernel build\"; fi'",

        # ── Phase 8: Install plyfile for post-processing
        "pip install --no-cache-dir plyfile",

        # NOTE: InstantSplat++ PP optimizer does not support runtime
        # densification (the required gradient-accum tensors are not
        # initialised by training_setup_pp).  Instead we rely on
        # aggressive post-processing: positional outlier removal,
        # tight scale clamping and opacity pruning.

        # ── Phase 9: Verify installation
        'python -c "'
        "import torch; "
        "print(f'torch={torch.__version__}  cuda={torch.version.cuda}'); "
        "assert torch.version.cuda is not None, 'torch has NO CUDA support!'; "
        "import numpy; print(f'numpy={numpy.__version__}'); "
        "print('All OK')\"",
    )
)

# Volume for caching any runtime downloads (HuggingFace, etc.)
volume = modal.Volume.from_name("instantplus-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=900,  # 15 minutes max
    volumes={"/cache": volume},
)
def process_images(image_bytes_list: list[bytes], filenames: list[str]) -> bytes:
    """
    Process images with InstantSplat++ and return a PLY file.

    Pipeline:
    1. Save images to disk in the expected directory structure
    2. Run init_geo.py (MASt3R-based camera pose estimation + point cloud)
    3. Run train.py (3D Gaussian Splatting training with pruning)
    4. Post-process and return the PLY
    """
    import io
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    # Route heavy downloads through the shared volume
    os.environ["TORCH_HOME"] = "/cache/torch"
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Add InstantSplat++ to Python path
    sys.path.insert(0, "/opt/instantsplat")

    # -----------------------------------------------------------
    # Helper: create synthetic views using perspective homography
    # -----------------------------------------------------------
    def create_synthetic_views(pil_img: Image.Image, n_views: int = 3) -> list[Image.Image]:
        """
        Generate perspective-transformed views from one image for MASt3R.

        Instead of tiny crops (which give MASt3R almost no parallax),
        we apply homography transforms that simulate the camera rotating
        ±8° around the subject. This produces the kind of baseline that
        MASt3R needs to estimate accurate depth.
        """
        img_np = np.array(pil_img)
        h, w = img_np.shape[:2]

        # Approximate focal length – a common heuristic is 1.2× the larger
        # image dimension, which corresponds to ~45° horizontal FoV.
        f = max(h, w) * 1.2
        cx, cy = w / 2.0, h / 2.0

        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0,  1]], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        views: list[Image.Image] = []

        # Horizontal rotation angles: center, left, right
        # ±8° gives enough parallax for MASt3R without too much distortion
        angles_deg = [0.0, -8.0, 8.0]
        if n_views > 3:
            angles_deg = np.linspace(-10, 10, n_views).tolist()

        for angle in angles_deg[:n_views]:
            theta = np.radians(angle)

            # Rotation around the vertical (Y) axis
            Ry = np.array([
                [ np.cos(theta), 0, np.sin(theta)],
                [ 0,             1, 0             ],
                [-np.sin(theta), 0, np.cos(theta)],
            ], dtype=np.float64)

            # Add a small vertical tilt (20% of horizontal) for richer geometry
            phi = np.radians(angle * 0.2)
            Rx = np.array([
                [1, 0,            0           ],
                [0, np.cos(phi), -np.sin(phi) ],
                [0, np.sin(phi),  np.cos(phi) ],
            ], dtype=np.float64)

            R = Rx @ Ry
            H = K @ R @ K_inv

            warped = cv2.warpPerspective(
                img_np, H, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            views.append(Image.fromarray(warped))

        return views

    # -----------------------------------------------------------
    # Helper: clean up / post-process the Gaussian PLY
    # -----------------------------------------------------------
    def clean_gaussian_ply(ply_path: Path, max_size_mb: float = 4.0) -> bytes:
        """
        Post-process the Gaussian PLY to remove degenerate splats:

        1. Remove positional outliers (far from scene center)
        2. Clamp extreme scale values (prevents needle-like splats)
        3. Clamp aspect ratios between axes (max 10:1)
        4. Remove very low-opacity Gaussians
        5. Trim to max_size_mb if necessary
        """
        from plyfile import PlyData, PlyElement

        ply = PlyData.read(str(ply_path))
        verts = ply["vertex"]
        data = np.copy(verts.data)
        n_original = len(data)

        prop_names = [p.name for p in verts.properties]
        has_scales = all(f"scale_{i}" in prop_names for i in range(3))
        has_opacity = "opacity" in prop_names
        has_xyz = all(c in prop_names for c in ("x", "y", "z"))

        # --- (a) Remove positional outliers ---
        # Gaussians placed far from the scene centre are artefacts
        if has_xyz:
            xyz = np.stack([data["x"].astype(np.float64),
                            data["y"].astype(np.float64),
                            data["z"].astype(np.float64)], axis=-1)
            centre = np.median(xyz, axis=0)
            dists = np.linalg.norm(xyz - centre, axis=1)
            # Keep points within 3× the 90th-percentile distance
            dist_p90 = np.percentile(dists, 90)
            dist_threshold = dist_p90 * 3.0
            pos_mask = dists < dist_threshold
            n_pos_removed = int(np.sum(~pos_mask))
            if n_pos_removed > 0:
                data = data[pos_mask]
                print(f"🗺️  Removed {n_pos_removed:,} positional outlier Gaussians "
                      f"(>{dist_threshold:.2f} from centre)")

        # --- (b) Clamp scale values ---
        if has_scales:
            s0 = data["scale_0"].astype(np.float64)
            s1 = data["scale_1"].astype(np.float64)
            s2 = data["scale_2"].astype(np.float64)

            # exp(-7) ≈ 0.0009,  exp(0.5) ≈ 1.65
            # Tighter upper bound than before (was 1.5 → now 0.5)
            # to prevent oversized Gaussians from dominating the scene
            MIN_LOG_SCALE = -7.0
            MAX_LOG_SCALE = 0.5
            s0 = np.clip(s0, MIN_LOG_SCALE, MAX_LOG_SCALE)
            s1 = np.clip(s1, MIN_LOG_SCALE, MAX_LOG_SCALE)
            s2 = np.clip(s2, MIN_LOG_SCALE, MAX_LOG_SCALE)

            # --- (c) Clamp aspect ratio ---
            # Max 10:1 ratio between any two scale axes
            MAX_ASPECT_RATIO_LOG = np.log(10.0)
            scales = np.stack([s0, s1, s2], axis=-1)
            s_min = scales.min(axis=-1)
            s_max = scales.max(axis=-1)
            spread = s_max - s_min
            stretched = spread > MAX_ASPECT_RATIO_LOG

            if np.any(stretched):
                s_median = np.median(scales, axis=-1)
                for _i, si in enumerate([s0, s1, s2]):
                    too_big = stretched & (si == s_max)
                    too_small = stretched & (si == s_min)
                    si[too_big] = s_median[too_big] + MAX_ASPECT_RATIO_LOG / 2
                    si[too_small] = s_median[too_small] - MAX_ASPECT_RATIO_LOG / 2
                n_fixed = int(np.sum(stretched))
                print(f"🔧 Fixed aspect ratio on {n_fixed:,} / {len(data):,} Gaussians")

            data["scale_0"] = s0.astype(np.float32)
            data["scale_1"] = s1.astype(np.float32)
            data["scale_2"] = s2.astype(np.float32)

        # --- (d) Remove very low-opacity Gaussians ---
        if has_opacity:
            opacities_logit = data["opacity"].astype(np.float64)
            real_opacity = 1.0 / (1.0 + np.exp(-opacities_logit))
            keep_mask = real_opacity > 0.01  # Remove splats with < 1% opacity
            n_removed = int(np.sum(~keep_mask))
            if n_removed > 0:
                data = data[keep_mask]
                print(f"🗑️  Removed {n_removed:,} near-invisible Gaussians (opacity < 1%)")

        n_after_clean = len(data)
        print(f"📊 Gaussians after cleanup: {n_after_clean:,} (was {n_original:,})")

        # --- (e) Trim to max_size_mb ---
        raw = ply_path.read_bytes()
        header_end = raw.index(b"end_header") + len(b"end_header\n")
        data_size = len(raw) - header_end
        bytes_per_vert = data_size / n_original if n_original > 0 else 68
        estimated_size_mb = (header_end + len(data) * bytes_per_vert) / (1024 * 1024)

        if estimated_size_mb > max_size_mb and has_opacity:
            max_verts = int((max_size_mb * 1024 * 1024 - header_end - 100) / bytes_per_vert)
            max_verts = min(max_verts, len(data))
            opacities_logit = data["opacity"].astype(np.float64)
            top_indices = np.argsort(opacities_logit)[::-1][:max_verts]
            top_indices = np.sort(top_indices)
            data = data[top_indices]
            print(f"⚠️  Trimming PLY: {n_after_clean:,} → {len(data):,} vertices "
                  f"({estimated_size_mb:.1f} MB → ~{max_size_mb} MB)")

        # Write cleaned PLY
        new_elem = PlyElement.describe(data, "vertex")
        new_ply = PlyData([new_elem], text=False)
        buf = io.BytesIO()
        new_ply.write(buf)
        return buf.getvalue()

    # ===================================================================
    # Main processing pipeline
    # ===================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        source_path = tmpdir_path / "scene"
        image_dir = source_path / "images"
        model_path = tmpdir_path / "output"

        image_dir.mkdir(parents=True)
        model_path.mkdir(parents=True)

        # ------------------------------------------------------------------
        # 1) Save images to disk
        # ------------------------------------------------------------------
        n_views = 0
        is_single_image = len(image_bytes_list) == 1

        if is_single_image:
            # Single image → create 3 perspective-shifted synthetic views
            pil_img = Image.open(io.BytesIO(image_bytes_list[0])).convert("RGB")
            w, h = pil_img.size
            print(f"🖼️  Single image: {filenames[0]} — {w}×{h}")

            views = create_synthetic_views(pil_img, n_views=3)
            for i, view in enumerate(views):
                view_path = image_dir / f"view_{i:03d}.jpg"
                view.save(str(view_path), quality=95)
            n_views = len(views)
            print(f"📐 Created {n_views} synthetic perspective views (±8° rotation)")
        else:
            # Multiple images → save each directly
            for idx, (img_bytes, fname) in enumerate(zip(image_bytes_list, filenames)):
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                w, h = pil_img.size
                print(f"🖼️  Image {idx}: {fname} — {w}×{h}")
                save_path = image_dir / f"image_{idx:03d}.jpg"
                pil_img.save(str(save_path), quality=95)
            n_views = len(image_bytes_list)

        print(f"📐 InstantSplat++ processing with {n_views} views")

        # ------------------------------------------------------------------
        # 2) Geometry initialization (MASt3R camera pose + point cloud)
        # ------------------------------------------------------------------
        print("🔧 Step 1/2: Geometry initialization with MASt3R...")
        init_cmd = [
            sys.executable, "-W", "ignore", "/opt/instantsplat/init_geo.py",
            "-s", str(source_path),
            "-m", str(model_path),
            "--n_views", str(n_views),
            "--focal_avg",
            "--co_vis_dsp",
            "--conf_aware_ranking",
            "--infer_video",
        ]

        init_result = subprocess.run(
            init_cmd,
            cwd="/opt/instantsplat",
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        print(f"init_geo stdout (last 2000 chars):\n{init_result.stdout[-2000:]}")
        if init_result.returncode != 0:
            print(f"init_geo stderr:\n{init_result.stderr[-3000:]}")
            raise RuntimeError(
                f"Geometry initialization failed (exit {init_result.returncode}): "
                f"{init_result.stderr[-500:]}"
            )
        print("✅ Geometry initialization complete")

        # ------------------------------------------------------------------
        # 3) Train 3D Gaussian Splatting (with pruning enabled)
        # ------------------------------------------------------------------
        n_iters = 2000
        print(f"🔧 Step 2/2: Training 3D Gaussian Splatting ({n_iters} iters, pruning ON)...")
        train_cmd = [
            sys.executable, "/opt/instantsplat/train.py",
            "-s", str(source_path),
            "-m", str(model_path),
            "-r", "1",
            "--n_views", str(n_views),
            "--iterations", str(n_iters),
            "--pp_optimizer",
            "--optim_pose",
            "--sh_degree", "0",                    # DC-only → compact PLY
            "--save_iterations", str(n_iters),
        ]

        train_result = subprocess.run(
            train_cmd,
            cwd="/opt/instantsplat",
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        print(f"train stdout (last 2000 chars):\n{train_result.stdout[-2000:]}")
        if train_result.returncode != 0:
            print(f"train stderr:\n{train_result.stderr[-3000:]}")
            raise RuntimeError(
                f"Training failed (exit {train_result.returncode}): "
                f"{train_result.stderr[-500:]}"
            )
        print("✅ Training complete")

        # ------------------------------------------------------------------
        # 4) Read and return the output PLY
        # ------------------------------------------------------------------
        ply_path = model_path / "point_cloud" / f"iteration_{n_iters}" / "point_cloud.ply"

        if not ply_path.exists():
            # Search for any Gaussian PLY file in the output tree
            ply_files = sorted(model_path.glob("**/point_cloud.ply"))
            if ply_files:
                ply_path = ply_files[-1]
                print(f"⚠️  Using fallback PLY: {ply_path}")
            else:
                all_files = list(model_path.rglob("*"))
                print(f"📁 Output tree ({len(all_files)} files):")
                for f in all_files[:50]:
                    print(f"   {f.relative_to(model_path)}")
                raise RuntimeError("InstantSplat++ did not produce a PLY file")

        raw_size_mb = ply_path.stat().st_size / (1024 * 1024)
        print(f"📦 Raw PLY: {ply_path.stat().st_size:,} bytes ({raw_size_mb:.1f} MB)")

        ply_bytes = clean_gaussian_ply(ply_path, max_size_mb=4.0)

        final_size_mb = len(ply_bytes) / (1024 * 1024)
        print(
            f"✅ InstantSplat++ PLY: {len(ply_bytes):,} bytes ({final_size_mb:.1f} MB), "
            f"DC-only SH, co-vis downsampled, pruning-cleaned"
        )
        return ply_bytes


@app.function(image=image, gpu="A100", timeout=900, volumes={"/cache": volume})
@modal.fastapi_endpoint(method="POST")
async def instantplus_router(request: dict) -> dict:
    """
    Single web endpoint that multiplexes:
    - op = "process" (default): start InstantSplat++ job (sync or async)
    - op = "status": get status for an async job
    - op = "health": simple health check
    """
    import base64
    from modal.functions import FunctionCall

    try:
        op = request.get("op") or "process"

        if op == "health":
            return {"status": "ok", "service": "instantplus", "endpoint": "router"}

        if op == "status":
            call_id = request.get("call_id")
            if not call_id:
                return {"error": "call_id required"}

            call = FunctionCall.from_id(call_id)
            try:
                ply_bytes = call.get(timeout=0)
                ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
                return {"status": "completed", "ply": ply_b64}
            except TimeoutError:
                return {"status": "processing"}
            except Exception as e:
                return {"status": "failed", "error": str(e)}

        # Default: process new image(s)
        is_async = request.get("async", False)

        # Collect images into lists
        images_b64: list[str] = []
        filenames: list[str] = []

        if request.get("images"):
            for item in request["images"]:
                images_b64.append(item["image"])
                filenames.append(item.get("filename", f"image_{len(filenames)}.jpg"))
        elif request.get("image"):
            images_b64.append(request["image"])
            filenames.append(request.get("filename", "image.jpg"))
        else:
            return {"error": "No image provided"}

        print(
            f"🔄 InstantSplat++ process: {len(images_b64)} image(s), async={is_async}, "
            f"filenames={filenames}"
        )

        image_bytes_list = [base64.b64decode(b) for b in images_b64]

        if is_async:
            call = process_images.spawn(image_bytes_list, filenames)
            return {"success": True, "call_id": call.object_id, "status": "processing"}

        ply_bytes = process_images.remote(image_bytes_list, filenames)
        ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
        return {"success": True, "ply": ply_b64}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


@app.local_entrypoint()
def main():
    """
    Local CLI helper:

        modal run modal_app.py -- <image_path> [<image_path2> ...]
    """
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: modal run modal_app.py -- <image_path> [<image_path2> ...]")
        return

    image_paths = [Path(p) for p in sys.argv[1:]]
    for p in image_paths:
        if not p.exists():
            print(f"Image not found: {p}")
            return

    image_bytes_list = []
    filenames = []
    for p in image_paths:
        with p.open("rb") as f:
            image_bytes_list.append(f.read())
        filenames.append(p.name)

    print(f"Processing {len(image_paths)} image(s) with InstantSplat++...")
    ply_bytes = process_images.remote(image_bytes_list, filenames)

    output_path = image_paths[0].parent / f"{image_paths[0].stem}_instantsplat.ply"
    with output_path.open("wb") as f:
        f.write(ply_bytes)

    print(f"Saved InstantSplat++ PLY to {output_path}")
