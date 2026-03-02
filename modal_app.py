"""
Modal app for running AnySplat on GPU.
AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views
Project page: https://city-super.github.io/anysplat/
Code: https://github.com/InternRobotics/AnySplat
Hugging Face: https://huggingface.co/lhjiang/anysplat

Deploy with: modal deploy modal_app.py
"""

import modal

# Create the Modal app
app = modal.App("anysplat")

# AnySplat is developed for Python 3.10, PyTorch 2.2.0 and CUDA 12.1.
# We start from NVIDIA's PyTorch image, then replace the preinstalled torch
# with the exact 2.2.0/cu121 stack and install AnySplat + its dependencies.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.07-py3",  # PyTorch 2.4.0 + CUDA 12.5, Python 3.10
        add_python=None,
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "8.0",  # Optimise for A100
            "DEBIAN_FRONTEND": "noninteractive",
        }
    )
    .run_commands(
        # System dependencies: git + FFmpeg + basic GL for OpenCV / Open3D + build tools
        "apt-get update && apt-get install -y "
        "git ffmpeg libgl1-mesa-glx libglib2.0-0 "
        "build-essential ninja-build cmake "
        "&& rm -rf /var/lib/apt/lists/*",
        # Clone AnySplat
        "git clone https://github.com/InternRobotics/AnySplat.git /opt/anysplat",
        # ── Phase 1: Remove the base-image torch and install the exact CUDA 12.1
        #    build that AnySplat was developed against.  This MUST happen BEFORE
        #    requirements.txt so that pytorch3d / torch_scatter / gsplat compile
        #    against the right torch ABI.
        "pip uninstall -y torch torchvision torchaudio torch-tensorrt || true",
        "pip install --no-cache-dir "
        "torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",

        # ── Phase 2: Install AnySplat requirements.  pytorch3d, torch_scatter,
        #    gsplat etc. will now be compiled / resolved against torch 2.2.0+cu121.
        "pip install --no-cache-dir -r /opt/anysplat/requirements.txt",

        # ── Phase 3 (removed): Do NOT force-reinstall torch after requirements.txt.
        #    requirements.txt does not overwrite torch, and force-reinstalling would
        #    replace the NVIDIA CUDA shared libraries that gsplat was linked against,
        #    causing "Not compiled with CUDA support" at runtime.

        # ── Phase 4: NUCLEAR FIX for OpenCV (NVIDIA base image conflict)
        "pip uninstall -y opencv-python opencv-python-headless "
        "opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true",
        "find /usr -name 'cv2*' -exec rm -rf {} + 2>/dev/null || true",
        "find /usr -name 'opencv*' -path '*/dist-packages/*' -exec rm -rf {} + 2>/dev/null || true",
        "pip install --no-cache-dir opencv-python-headless==4.8.0.76",

        # ── Phase 5: Pin numpy < 2 (must be LAST to override any 2.x)
        "pip install --no-cache-dir 'numpy<2'",

        # ── Phase 6: Verify torch sees CUDA and gsplat imports cleanly
        'python -c "import torch; '
        "print(f'torch={torch.__version__}  cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.is_available()}'); "
        "assert torch.version.cuda is not None, 'torch has NO CUDA support!'; "
        "import gsplat; print(f'gsplat={gsplat.__version__}')\"",
    )
)

# Volume for caching Hugging Face weights and AnySplat assets
volume = modal.Volume.from_name("anysplat-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=900,  # 15 minutes is plenty for feed-forward AnySplat
    volumes={"/cache": volume},
)
def process_image(image_bytes: bytes, filename: str, prompt: str = "", elevation: int = 20) -> bytes:
    """
    Process a single image with AnySplat and return a PLY file with 3D Gaussians.

    - Saves the uploaded image to a temporary folder.
    - Runs AnySplat in feed-forward mode to predict 3D Gaussians + poses.
    - Exports a Gaussian splat as a .ply using AnySplat's own export utility.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    import torch

    # Route heavy downloads through the shared volume
    os.environ["TORCH_HOME"] = "/cache/torch"
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/cache/huggingface/datasets"

    # Add AnySplat to Python path
    sys.path.insert(0, "/opt/anysplat")

    from src.model.model.anysplat import AnySplat  # type: ignore
    from src.utils.image import process_image as preprocess_image  # type: ignore
    from src.model.ply_export import export_ply  # type: ignore

    # Cache the model at module level to avoid re-loading on warm containers
    global _ANYSPLAT_MODEL  # type: ignore
    try:
        model = _ANYSPLAT_MODEL  # type: ignore[name-defined]
    except NameError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AnySplat.from_pretrained("lhjiang/anysplat")
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _ANYSPLAT_MODEL = model  # type: ignore

    device = next(model.parameters()).device  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Save uploaded bytes
        input_path = tmpdir_path / filename
        input_path.write_bytes(image_bytes)

        print(f"🔄 Starting AnySplat processing for {filename} ({len(image_bytes)} bytes)...")

        # AnySplat requires at least 2 views.  When we only have one image we
        # duplicate it (exactly like the official HuggingFace Space does).
        img_tensor = preprocess_image(str(input_path))
        views = [img_tensor, img_tensor]  # duplicate for 2-view minimum
        images = torch.stack(views, dim=0).unsqueeze(0).to(device)  # [1, K=2, 3, 448, 448]
        b, v, _, h, w = images.shape
        print(f"📐 AnySplat input tensor shape: {images.shape} (batch={b}, views={v}, H={h}, W={w})")

        # Run inference
        with torch.no_grad():
            gaussians, pred_context_pose = model.inference((images + 1) * 0.5)  # type: ignore[attr-defined]

        # Export to PLY using AnySplat's helper (mirrors inference.py)
        ply_path = tmpdir_path / "gaussians.ply"
        export_ply(
            gaussians.means[0],
            gaussians.scales[0],
            gaussians.rotations[0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            ply_path,
        )

        if not ply_path.exists():
            raise RuntimeError(f"AnySplat did not produce a PLY file at {ply_path}")

        print(f"✅ AnySplat generated PLY: {ply_path} ({ply_path.stat().st_size} bytes)")
        return ply_path.read_bytes()


@app.function(image=image, gpu="A100", timeout=900, volumes={"/cache": volume})
@modal.fastapi_endpoint(method="POST")
async def anysplat_router(request: dict) -> dict:
    """
    Single web endpoint that multiplexes:
    - op = \"process\" (default): start AnySplat job (sync or async)
    - op = \"status\": get status for an async job
    - op = \"health\": simple health check
    """
    import base64
    from modal.functions import FunctionCall

    try:
        op = request.get("op") or "process"

        if op == "health":
            return {"status": "ok", "service": "anysplat", "endpoint": "router"}

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

        # Default: process a new image
        print(
            f"🔄 Received AnySplat process request: async={request.get('async', False)}, "
            f"filename={request.get('filename', 'N/A')}"
        )

        image_b64 = request.get("image")
        filename = request.get("filename", "image.jpg")
        prompt = request.get("prompt", "")
        elevation = request.get("elevation", 20)  # kept for API compatibility, unused
        is_async = request.get("async", False)

        if not image_b64:
            return {"error": "No image provided"}

        image_bytes = base64.b64decode(image_b64)

        if is_async:
            call = process_image.spawn(image_bytes, filename, prompt, elevation)
            return {"success": True, "call_id": call.object_id, "status": "processing"}

        ply_bytes = process_image.remote(image_bytes, filename, prompt, elevation)
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

        modal run modal_app.py -- <image_path>
    """
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: modal run modal_app.py -- <image_path>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    with image_path.open("rb") as f:
        image_bytes = f.read()

    print(f"Processing {image_path} with AnySplat...")
    ply_bytes = process_image.remote(image_bytes, image_path.name)

    output_path = image_path.with_suffix(".ply")
    with output_path.open("wb") as f:
        f.write(ply_bytes)

    print(f"Saved AnySplat PLY to {output_path}")
