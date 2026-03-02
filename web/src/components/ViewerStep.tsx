"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  Maximize2,
  Download,
  AlertCircle,
  Settings2,
  RotateCcw,
  ChevronDown,
  Cpu,
  Hand,
  ZoomIn,
  MousePointer2,
  Wand2,
  Loader2,
  Zap,
} from "lucide-react";
import type { ProcessedResult } from "@/app/page";

interface ViewerStepProps {
  result: ProcessedResult;
  onBack: () => void;
}

// Blend modes
type BlendMode =
  | "normal"
  | "multiply"
  | "screen"
  | "overlay"
  | "soft-light"
  | "hard-light"
  | "color-dodge"
  | "color-burn"
  | "difference"
  | "exclusion"
  | "hue"
  | "saturation"
  | "color"
  | "luminosity";

// Color filter presets
type ColorFilter = "none" | "warm" | "cool" | "vintage" | "noir" | "vivid" | "muted" | "cinematic" | "sunset" | "forest" | "ocean" | "sepia" | "cyberpunk" | "dreamy" | "gameboy" | "retro" | "neon" | "pastel" | "monochrome" | "polaroid" | "lomo" | "hdr";

interface Settings {
  // Splat Controls (real GaussianSplats3D API)
  splatScale: number;        // splatMesh.setSplatScale() — per-splat rendering scale
  focalAdjustment: number;   // viewer.focalAdjustment — focal length multiplier
  maxSplatSize: number;      // viewer.maxScreenSpaceSplatSize
  sceneOpacity: number;      // scene opacity via enableOptionalEffects uniform
  pointCloudMode: boolean;   // splatMesh.setPointCloudModeEnabled()
  splatThreshold: number;    // splatAlphaRemovalThreshold (load-time)
  // Camera
  fov: number;
  nearClip: number;
  farClip: number;
  exposure: number;
  // Color Grading
  brightness: number;
  contrast: number;
  saturation: number;
  hue: number;
  temperature: number;
  tint: number;
  shadows: number;
  midtones: number;
  highlights: number;
  colorFilter: ColorFilter;
  // Blend
  blendMode: BlendMode;
  // Post FX
  grain: number;
  bloom: number;
  bloomThreshold: number;
  bloomRadius: number;
  vignette: number;
  vignetteRoundness: number;
  chromaticAberration: number;
  sharpen: number;
  blur: number;
  pixelate: number;
  scanlines: number;
  noise: number;
  glitch: number;
  // Atmosphere
  fogEnabled: boolean;
  fogDensity: number;
  fogColor: string;
  // Background
  bgColor: string;
  // Enhancement
  autoEnhance: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  // Splat Controls (real GS3D API)
  splatScale: 1.0,
  focalAdjustment: 1.0,
  maxSplatSize: 1024,
  sceneOpacity: 1.0,
  pointCloudMode: false,
  splatThreshold: 1,
  // Camera
  fov: 60,
  nearClip: 0.1,
  farClip: 1000,
  exposure: 1.0,
  // Color Grading
  brightness: 1.0,
  contrast: 1.0,
  saturation: 1.0,
  hue: 0,
  temperature: 0,
  tint: 0,
  shadows: 0,
  midtones: 0,
  highlights: 0,
  colorFilter: "none",
  blendMode: "normal",
  // Post FX
  grain: 0,
  bloom: 0,
  bloomThreshold: 0.8,
  bloomRadius: 0.5,
  vignette: 0,
  vignetteRoundness: 0.5,
  chromaticAberration: 0,
  sharpen: 0,
  blur: 0,
  pixelate: 0,
  scanlines: 0,
  noise: 0,
  glitch: 0,
  // Atmosphere
  fogEnabled: false,
  fogDensity: 0.5,
  fogColor: "#808080",
  // Background
  bgColor: "#f5f5f5",
  autoEnhance: false,
};

// Optimized preset for better scene quality
const ENHANCED_SETTINGS: Partial<Settings> = {
  brightness: 1.05,
  contrast: 1.1,
  saturation: 1.15,
  shadows: 0.1,
  highlights: -0.05,
  sharpen: 0.3,
  bloom: 0.15,
  bloomThreshold: 0.7,
  vignette: 0.2,
  exposure: 1.05,
  splatScale: 1.1,
  focalAdjustment: 1.1,
};

const BLEND_MODES: { value: BlendMode; label: string }[] = [
  { value: "normal", label: "Normal" },
  { value: "multiply", label: "Multiply" },
  { value: "screen", label: "Screen" },
  { value: "overlay", label: "Overlay" },
  { value: "soft-light", label: "Soft Light" },
  { value: "hard-light", label: "Hard Light" },
  { value: "color-dodge", label: "Color Dodge" },
  { value: "color-burn", label: "Color Burn" },
  { value: "difference", label: "Difference" },
  { value: "exclusion", label: "Exclusion" },
  { value: "hue", label: "Hue" },
  { value: "saturation", label: "Saturation" },
  { value: "color", label: "Color Only" },
  { value: "luminosity", label: "Luminance Only" },
];

const COLOR_FILTERS: { value: ColorFilter; label: string; preview: string }[] = [
  { value: "none", label: "None", preview: "linear-gradient(135deg, #fff 0%, #f0f0f0 100%)" },
  { value: "warm", label: "Warm", preview: "linear-gradient(135deg, #ff9966 0%, #ff5e62 100%)" },
  { value: "cool", label: "Cool", preview: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" },
  { value: "vintage", label: "Vintage", preview: "linear-gradient(135deg, #d4a574 0%, #c4956a 100%)" },
  { value: "noir", label: "Noir", preview: "linear-gradient(135deg, #2c2c2c 0%, #1a1a1a 100%)" },
  { value: "vivid", label: "Vivid", preview: "linear-gradient(135deg, #ff0080 0%, #00ff80 100%)" },
  { value: "muted", label: "Muted", preview: "linear-gradient(135deg, #a8a8a8 0%, #8f8f8f 100%)" },
  { value: "cinematic", label: "Cinematic", preview: "linear-gradient(135deg, #1a3a4a 0%, #2d1b3d 100%)" },
  { value: "sunset", label: "Sunset", preview: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)" },
  { value: "forest", label: "Forest", preview: "linear-gradient(135deg, #134e5e 0%, #71b280 100%)" },
  { value: "ocean", label: "Ocean", preview: "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)" },
  { value: "sepia", label: "Sepia", preview: "linear-gradient(135deg, #8b7355 0%, #704214 100%)" },
  { value: "cyberpunk", label: "Cyberpunk", preview: "linear-gradient(135deg, #ff00ff 0%, #00ffff 100%)" },
  { value: "dreamy", label: "Dreamy", preview: "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)" },
  { value: "gameboy", label: "Game Boy", preview: "linear-gradient(135deg, #9bbc0f 0%, #8bac0f 100%)" },
  { value: "retro", label: "Retro", preview: "linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%)" },
  { value: "neon", label: "Neon", preview: "linear-gradient(135deg, #ff00ff 0%, #00ffff 50%, #ff00ff 100%)" },
  { value: "pastel", label: "Pastel", preview: "linear-gradient(135deg, #ffd6e8 0%, #c8e6ff 100%)" },
  { value: "monochrome", label: "Monochrome", preview: "linear-gradient(135deg, #808080 0%, #404040 100%)" },
  { value: "polaroid", label: "Polaroid", preview: "linear-gradient(135deg, #f5f5dc 0%, #e8e8d3 100%)" },
  { value: "lomo", label: "Lomo", preview: "linear-gradient(135deg, #2c1810 0%, #1a0f08 100%)" },
  { value: "hdr", label: "HDR", preview: "linear-gradient(135deg, #ffffff 0%, #000000 100%)" },
];

const BG_COLORS = [
  { color: "#f5f5f5", label: "Light" },
  { color: "#1a1a1a", label: "Dark" },
  { color: "#0a0a0a", label: "Black" },
  { color: "#ffffff", label: "White" },
  { color: "#1e3a5f", label: "Navy" },
  { color: "#2d3436", label: "Charcoal" },
  { color: "#2c1810", label: "Espresso" },
  { color: "#0f1419", label: "Midnight" },
];

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

function base64ToBlobUrl(base64: string): string {
  const arrayBuffer = base64ToArrayBuffer(base64);
  const blob = new Blob([arrayBuffer], { type: "application/octet-stream" });
  return URL.createObjectURL(blob);
}

// Collapsible Section - all open by default
function Section({
  title,
  children,
  defaultOpen = true,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-5 hover:bg-neutral-50 transition-colors"
      >
        <span className="text-xs font-semibold text-foreground uppercase tracking-wide">
          {title}
        </span>
        <ChevronDown
          className={`w-4 h-4 text-muted transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-4 space-y-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit = "",
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
  format?: (v: number) => string;
}) {
  const displayValue = format ? format(value) : value.toFixed(step < 1 ? 2 : 0);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted">{label}</span>
        <span className="text-xs text-muted tabular-nums">
          {displayValue}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-foreground h-1.5"
      />
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
  description,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  description?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <span className="text-sm text-foreground">{label}</span>
        {description && (
          <p className="text-xs text-muted mt-0.5">{description}</p>
        )}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-5 rounded-full transition-colors ${
          checked ? "bg-foreground" : "bg-neutral-300"
        }`}
      >
        <div
          className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
            checked ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </button>
    </div>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2">
      <span className="text-sm text-muted">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-border bg-white text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-foreground/20"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type GS3DViewer = any; // The real GaussianSplats3D.Viewer - we use 'any' to access internal properties

export default function ViewerStep({ result, onBack }: ViewerStepProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const grainCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [viewerError, setViewerError] = useState<string | null>(null);
  // Sidebar: collapsed by default on mobile, open by default on desktop
  const [showSettings, setShowSettings] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const viewerInstanceRef = useRef<GS3DViewer | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const grainAnimationRef = useRef<number | null>(null);
  const [isEnhancing, setIsEnhancing] = useState(false);

  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);

  // Get color filter CSS
  const getColorFilterCSS = useCallback((filter: ColorFilter): string => {
    switch (filter) {
      case "warm": return "sepia(0.2) saturate(1.3) hue-rotate(-10deg)";
      case "cool": return "saturate(0.9) hue-rotate(10deg) brightness(1.05)";
      case "vintage": return "sepia(0.4) contrast(1.1) brightness(0.95) saturate(0.8)";
      case "noir": return "grayscale(1) contrast(1.3) brightness(0.9)";
      case "vivid": return "saturate(1.8) contrast(1.15)";
      case "muted": return "saturate(0.6) contrast(0.95)";
      case "cinematic": return "contrast(1.2) saturate(0.85) brightness(0.95)";
      case "sunset": return "sepia(0.15) saturate(1.4) hue-rotate(-15deg) brightness(1.05)";
      case "forest": return "saturate(1.1) hue-rotate(30deg) brightness(0.95)";
      case "ocean": return "saturate(1.2) hue-rotate(180deg) brightness(1.05)";
      case "sepia": return "sepia(0.8) contrast(1.1) brightness(0.95)";
      case "cyberpunk": return "saturate(1.5) contrast(1.2) hue-rotate(300deg) brightness(1.1)";
      case "dreamy": return "saturate(0.8) brightness(1.1) contrast(0.9) blur(0.5px)";
      case "gameboy": return "contrast(1.5) saturate(0.3) brightness(0.9) hue-rotate(90deg)";
      case "retro": return "saturate(1.4) contrast(1.2) hue-rotate(15deg) brightness(1.05)";
      case "neon": return "saturate(2) contrast(1.3) brightness(1.2) hue-rotate(180deg)";
      case "pastel": return "saturate(0.5) brightness(1.15) contrast(0.9)";
      case "monochrome": return "grayscale(1) contrast(1.1)";
      case "polaroid": return "sepia(0.15) contrast(1.05) brightness(1.05) saturate(0.9)";
      case "lomo": return "contrast(1.3) saturate(1.2) brightness(0.9) sepia(0.1)";
      case "hdr": return "contrast(1.4) saturate(1.1) brightness(1.05)";
      default: return "";
    }
  }, []);

  const getCSSFilters = useCallback(() => {
    const { brightness, contrast, saturation, hue, blur, sharpen, exposure, colorFilter, temperature, tint } = settings;
    
    let filter = `brightness(${brightness * exposure}) contrast(${contrast}) saturate(${saturation})`;
    
    if (hue !== 0) {
      filter += ` hue-rotate(${hue}deg)`;
    }
    
    // Temperature (warm/cool shift)
    if (temperature !== 0) {
      filter += ` sepia(${Math.abs(temperature) * 0.2}) hue-rotate(${temperature > 0 ? -10 : 10}deg)`;
    }
    
    // Tint (green/magenta shift) - simulated
    if (tint !== 0) {
      filter += ` hue-rotate(${tint * 30}deg)`;
    }
    
    // Color filter preset
    const filterCSS = getColorFilterCSS(colorFilter);
    if (filterCSS) {
      filter += ` ${filterCSS}`;
    }
    
    // Blur
    if (blur > 0) {
      filter += ` blur(${blur * 5}px)`;
    }
    
    // Sharpen is simulated via contrast boost
    if (sharpen > 0) {
      filter += ` contrast(${1 + sharpen * 0.2})`;
    }
    
    // Bloom glow effect
    if (settings.bloom > 0) {
      filter += ` drop-shadow(0 0 ${settings.bloom * 20}px rgba(255,255,255,${settings.bloom * 0.5}))`;
    }
    
    return filter;
  }, [settings, getColorFilterCSS]);

  // Grain animation
  useEffect(() => {
    const grainIntensity = settings.grain + settings.noise;
    if (grainIntensity <= 0 || !grainCanvasRef.current) {
      if (grainAnimationRef.current) {
        cancelAnimationFrame(grainAnimationRef.current);
        grainAnimationRef.current = null;
      }
      return;
    }

    const canvas = grainCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const animate = () => {
      const width = canvas.width;
      const height = canvas.height;
      const imageData = ctx.createImageData(width, height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const noise = Math.random() * 255;
        data[i] = noise;
        data[i + 1] = noise;
        data[i + 2] = noise;
        data[i + 3] = Math.floor(grainIntensity * 60);
      }
      ctx.putImageData(imageData, 0, 0);
      grainAnimationRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      if (grainAnimationRef.current) cancelAnimationFrame(grainAnimationRef.current);
    };
  }, [settings.grain, settings.noise]);

  // Mobile detection and sidebar initial state
  useEffect(() => {
    let isInitialMount = true;
    
    const checkMobile = () => {
      const mobile = window.innerWidth < 640; // sm breakpoint
      setIsMobile(mobile);
      // Only set initial state on mount, not on resize
      // On desktop (sm and above), show sidebar by default
      // On mobile, keep it collapsed (false)
      if (isInitialMount) {
        setShowSettings(!mobile);
        isInitialMount = false;
      }
    };
    
    const handleResize = () => {
      setIsMobile(window.innerWidth < 640);
    };
    
    checkMobile();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Resize grain canvas
  useEffect(() => {
    const handleResize = () => {
      if (grainCanvasRef.current && containerRef.current) {
        grainCanvasRef.current.width = containerRef.current.clientWidth / 4;
        grainCanvasRef.current.height = containerRef.current.clientHeight / 4;
      }
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Apply CSS effects
  useEffect(() => {
    if (canvasRef.current) {
      canvasRef.current.style.filter = getCSSFilters();
      canvasRef.current.style.mixBlendMode = settings.blendMode;
      
      // Pixelate effect
      if (settings.pixelate > 0) {
        canvasRef.current.style.imageRendering = "pixelated";
        canvasRef.current.style.transform = `scale(${1 + settings.pixelate * 0.1})`;
      } else {
        canvasRef.current.style.imageRendering = "auto";
        canvasRef.current.style.transform = "scale(1)";
      }
    }
  }, [getCSSFilters, settings.blendMode, settings.pixelate]);

  // Camera settings — the GS3D viewer exposes a Three.js PerspectiveCamera
  useEffect(() => {
    const viewer = viewerInstanceRef.current;
    const camera = viewer?.camera || viewer?.perspectiveCamera;
    if (camera) {
      camera.fov = settings.fov;
      camera.near = settings.nearClip;
      camera.far = settings.farClip;
      camera.updateProjectionMatrix?.();
    }
  }, [settings.fov, settings.nearClip, settings.farClip]);

  // Splat controls — wired to the real GaussianSplats3D Viewer API
  useEffect(() => {
    const viewer = viewerInstanceRef.current;
    if (!viewer) return;

    // (1) Splat Scale — via splatMesh.setSplatScale() which sets a shader uniform
    //     This is the '=' / '-' keyboard shortcut internally
    if (viewer.splatMesh?.setSplatScale) {
      viewer.splatMesh.setSplatScale(settings.splatScale);
    }

    // (2) Focal Adjustment — affects apparent splat size via focal length multiplier
    viewer.focalAdjustment = settings.focalAdjustment;

    // (3) Max Screen-Space Splat Size — caps how large a splat can appear on screen
    viewer.maxScreenSpaceSplatSize = settings.maxSplatSize;

    // (4) Scene Opacity — via enableOptionalEffects uniform
    if (viewer.splatMesh?.material?.uniforms?.sceneOpacity) {
      const opacityArray = viewer.splatMesh.material.uniforms.sceneOpacity.value;
      if (opacityArray && opacityArray.length > 0) {
        opacityArray[0] = settings.sceneOpacity;
        viewer.splatMesh.material.uniformsNeedUpdate = true;
      }
    }

    // (5) Point Cloud Mode — renders splats as small dots
    if (viewer.splatMesh?.setPointCloudModeEnabled) {
      viewer.splatMesh.setPointCloudModeEnabled(settings.pointCloudMode);
    }

    // Force a re-render after property changes
    viewer.forceRenderNextFrame?.();
  }, [settings.splatScale, settings.focalAdjustment, settings.maxSplatSize, settings.sceneOpacity, settings.pointCloudMode]);

  // Background color
  useEffect(() => {
    const viewer = viewerInstanceRef.current;
    const renderer = viewer?.renderer;
    if (renderer) {
      const color = parseInt(settings.bgColor.replace("#", ""), 16);
      renderer.setClearColor?.(color, 1);
    }
  }, [settings.bgColor]);

  // Initialize viewer
  useEffect(() => {
    if (!containerRef.current) return;
    if (!result.splatUrl && !result.plyBase64) return;

    let mounted = true;

    const initViewer = async () => {
      try {
        setIsLoading(true);
        setLoadingProgress(0);
        setViewerError(null);

        const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");

        if (!mounted || !containerRef.current) return;
        containerRef.current.innerHTML = "";

        const viewer = new GaussianSplats3D.Viewer({
          cameraUp: [0, -1, 0],
          initialCameraPosition: [0, 0, 0],
          initialCameraLookAt: [0, 0, 5],
          rootElement: containerRef.current,
          selfDrivenMode: true,
          useBuiltInControls: true,
          dynamicScene: true,  // Allow runtime scale/position changes on splatMesh
          sharedMemoryForWorkers: false,
          antialiased: false,
          logLevel: GaussianSplats3D.LogLevel.None,
          splatSortDistanceMapPrecision: 16,
          sphericalHarmonicsDegree: 0,
          integerBasedSort: true,
          halfPrecisionCovariancesOnGPU: true,
          enableOptionalEffects: true,   // Allow runtime opacity & effects
          focalAdjustment: 1.0,          // Controls apparent splat size
          kernel2DSize: 0.3,             // 2D screen-space kernel constant
        });

        viewerInstanceRef.current = viewer;

        let splatPath: string;

        if (result.plyBase64) {
          const arrayBuffer = base64ToArrayBuffer(result.plyBase64);
          const blob = new Blob([arrayBuffer], { type: "application/octet-stream" });
          splatPath = URL.createObjectURL(blob);
          blobUrlRef.current = splatPath;

          await viewer.addSplatScene(splatPath, {
            splatAlphaRemovalThreshold: settings.splatThreshold,
            showLoadingUI: false,
            progressiveLoad: true,
            format: GaussianSplats3D.SceneFormat.Ply,
            onProgress: (progress: number) => {
              if (mounted) setLoadingProgress(Math.min(progress * 100, 100));
            },
          });
        } else if (result.splatUrl) {
          splatPath = result.splatUrl;
          await viewer.addSplatScene(splatPath, {
            splatAlphaRemovalThreshold: settings.splatThreshold,
            showLoadingUI: false,
            progressiveLoad: true,
            onProgress: (progress: number) => {
              if (mounted) setLoadingProgress(Math.min(progress * 100, 100));
            },
          });
        }

        viewer.start();

        // Access internal properties of the GS3D viewer
        const gViewer = viewerInstanceRef.current;
        if (gViewer?.controls) {
          // Detect mobile device
          const isMobileDevice = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || 
                          (typeof window !== 'undefined' && window.innerWidth < 768);
          
          gViewer.controls.enablePan = true;
          gViewer.controls.panSpeed = isMobileDevice ? 0.8 : 1.5;
          gViewer.controls.rotateSpeed = isMobileDevice ? 0.35 : 1.0;
          gViewer.controls.zoomSpeed = isMobileDevice ? 0.8 : 1.2;
          gViewer.controls.mouseButtons = { LEFT: 0, MIDDLE: 2, RIGHT: 1 };
          
          if ('enableDamping' in gViewer.controls) {
            gViewer.controls.enableDamping = true;
            gViewer.controls.dampingFactor = isMobileDevice ? 0.2 : 0.05;
          }
          
          // Touch event handling for mobile
          let touchCleanup: (() => void) | null = null;
          if (isMobileDevice && gViewer.renderer?.domElement) {
            const canvas = gViewer.renderer.domElement;
            let isRotating = false;
            
            const handleTouchStart = (e: TouchEvent) => {
              if (e.touches.length === 1) {
                isRotating = true;
                const rect = canvas.getBoundingClientRect();
                const x = ((e.touches[0].clientX - rect.left) / rect.width) * 2 - 1;
                if (gViewer.controls) {
                  gViewer.controls.rotateSpeed = 0.35 * (1 + Math.abs(x) * 0.2);
                }
              }
            };
            
            const handleTouchMove = (e: TouchEvent) => {
              if (e.touches.length === 1 && isRotating) {
                e.preventDefault();
              }
            };
            
            const handleTouchEnd = () => {
              isRotating = false;
              if (gViewer.controls) {
                gViewer.controls.rotateSpeed = 0.35;
              }
            };
            
            canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
            canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
            canvas.addEventListener('touchend', handleTouchEnd);
            canvas.addEventListener('touchcancel', handleTouchEnd);
            
            touchCleanup = () => {
              canvas.removeEventListener('touchstart', handleTouchStart);
              canvas.removeEventListener('touchmove', handleTouchMove);
              canvas.removeEventListener('touchend', handleTouchEnd);
              canvas.removeEventListener('touchcancel', handleTouchEnd);
            };
          }
          
          if (touchCleanup) {
            gViewer.__touchCleanup = touchCleanup;
          }
        }

        // Get the canvas for CSS-based effects
        if (gViewer?.renderer?.domElement) {
          canvasRef.current = gViewer.renderer.domElement;
        }

        // Set background color
        if (gViewer?.renderer) {
          const color = parseInt(settings.bgColor.replace("#", ""), 16);
          gViewer.renderer.setClearColor(color, 1);
        }

        if (mounted) setIsLoading(false);
      } catch (error) {
        console.error("Error initializing viewer:", error);
        if (mounted) {
          setIsLoading(false);
          setViewerError(error instanceof Error ? error.message : "Failed to load 3D viewer");
        }
      }
    };

    const timer = setTimeout(initViewer, 100);

    return () => {
      mounted = false;
      clearTimeout(timer);
      if (viewerInstanceRef.current) {
        try {
          if (viewerInstanceRef.current.__touchCleanup) {
            viewerInstanceRef.current.__touchCleanup();
          }
          if (typeof viewerInstanceRef.current.dispose === "function") {
            viewerInstanceRef.current.dispose();
          }
        } catch (e) {
          console.error("Error disposing viewer:", e);
        }
        viewerInstanceRef.current = null;
      }
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [result.splatUrl, result.plyBase64, settings.bgColor]);

  // Enhance handler
  const handleEnhance = useCallback(async () => {
    setIsEnhancing(true);
    
    // Animate the enhancement
    await new Promise(resolve => setTimeout(resolve, 500));
    
    setSettings(prev => ({
      ...prev,
      ...ENHANCED_SETTINGS,
      autoEnhance: true,
    }));
    
    setIsEnhancing(false);
  }, []);

  const handleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      containerRef.current?.parentElement?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }, []);

  const handleDownload = useCallback(() => {
    let downloadUrl: string;
    let shouldRevoke = false;

    if (result.plyBase64) {
      downloadUrl = base64ToBlobUrl(result.plyBase64);
      shouldRevoke = true;
    } else if (result.splatUrl) {
      downloadUrl = result.splatUrl;
    } else {
      return;
    }

    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `${result.fileName}.ply`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    if (shouldRevoke) URL.revokeObjectURL(downloadUrl);
  }, [result.splatUrl, result.plyBase64, result.fileName]);

  const handleResetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
  }, []);

  const updateSetting = useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative flex items-center justify-between px-3 sm:px-5 py-3 sm:py-4 bg-card border-b border-border"
        >
          <div className="flex items-center gap-2 sm:gap-4 flex-1 min-w-0">
            <button onClick={onBack} className="icon-btn shrink-0">
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="h-5 w-px bg-border hidden sm:block" />
            <div className="flex items-center gap-2 shrink-0">
              <img
                src="https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/wordmark.svg"
                alt="Revelium Studio"
                className="h-4 sm:h-5 w-auto"
              />
              <span className="text-[10px] sm:text-xs font-medium text-muted opacity-80 border border-muted/30 rounded px-1.5 py-0.5">
                BETA
              </span>
            </div>
          </div>

          <h2 className="absolute left-1/2 transform -translate-x-1/2 text-xs sm:text-sm font-medium text-foreground truncate max-w-[200px] sm:max-w-[300px] hidden sm:block">
            {result.fileName}
          </h2>

        <div className="flex items-center gap-1 sm:gap-2 shrink-0">
          <button
            onClick={handleEnhance}
            disabled={isEnhancing || settings.autoEnhance}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              settings.autoEnhance 
                ? "bg-green-100 text-green-700 border border-green-200" 
                : "bg-black text-white border-black hover:bg-neutral-800"
            } disabled:opacity-50`}
          >
            {isEnhancing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : settings.autoEnhance ? (
              <Zap className="w-4 h-4" />
            ) : (
              <Wand2 className="w-4 h-4" />
            )}
            <span>{settings.autoEnhance ? "Enhanced" : "Enhance"}</span>
          </button>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`icon-btn ${showSettings ? "bg-black text-white border-black" : "bg-black text-white border-black"}`}
          >
            <Settings2 className="w-4 h-4" />
          </button>
          <button onClick={handleFullscreen} className="icon-btn">
            <Maximize2 className="w-4 h-4" />
          </button>
          <button onClick={handleDownload} className="icon-btn">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </motion.header>

      {/* Main viewer area */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* 3D Viewer Container */}
        <div className="flex-1 relative overflow-hidden" style={{ backgroundColor: settings.bgColor }}>
          <div ref={containerRef} className="absolute inset-0" />

          {(settings.grain > 0 || settings.noise > 0) && (
            <canvas ref={grainCanvasRef} className="absolute inset-0 pointer-events-none z-10 w-full h-full" style={{ mixBlendMode: "overlay", imageRendering: "pixelated" }} />
          )}

          {settings.bloom > 0 && (
            <div className="absolute inset-0 pointer-events-none z-10" style={{ background: `radial-gradient(ellipse at center, rgba(255,255,255,${settings.bloom * 0.15}) 0%, transparent 70%)`, mixBlendMode: "screen" }} />
          )}

          {settings.vignette > 0 && (
            <div className="absolute inset-0 pointer-events-none z-10" style={{ 
              background: `radial-gradient(ellipse at center, transparent 0%, transparent ${50 - settings.vignette * 25 * settings.vignetteRoundness}%, rgba(0,0,0,${settings.vignette * 0.8}) 100%)` 
            }} />
          )}

          {settings.chromaticAberration > 0 && (
            <div className="absolute inset-0 pointer-events-none z-10 overflow-hidden">
              <div className="absolute inset-0" style={{ background: `linear-gradient(90deg, rgba(255,0,0,${settings.chromaticAberration * 0.08}) 0%, transparent 5%, transparent 95%, rgba(0,255,255,${settings.chromaticAberration * 0.08}) 100%)`, mixBlendMode: "screen" }} />
            </div>
          )}
          
          {settings.scanlines > 0 && (
            <div 
              className="absolute inset-0 pointer-events-none z-10" 
              style={{ 
                background: `repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,${settings.scanlines * 0.3}) 2px, rgba(0,0,0,${settings.scanlines * 0.3}) 4px)`,
                mixBlendMode: "multiply"
              }} 
            />
          )}
          
          {settings.glitch > 0 && (
            <div 
              className="absolute inset-0 pointer-events-none z-10" 
              style={{ 
                background: `linear-gradient(90deg, rgba(255,0,255,${settings.glitch * 0.1}) 0%, transparent 10%, transparent 90%, rgba(0,255,255,${settings.glitch * 0.1}) 100%)`,
                mixBlendMode: "exclusion",
                animation: settings.glitch > 0.5 ? "glitch 0.3s infinite" : "none"
              }} 
            />
          )}

          {settings.fogEnabled && (
            <div className="absolute inset-0 pointer-events-none z-10" style={{ background: `linear-gradient(to top, ${settings.fogColor}${Math.round(settings.fogDensity * 200).toString(16).padStart(2, "0")} 0%, transparent 60%)` }} />
          )}

          {/* Scene Loading overlay */}
          {isLoading && !viewerError && (
            <div className="absolute inset-0 flex items-center justify-center bg-neutral-100 z-20">
              <div className="flex flex-col items-center gap-5 w-72">
                <div className="relative w-14 h-14">
                  <div className="absolute inset-0 rounded-full border-2 border-border" />
                  <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-foreground animate-spin" />
                </div>
                <div className="w-full">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-foreground font-medium">Loading scene</span>
                    <span className="text-sm text-muted">{Math.min(loadingProgress, 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-border rounded-full overflow-hidden">
                    <div className="h-full bg-foreground transition-all duration-300" style={{ width: `${Math.min(loadingProgress, 100)}%` }} />
                  </div>
                </div>
                <div className="flex items-center gap-2 text-muted text-xs">
                  <Cpu className="w-3.5 h-3.5" />
                  <span>WebGL Renderer</span>
                </div>
              </div>
            </div>
          )}

          {viewerError && (
            <div className="absolute inset-0 flex items-center justify-center bg-neutral-100 z-20">
              <div className="flex flex-col items-center gap-4 max-w-sm text-center p-8">
                <div className="w-14 h-14 rounded-full bg-red-50 flex items-center justify-center">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <h3 className="text-foreground font-medium">Failed to load viewer</h3>
                <p className="text-muted text-sm">{viewerError}</p>
                <button onClick={handleDownload} className="btn-primary flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm">
                  <Download className="w-4 h-4" />
                  <span>Download PLY</span>
                </button>
              </div>
            </div>
          )}

          {!isLoading && !viewerError && (
            <>
              {/* Desktop Controls - Bottom Left */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="hidden sm:flex absolute bottom-4 left-4 items-center gap-4 text-xs text-muted bg-card/80 backdrop-blur-sm px-4 py-2.5 rounded-xl border border-border z-20"
              >
                <div className="flex items-center gap-2">
                  <MousePointer2 className="w-3.5 h-3.5" />
                  <span>Left: Rotate</span>
                </div>
                <div className="w-px h-3 bg-border" />
                <div className="flex items-center gap-2">
                  <Hand className="w-3.5 h-3.5" />
                  <span>Middle: Pan</span>
                </div>
                <div className="w-px h-3 bg-border" />
                <div className="flex items-center gap-2">
                  <ZoomIn className="w-3.5 h-3.5" />
                  <span>Scroll: Zoom</span>
                </div>
              </motion.div>

              {/* Rotation Anchor Disclaimer - Desktop (Bottom Right - Fixed Position) */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                className="hidden sm:flex fixed bottom-4 right-4 text-xs text-muted bg-card/80 backdrop-blur-sm px-4 py-2.5 rounded-xl border border-border z-20"
              >
                <div className="flex items-center gap-2">
                  <MousePointer2 className="w-3.5 h-3.5" />
                  <span>Click anywhere to anchor camera rotation</span>
                </div>
              </motion.div>

              {/* Rotation Anchor Disclaimer - Mobile (Center) */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                className="sm:hidden absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-xs text-muted bg-card/90 backdrop-blur-sm px-4 py-3 rounded-xl border border-border z-20 text-center w-[90%] max-w-[280px]"
              >
                <div className="flex flex-col items-center gap-2">
                  <MousePointer2 className="w-4 h-4" />
                  <span className="break-words">Tap anywhere to anchor camera rotation</span>
                </div>
              </motion.div>
            </>
          )}
        </div>

        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && !isLoading && !viewerError && (
            <>
              {/* Mobile overlay */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowSettings(false)}
                className="fixed inset-0 bg-black/50 z-40 sm:hidden"
              />
              <motion.aside
                initial={{ x: "100%", opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: "100%", opacity: 0 }}
                transition={{ type: "spring", damping: 25, stiffness: 200 }}
                className="fixed right-0 top-0 bottom-0 w-full sm:w-80 max-w-sm bg-card border-l border-border flex flex-col overflow-hidden z-50"
              >
              <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-neutral-50 shrink-0">
                <h3 className="text-sm font-semibold text-foreground">Scene Settings</h3>
                <button onClick={handleResetSettings} className="text-xs text-muted hover:text-foreground transition-colors flex items-center gap-1.5">
                  <RotateCcw className="w-3 h-3" />
                  Reset All
                </button>
              </div>

              <div className="flex-1 overflow-y-auto min-h-0">
                <Section title="Splat Controls" defaultOpen={true}>
                  <div className="space-y-1">
                    <Slider label="Splat Scale" value={settings.splatScale} min={0.05} max={3} step={0.05} onChange={(v) => updateSetting("splatScale", v)} format={(v) => `${v.toFixed(2)}x`} />
                    <p className="text-[10px] text-muted -mt-1">Per-splat rendering size (shader uniform)</p>
                  </div>
                  <div className="space-y-1">
                    <Slider label="Focal Adjustment" value={settings.focalAdjustment} min={0.2} max={3} step={0.05} onChange={(v) => updateSetting("focalAdjustment", v)} format={(v) => `${v.toFixed(2)}x`} />
                    <p className="text-[10px] text-muted -mt-1">Focal length multiplier — larger = bigger splats</p>
                  </div>
                  <div className="space-y-1">
                    <Slider label="Max Screen Size" value={settings.maxSplatSize} min={64} max={4096} step={64} onChange={(v) => updateSetting("maxSplatSize", v)} />
                    <p className="text-[10px] text-muted -mt-1">Max screen-space splat size in pixels</p>
                  </div>
                  <div className="space-y-1">
                    <Slider label="Scene Opacity" value={settings.sceneOpacity} min={0} max={1} step={0.05} onChange={(v) => updateSetting("sceneOpacity", v)} format={(v) => `${Math.round(v * 100)}%`} />
                    <p className="text-[10px] text-muted -mt-1">Global scene opacity (GPU uniform)</p>
                  </div>
                  <Toggle label="Point Cloud Mode" checked={settings.pointCloudMode} onChange={(v) => updateSetting("pointCloudMode", v)} description="Render as small dots instead of splats" />
                </Section>

                <Section title="Camera" defaultOpen={false}>
                  <Slider label="Field of View" value={settings.fov} min={20} max={120} step={1} onChange={(v) => updateSetting("fov", v)} unit="°" />
                  <Slider label="Near Clip" value={settings.nearClip} min={0.01} max={10} step={0.01} onChange={(v) => updateSetting("nearClip", v)} />
                  <Slider label="Far Clip" value={settings.farClip} min={100} max={10000} step={100} onChange={(v) => updateSetting("farClip", v)} />
                  <Slider label="Exposure" value={settings.exposure} min={0.5} max={2} step={0.05} onChange={(v) => updateSetting("exposure", v)} format={(v) => `${Math.round(v * 100)}%`} />
                </Section>

                <Section title="Color Grading" defaultOpen={true}>
                  <Slider label="Brightness" value={settings.brightness} min={0.2} max={2} step={0.05} onChange={(v) => updateSetting("brightness", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Contrast" value={settings.contrast} min={0.5} max={2} step={0.05} onChange={(v) => updateSetting("contrast", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Saturation" value={settings.saturation} min={0} max={2} step={0.05} onChange={(v) => updateSetting("saturation", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Hue Shift" value={settings.hue} min={-180} max={180} step={1} onChange={(v) => updateSetting("hue", v)} unit="°" />
                  <Slider label="Temperature" value={settings.temperature} min={-1} max={1} step={0.05} onChange={(v) => updateSetting("temperature", v)} format={(v) => v > 0 ? "Warm" : v < 0 ? "Cool" : "Neutral"} />
                  <Slider label="Tint" value={settings.tint} min={-1} max={1} step={0.05} onChange={(v) => updateSetting("tint", v)} format={(v) => v > 0 ? "Magenta" : v < 0 ? "Green" : "Neutral"} />
                  <Slider label="Shadows" value={settings.shadows} min={-1} max={1} step={0.05} onChange={(v) => updateSetting("shadows", v)} format={(v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%`} />
                  <Slider label="Midtones" value={settings.midtones} min={-1} max={1} step={0.05} onChange={(v) => updateSetting("midtones", v)} format={(v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%`} />
                  <Slider label="Highlights" value={settings.highlights} min={-1} max={1} step={0.05} onChange={(v) => updateSetting("highlights", v)} format={(v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%`} />
                </Section>

                <Section title="Color Filters" defaultOpen={true}>
                  <div className="grid grid-cols-5 gap-2">
                    {COLOR_FILTERS.map(({ value, label, preview }) => (
                      <button
                        key={value}
                        onClick={() => updateSetting("colorFilter", value)}
                        className={`flex flex-col items-center gap-1 p-2 rounded-lg border-2 transition-all ${
                          settings.colorFilter === value 
                            ? "border-foreground bg-neutral-50" 
                            : "border-transparent hover:border-border"
                        }`}
                      >
                        <div 
                          className="w-8 h-8 rounded-md" 
                          style={{ background: preview }}
                        />
                        <span className="text-[10px] text-muted">{label}</span>
                      </button>
                    ))}
                  </div>
                </Section>

                <Section title="Blend Mode" defaultOpen={true}>
                  <Select label="Scene Blend" value={settings.blendMode} options={BLEND_MODES} onChange={(v) => updateSetting("blendMode", v as BlendMode)} />
                </Section>

                <Section title="Post Effects" defaultOpen={true}>
                  <Slider label="Sharpen" value={settings.sharpen} min={0} max={1} step={0.05} onChange={(v) => updateSetting("sharpen", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Blur" value={settings.blur} min={0} max={1} step={0.05} onChange={(v) => updateSetting("blur", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Film Grain" value={settings.grain} min={0} max={1} step={0.05} onChange={(v) => updateSetting("grain", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Noise" value={settings.noise} min={0} max={1} step={0.05} onChange={(v) => updateSetting("noise", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Bloom" value={settings.bloom} min={0} max={1} step={0.05} onChange={(v) => updateSetting("bloom", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Bloom Threshold" value={settings.bloomThreshold} min={0} max={1} step={0.05} onChange={(v) => updateSetting("bloomThreshold", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Vignette" value={settings.vignette} min={0} max={1} step={0.05} onChange={(v) => updateSetting("vignette", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Vignette Shape" value={settings.vignetteRoundness} min={0} max={1} step={0.05} onChange={(v) => updateSetting("vignetteRoundness", v)} format={(v) => v < 0.5 ? "Wide" : v > 0.5 ? "Tight" : "Normal"} />
                  <Slider label="Chromatic Aberration" value={settings.chromaticAberration} min={0} max={1} step={0.05} onChange={(v) => updateSetting("chromaticAberration", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Scanlines" value={settings.scanlines} min={0} max={1} step={0.05} onChange={(v) => updateSetting("scanlines", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Pixelate" value={settings.pixelate} min={0} max={1} step={0.05} onChange={(v) => updateSetting("pixelate", v)} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider label="Glitch" value={settings.glitch} min={0} max={1} step={0.05} onChange={(v) => updateSetting("glitch", v)} format={(v) => `${Math.round(v * 100)}%`} />
                </Section>

                <Section title="Atmosphere" defaultOpen={true}>
                  <Toggle label="Fog" checked={settings.fogEnabled} onChange={(v) => updateSetting("fogEnabled", v)} />
                  {settings.fogEnabled && (
                    <>
                      <Slider label="Fog Density" value={settings.fogDensity} min={0} max={1} step={0.05} onChange={(v) => updateSetting("fogDensity", v)} format={(v) => `${Math.round(v * 100)}%`} />
                      <div className="space-y-2">
                        <span className="text-sm text-muted">Fog Color</span>
                        <input type="color" value={settings.fogColor} onChange={(e) => updateSetting("fogColor", e.target.value)} className="w-full h-8 rounded-lg border border-border cursor-pointer" />
                      </div>
                    </>
                  )}
                </Section>

                <Section title="Background" defaultOpen={true}>
                  <div className="space-y-3">
                    <span className="text-sm text-muted">Preset Colors</span>
                    <div className="grid grid-cols-8 gap-2">
                      {BG_COLORS.map(({ color, label }) => (
                        <button
                          key={color}
                          onClick={() => updateSetting("bgColor", color)}
                          className={`w-7 h-7 rounded-lg border-2 transition-all ${settings.bgColor === color ? "border-foreground scale-110" : "border-transparent hover:border-border"}`}
                          style={{ backgroundColor: color }}
                          title={label}
                        />
                      ))}
                    </div>
                    <div className="space-y-2">
                      <span className="text-sm text-muted">Custom Color</span>
                      <input type="color" value={settings.bgColor} onChange={(e) => updateSetting("bgColor", e.target.value)} className="w-full h-8 rounded-lg border border-border cursor-pointer" />
                    </div>
                  </div>
                </Section>
              </div>

              <div className="shrink-0 border-t border-border bg-neutral-50">
                <div className="px-5 py-3 border-b border-border">
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted">Splats</span>
                      <span className="text-foreground">{result.plyBase64 ? `~${Math.round(atob(result.plyBase64).length / 250 / 1000)}K` : "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">Format</span>
                      <span className="text-foreground">3DGS</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">Renderer</span>
                      <span className="text-foreground">WebGL</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted">Enhanced</span>
                      <span className="text-foreground">{settings.autoEnhance ? "Yes" : "No"}</span>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  <button onClick={handleDownload} className="btn-secondary w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm">
                    <Download className="w-4 h-4" />
                    <span>Export Scene</span>
                  </button>
                </div>
              </div>
              </motion.aside>
            </>
          )}
        </AnimatePresence>

        {/* Follow Us CTA - Bottom Right */}
        {!isLoading && !viewerError && (
          <motion.a
            href="https://x.com/reveliumstudio"
            target="_blank"
            rel="noopener noreferrer"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 }}
            className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 flex items-center gap-2 text-sm text-foreground hover:text-foreground/80 transition-colors z-30 group"
          >
            <img
              src="https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/X.svg"
              alt="X Logo"
              className="w-4 h-4 opacity-80 group-hover:opacity-100 transition-opacity"
            />
            <span className="underline decoration-1 underline-offset-2">Follow us</span>
          </motion.a>
        )}
      </div>
    </div>
  );
}
