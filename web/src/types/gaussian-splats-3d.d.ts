declare module "@mkkellogg/gaussian-splats-3d" {
  export enum LogLevel {
    None = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Debug = 4,
  }

  export enum SceneFormat {
    Ply = 0,
    Splat = 1,
    Ksplat = 2,
  }

  export enum SplatRenderMode {
    ThreeD = 0,
    TwoD = 1,
  }

  export interface ViewerOptions {
    cameraUp?: [number, number, number];
    initialCameraPosition?: [number, number, number];
    initialCameraLookAt?: [number, number, number];
    rootElement?: HTMLElement;
    selfDrivenMode?: boolean;
    useBuiltInControls?: boolean;
    dynamicScene?: boolean;
    sharedMemoryForWorkers?: boolean;
    antialiased?: boolean;
    logLevel?: LogLevel;
    splatSortDistanceMapPrecision?: number;
    sphericalHarmonicsDegree?: number;
    integerBasedSort?: boolean;
    halfPrecisionCovariancesOnGPU?: boolean;
    enableOptionalEffects?: boolean;
    focalAdjustment?: number;
    kernel2DSize?: number;
    splatRenderMode?: SplatRenderMode;
    gpuAcceleratedSort?: boolean;
    enableSIMDInSort?: boolean;
    ignoreDevicePixelRatio?: boolean;
    [key: string]: unknown; // Allow additional properties
  }

  export interface SplatSceneOptions {
    splatAlphaRemovalThreshold?: number;
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    format?: SceneFormat;
    position?: [number, number, number];
    rotation?: [number, number, number, number];
    scale?: [number, number, number];
    onProgress?: (progress: number, message?: string) => void;
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(url: string, options?: SplatSceneOptions): Promise<void>;
    start(): void;
    stop(): void;
    dispose(): void;
    update(): void;
    render(): void;
    // Internal properties accessible at runtime
    camera?: unknown;
    perspectiveCamera?: unknown;
    renderer?: unknown;
    controls?: unknown;
    splatMesh?: unknown;
    [key: string]: unknown;
  }
}
