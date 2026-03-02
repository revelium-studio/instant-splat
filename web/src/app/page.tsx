"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import UploadStep from "@/components/UploadStep";
import ProcessingStep from "@/components/ProcessingStep";
import ViewerStep from "@/components/ViewerStep";

export type AppState = "upload" | "processing" | "viewer";

export interface ProcessedResult {
  originalImage: string;
  splatUrl?: string;
  plyBase64?: string;
  fileName: string;
}

// Get the API URL - bypass router for API calls to avoid edge timeout
function getApiUrl(path: string): string {
  if (typeof window === "undefined") return path;
  
  // If we're on the router domain (lab.revelium.studio), call the backend directly
  // This bypasses the edge middleware which has a 30-second timeout
  const hostname = window.location.hostname;
  if (hostname === "lab.revelium.studio") {
    return `https://web-beta-gilt-54.vercel.app${path}`;
  }
  
  // Otherwise use relative path (works on direct deployment)
  return path;
}

export default function Home() {
  const [state, setState] = useState<AppState>("upload");
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [result, setResult] = useState<ProcessedResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const progressRef = useRef<NodeJS.Timeout | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (progressRef.current) clearTimeout(progressRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const handleImageUpload = useCallback((file: File) => {
    setUploadedImage(file);
    setUploadedImages([file]);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target?.result as string;
      setImagePreview(url);
      setImagePreviews([url]);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleMultiImageUpload = useCallback((files: File[]) => {
    setUploadedImage(files[0]);
    setUploadedImages(files);
    setError(null);

    // Build previews for all files
    const previews: string[] = [];
    let loaded = 0;
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        previews.push(e.target?.result as string);
        loaded++;
        if (loaded === files.length) {
          setImagePreviews([...previews]);
          setImagePreview(previews[0]);
        }
      };
      reader.readAsDataURL(file);
    });
  }, []);

  // For local development - poll for job status
  const pollJobStatus = useCallback(
    async (jobId: string, imagePreviewUrl: string, fileName: string) => {
      const startTime = Date.now();
      const maxDuration = 300000; // 5 minutes max

      const poll = async () => {
        try {
          const elapsed = Date.now() - startTime;

          if (elapsed > maxDuration) {
            throw new Error("Processing timeout");
          }

          const response = await fetch(getApiUrl(`/api/process?jobId=${jobId}`));
          
          // Handle 504 Gateway Timeout specifically
          if (response.status === 504) {
            throw new Error("Request timed out. The server may be processing a large image. Please try again.");
          }

          // Check content-type before parsing
          const contentType = response.headers.get("content-type") || "";
          const isJson = contentType.includes("application/json");
          
          // Read response body once (can only be read once)
          const responseText = await response.text();
          
          if (!response.ok) {
            let errorMessage = `Processing failed (${response.status})`;
            
            // Check if response is HTML (common for error pages from router/gateway)
            if (responseText.trim().startsWith("<!DOCTYPE") || responseText.trim().startsWith("<html")) {
              if (response.status === 504) {
                errorMessage = "Request timed out. The server may be processing a large image. Please try again.";
              } else if (response.status === 404) {
                errorMessage = "API endpoint not found. Please check the deployment.";
              } else if (responseText.includes("Request Entity Too Large") || responseText.includes("413")) {
                errorMessage = "Image too large. Please use a smaller image.";
              } else {
                errorMessage = `Server error (${response.status}). Please try again.`;
              }
            } else if (isJson) {
              // Try to parse as JSON if content-type says it's JSON
              try {
                const errorData = JSON.parse(responseText);
                errorMessage = errorData.error || errorMessage;
              } catch (e) {
                // Content-type said JSON but parsing failed, use text
                if (responseText.length < 200) {
                  errorMessage = responseText.substring(0, 100);
                }
              }
            } else {
              // Not HTML and not JSON, use text if short
              if (responseText.length < 200) {
                errorMessage = responseText.substring(0, 100);
              }
            }
            throw new Error(errorMessage);
          }

          // Parse JSON only if content-type is JSON
          if (!isJson) {
            throw new Error(`Expected JSON but got ${contentType}. Response: ${responseText.substring(0, 200)}`);
          }

          // Safely parse JSON with error handling
          let data;
          try {
            data = JSON.parse(responseText);
          } catch (jsonError) {
            // If JSON parsing fails, show what we got
            throw new Error(`Failed to parse JSON response. Got: ${responseText.substring(0, 200)}`);
          }

          // Enhanced logging to debug status issues
          console.log(`📊 Polling response status:`, data.status);
          console.log(`📊 Polling response data:`, JSON.stringify(data, null, 2));
          console.log(`📊 Has plyBase64:`, !!data.plyBase64);

          if (data.status === "completed") {
            if (!data.plyBase64) {
              console.error("❌ Polling: Job marked as completed but no plyBase64 found!");
              console.error("Full response:", data);
              throw new Error("Processing completed but no output received. The job may have completed without generating a PLY file.");
            }
            
            console.log("✅ Polling: Job completed, animating to 100%");
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
            if (progressRef.current) {
              clearTimeout(progressRef.current);
              progressRef.current = null;
            }

            // Job completed - animate to 100%
            const animateTo100 = () => {
              setProgress((prev) => {
                if (prev >= 100) {
                  return 100;
                }
                const increment = Math.min(3, 100 - prev);
                const newProgress = prev + increment;
                if (newProgress < 100) {
                  progressRef.current = setTimeout(animateTo100, 50) as unknown as NodeJS.Timeout;
                }
                return newProgress;
              });
            };
            animateTo100();

            setTimeout(() => {
              console.log("✅ Polling: Transitioning to viewer");
              setResult({
                originalImage: imagePreviewUrl,
                splatUrl: data.splatUrl,
                plyBase64: data.plyBase64,
                fileName: fileName.replace(/\.[^/.]+$/, ""),
              });
              setState("viewer");
            }, 400);
          } else if (data.status === "failed") {
            console.error("❌ Polling: Job failed:", data.error);
            throw new Error(data.error || "Processing failed");
          } else {
            // Job still processing - update progress based on RunPod status
            const runPodStatus = data.runPodStatus || data.status || "processing";
            const elapsed = Date.now() - startTime;
            
            // Calculate progress based on RunPod status and elapsed time
            // Calculate progress based on status and elapsed time
            let progress = 0;
            const statusUpper = String(runPodStatus).toUpperCase();
            
            if (statusUpper === "IN_QUEUE" || statusUpper === "QUEUED") {
              // In queue: 5% to 15% over first 10 seconds
              const queueTime = Math.min(elapsed, 10000);
              progress = 5 + (queueTime / 10000) * 10;
            } else if (statusUpper === "IN_PROGRESS" || statusUpper === "IN_PROCESS" || statusUpper === "PROCESSING") {
              // In progress: 15% to 85% over estimated duration
              const processingTime = Math.max(0, elapsed - 10000);
              const estimatedDuration = 90000; // 90 seconds
              const ratio = Math.min(processingTime / estimatedDuration, 1);
              const eased = 1 - Math.pow(1 - ratio, 2); // Ease-out curve
              progress = 15 + (eased * 70);
            } else {
              // Unknown status or "processing": gradual time-based increase
              progress = Math.min((elapsed / 120000) * 80, 80);
            }
            
            setProgress((prev) => {
              const target = Math.min(85, progress);
              const newProgress = prev + (target - prev) * 0.2;
              return Math.max(prev, newProgress);
            });
            
            console.log(`⏳ Polling: Job still processing, RunPod status: "${runPodStatus}", progress: ${Math.round(progress)}%`);
            console.log(`⏳ Elapsed time: ${Math.round(elapsed / 1000)}s`);
          }
        } catch (err) {
          console.error("❌ Polling error:", err);
          if (pollingRef.current) clearInterval(pollingRef.current);
          if (progressRef.current) clearTimeout(progressRef.current);
          setError(err instanceof Error ? err.message : "Processing failed");
          setState("upload");
          setProgress(0);
        }
      };

      pollingRef.current = setInterval(poll, 3000);
      poll();
    },
    []
  );

  const handleProcess = useCallback(async () => {
    if (!uploadedImage || !imagePreview) return;

    setState("processing");
    setProgress(0);
    setError(null);

    const formData = new FormData();
    // Send all images if multi-image, otherwise single
    if (uploadedImages.length > 1) {
      uploadedImages.forEach((file) => formData.append("images", file));
    } else {
      formData.append("image", uploadedImage);
    }

    console.log(`🚀 Starting InstantSplat++ processing with ${uploadedImages.length} image(s)...`);

    try {
      console.log("Submitting image for processing...");

      abortRef.current = new AbortController();

      const response = await fetch(getApiUrl("/api/process"), {
        method: "POST",
        body: formData,
        signal: abortRef.current.signal,
      });

      // Read response body first (can only be read once)
      const responseText = await response.text();
      
      // Check content-type before parsing
      const contentType = response.headers.get("content-type") || "";
      const isJson = contentType.includes("application/json");

      // Handle specific status codes
      if (response.status === 413) {
        throw new Error("Image file is too large. Maximum size is 4MB. Please compress or resize your image and try again.");
      }
      
      if (response.status === 504) {
        throw new Error(
          "Processing timeout: The image processing is taking longer than expected. " +
          "This may be due to high load on the processing service. Please wait a moment and try again with a smaller image, or check back in a few minutes."
        );
      }

      if (response.status === 503) {
        // Service Unavailable - RunPod is having issues
        const errorData = isJson ? JSON.parse(responseText) : null;
        throw new Error(errorData?.error || "The processing service is currently experiencing issues. All requests are pending. Please try again in a few minutes.");
      }

      // Check if response is ok before parsing JSON
      if (!response.ok) {
        let errorMessage = `Processing failed (${response.status})`;
        
        // Handle 429 (Rate Limit / Billing Limit) specifically
        if (response.status === 429) {
          if (isJson) {
            try {
              const errorData = JSON.parse(responseText);
              errorMessage = errorData.error || "Service temporarily unavailable. Please try again later.";
            } catch (e) {
              errorMessage = "Service temporarily unavailable: The processing service has reached its limit. Please try again later.";
            }
          } else {
            errorMessage = "Service temporarily unavailable: The processing service has reached its limit. Please try again later.";
          }
          throw new Error(errorMessage);
        }
        
        // Check if response is HTML (common for error pages from router/gateway)
        if (responseText.trim().startsWith("<!DOCTYPE") || responseText.trim().startsWith("<html")) {
          if (response.status === 504) {
            errorMessage = "Request timed out. The server may be processing a large image. Please try again.";
          } else if (response.status === 404) {
            errorMessage = "API endpoint not found. Please check the deployment.";
          } else if (responseText.includes("Request Entity Too Large") || responseText.includes("413")) {
            errorMessage = "Image too large. Please use a smaller image.";
          } else {
            errorMessage = `Server error (${response.status}). Please try again.`;
          }
        } else if (isJson) {
          // Try to parse as JSON if content-type says it's JSON
          try {
            const errorData = JSON.parse(responseText);
            errorMessage = errorData.error || errorMessage;
          } catch (e) {
            // Content-type said JSON but parsing failed, use text
            if (responseText.length < 200) {
              errorMessage = responseText.substring(0, 100);
            }
          }
        } else {
          // Not HTML and not JSON, use text if short
          if (responseText.length < 200) {
            errorMessage = responseText.substring(0, 100);
          }
        }
        throw new Error(errorMessage);
      }

      // Parse JSON only if content-type is JSON
      if (!isJson) {
        throw new Error(`Expected JSON but got ${contentType}. Response: ${responseText.substring(0, 200)}`);
      }

      // Safely parse JSON with error handling
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (jsonError) {
        // If JSON parsing fails, show what we got
        throw new Error(`Failed to parse JSON response. Got: ${responseText.substring(0, 200)}`);
      }
      console.log("✅ Response received:", data);
      console.log("✅ Response status:", response.status);
      console.log("✅ Response OK:", response.ok);

      if (!response.ok) {
        // Only clear progress on error
        if (progressRef.current) {
          clearTimeout(progressRef.current);
          progressRef.current = null;
        }
        throw new Error(data.error || "Processing failed");
      }

      // Check if this is a synchronous result (Vercel/Modal) or async (local)
      if (data.status === "completed" && data.plyBase64) {
        // Synchronous result - smoothly animate to 100%
        console.log("✅ Synchronous completion: Animating to 100%");
        if (progressRef.current) {
          clearTimeout(progressRef.current);
          progressRef.current = null;
        }
        
        const animateTo100 = () => {
          setProgress((prev) => {
            if (prev >= 100) {
              return 100;
            }
            const increment = Math.min(3, 100 - prev);
            const newProgress = prev + increment;
            if (newProgress < 100) {
              progressRef.current = setTimeout(animateTo100, 50) as unknown as NodeJS.Timeout;
            }
            return newProgress;
          });
        };
        animateTo100();

        setTimeout(() => {
          console.log("✅ Transitioning to viewer with completed result");
          setResult({
            originalImage: imagePreview,
            plyBase64: data.plyBase64,
            fileName: uploadedImage.name.replace(/\.[^/.]+$/, ""),
          });
          setState("viewer");
        }, 500);
      } else if (data.jobId) {
        // Async result (local) - start polling
        // Keep progress animation running - don't clear it!
        console.log("🔄 Async job started, polling for status:", data.jobId);
        pollJobStatus(data.jobId, imagePreview, uploadedImage.name);
      } else {
        console.error("❌ Unexpected response format:", data);
        throw new Error("Unexpected response format");
      }
    } catch (err) {
      console.error("❌ Processing error:", err);
      if (progressRef.current) {
        console.log("🛑 Clearing progress timeout due to error...");
        clearTimeout(progressRef.current);
        progressRef.current = null;
      }

      if (err instanceof Error && err.name === "AbortError") {
        console.log("⚠️ Request aborted by user");
        return; // User cancelled
      }

      setError(err instanceof Error ? err.message : "Processing failed");
      setState("upload");
      setProgress(0);
    }
  }, [uploadedImage, uploadedImages, imagePreview, pollJobStatus]);

  const handleReset = useCallback(() => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    if (progressRef.current) clearTimeout(progressRef.current);
    if (abortRef.current) abortRef.current.abort();

    setState("upload");
    setUploadedImage(null);
    setUploadedImages([]);
    setImagePreview(null);
    setImagePreviews([]);
    setResult(null);
    setProgress(0);
    setError(null);
  }, []);

  return (
    <main className="min-h-screen bg-background flex flex-col">
      <AnimatePresence mode="wait">
        {state === "upload" && (
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="flex-1 flex flex-col"
          >
            <UploadStep
              onImageUpload={handleImageUpload}
              onMultiImageUpload={handleMultiImageUpload}
              uploadedImage={uploadedImage}
              uploadedImages={uploadedImages}
              imagePreview={imagePreview}
              imagePreviews={imagePreviews}
              onProcess={handleProcess}
              onClear={() => {
                setUploadedImage(null);
                setUploadedImages([]);
                setImagePreview(null);
                setImagePreviews([]);
                setError(null);
              }}
              error={error}
            />
          </motion.div>
        )}

        {state === "processing" && (
          <motion.div
            key="processing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="flex-1 flex flex-col"
          >
            <ProcessingStep
              imagePreview={imagePreview!}
              progress={progress}
              fileName={uploadedImage?.name || ""}
            />
          </motion.div>
        )}

        {state === "viewer" && result && (
          <motion.div
            key="viewer"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="flex-1 flex flex-col"
          >
            <ViewerStep result={result} onBack={handleReset} />
          </motion.div>
        )}
      </AnimatePresence>
      </main>
  );
}
