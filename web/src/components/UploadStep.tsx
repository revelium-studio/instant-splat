"use client";

import { useCallback, useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, Image as ImageIcon, X, ArrowRight, AlertCircle } from "lucide-react";

interface UploadStepProps {
  onImageUpload: (file: File) => void;
  onMultiImageUpload?: (files: File[]) => void;
  uploadedImage: File | null;
  uploadedImages?: File[];
  imagePreview: string | null;
  imagePreviews?: string[];
  onProcess: () => void;
  onClear: () => void;
  error?: string | null;
}

export default function UploadStep({
  onImageUpload,
  onMultiImageUpload,
  uploadedImage,
  uploadedImages = [],
  imagePreview,
  imagePreviews = [],
  onProcess,
  onClear,
  error,
}: UploadStepProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const maxSize = 4 * 1024 * 1024; // 4MB per file
      const imageFiles: File[] = [];
      
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        const file = e.dataTransfer.files[i];
        if (file.type.startsWith("image/")) {
          if (file.size > maxSize) {
            alert(`"${file.name}" is too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum 4MB per image.`);
            continue;
          }
          imageFiles.push(file);
        }
      }

      if (imageFiles.length > 1 && onMultiImageUpload) {
        onMultiImageUpload(imageFiles.slice(0, 10)); // Max 10 images
      } else if (imageFiles.length === 1) {
        onImageUpload(imageFiles[0]);
      }
    },
    [onImageUpload, onMultiImageUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;

      const maxSize = 4 * 1024 * 1024;
      const validFiles: File[] = [];

      for (let i = 0; i < files.length; i++) {
        if (files[i].size > maxSize) {
          alert(`"${files[i].name}" is too large (${(files[i].size / 1024 / 1024).toFixed(1)}MB). Maximum 4MB per image.`);
          continue;
        }
        validFiles.push(files[i]);
      }

      if (validFiles.length > 1 && onMultiImageUpload) {
        onMultiImageUpload(validFiles.slice(0, 10));
      } else if (validFiles.length === 1) {
        onImageUpload(validFiles[0]);
      }
    },
    [onImageUpload, onMultiImageUpload]
  );

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header with Logo */}
      <header className="py-6 sm:py-10 px-4 sm:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Logo */}
          <img
            src="https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/wordmark.svg"
            alt="Revelium Studio"
            className="h-4 sm:h-10 mx-auto mb-3 sm:mb-4 w-auto"
          />
          {/* Subtitle */}
          <p className="text-muted text-base sm:text-lg font-medium">InstantSplat++</p>
        </motion.div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 sm:px-6 pb-20">
        {/* Upload Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full max-w-2xl mx-auto"
        >
          {!uploadedImage ? (
            /* Drop Zone */
            <div
              className={`relative border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer ${
                isDragging
                  ? "border-primary bg-primary/5 scale-[1.02]"
                  : "border-border hover:border-primary/50 hover:bg-card/50"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                multiple
                onChange={handleFileSelect}
              />
              
              <div className="flex flex-col items-center justify-center py-16 sm:py-24 px-6">
                <motion.div
                  initial={{ scale: 0.9 }}
                  animate={{ scale: isDragging ? 1.1 : 1 }}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  className={`w-16 h-16 sm:w-20 sm:h-20 rounded-full flex items-center justify-center mb-6 ${
                    isDragging ? "bg-primary/20" : "bg-card"
                  }`}
                >
                  <Upload className={`w-8 h-8 sm:w-10 sm:h-10 ${isDragging ? "text-primary" : "text-muted"}`} />
                </motion.div>
                
                <h3 className="text-lg sm:text-xl font-semibold text-foreground mb-2">
                  Upload images
                </h3>
                <p className="text-muted text-sm sm:text-base text-center max-w-md">
                  Drag and drop or click to select. Supports PNG, JPG, WebP, and HEIC.
                </p>
                <p className="text-primary/80 text-xs mt-2 font-medium">
                  💡 Upload multiple images of the same scene for much better 3D quality
                </p>
                <p className="text-muted/60 text-xs mt-1">
                  Max 4 MB per image · up to 10 images
                </p>
              </div>
            </div>
          ) : (
            /* Image Preview — single or multi */
            <div className="bg-card border border-border rounded-2xl p-6 sm:p-8">
              <div className="flex flex-col sm:flex-row items-center gap-6">
                {/* Preview thumbnails */}
                <div className="flex flex-wrap gap-2 flex-shrink-0">
                  {(imagePreviews.length > 0 ? imagePreviews : (imagePreview ? [imagePreview] : [])).map((src, i) => (
                    <div
                      key={i}
                      className="relative w-24 h-24 sm:w-32 sm:h-32 rounded-xl overflow-hidden bg-background"
                    >
                      <img src={src} alt={`Preview ${i + 1}`} className="w-full h-full object-cover" />
                    </div>
                  ))}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onClear();
                    }}
                    className="absolute top-2 right-2 w-7 h-7 rounded-full bg-background/80 backdrop-blur-sm flex items-center justify-center hover:bg-background transition-colors z-10"
                    style={{ position: "relative", top: 0, right: 0 }}
                  >
                    <X className="w-4 h-4 text-foreground" />
                  </button>
                </div>

                {/* File Info & Actions */}
                <div className="flex-1 text-center sm:text-left">
                  <div className="flex items-center justify-center sm:justify-start gap-2 mb-2">
                    <ImageIcon className="w-5 h-5 text-primary" />
                    <span className="text-foreground font-medium truncate max-w-[200px]">
                      {uploadedImages.length > 1
                        ? `${uploadedImages.length} images selected`
                        : uploadedImage?.name || "image"}
                    </span>
                  </div>
                  <p className="text-muted text-sm mb-1">
                    {uploadedImages.length > 1
                      ? `${(uploadedImages.reduce((s, f) => s + f.size, 0) / 1024).toFixed(0)} KB total`
                      : uploadedImage ? `${(uploadedImage.size / 1024).toFixed(1)} KB` : ""}
                  </p>
                  {uploadedImages.length > 1 && (
                    <p className="text-primary/70 text-xs mb-3 font-medium">
                      ✨ Multiple views → higher quality 3D
                    </p>
                  )}
                  {(uploadedImages.length <= 1 && uploadedImage) && (
                    <p className="text-muted/60 text-xs mb-3">
                      Single image mode — 3 synthetic views will be generated
                    </p>
                  )}
                  
                  <button
                    onClick={onProcess}
                    className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:bg-primary/90 transition-colors"
                  >
                    Generate 3D Splat
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </motion.div>

        {/* Error message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 w-full max-w-2xl"
          >
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-50 border border-red-100 text-red-600">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="py-4 sm:py-6 text-center px-4 sm:px-0">
        <div className="w-[90%] sm:w-1/2 mx-auto">
          <p className="text-[10px] font-medium opacity-50 leading-tight sm:leading-relaxed">
            Disclaimer. This demo uses the InstantSplat++ model ([phai-lab/InstantSplatPP](https://github.com/phai-lab/InstantSplatPP)) for 3D Gaussian Splat generation. All outputs are experimental and provided "as is." You are solely responsible for any use of the results. By uploading images, you confirm you have the necessary rights and that your content does not violate any laws or third-party rights.
          </p>
        </div>
      </footer>
    </div>
  );
}
