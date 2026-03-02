"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useRef } from "react";
import { Cpu, Layers, Sparkles } from "lucide-react";

interface ProcessingStepProps {
  imagePreview: string;
  progress: number;
  fileName: string;
}

// Messages that match the 3 step phases
const PHASE_MESSAGES = {
  depth: [
    "Analyzing depth layers",
    "Extracting spatial data",
    "Mapping 3D structure",
  ],
  gaussian: [
    "Generating Gaussians",
    "Computing point cloud",
    "Building splat data",
  ],
  optimization: [
    "Optimizing scene",
    "Refining positions",
    "Finalizing output",
  ],
};

const steps = [
  { icon: Cpu, label: "Depth Analysis", threshold: 0, phase: "depth" as const },
  { icon: Layers, label: "Gaussian Generation", threshold: 33, phase: "gaussian" as const },
  { icon: Sparkles, label: "Scene Optimization", threshold: 66, phase: "optimization" as const },
];

export default function ProcessingStep({
  imagePreview,
  progress,
  fileName,
}: ProcessingStepProps) {
  const [displayedProgress, setDisplayedProgress] = useState(0);
  const [currentMessage, setCurrentMessage] = useState(PHASE_MESSAGES.depth[0]);
  const lastProgressRef = useRef(0);
  const messageIndexRef = useRef(0);

  // Never go backwards - only increase
  useEffect(() => {
    if (progress > lastProgressRef.current) {
      setDisplayedProgress(progress);
      lastProgressRef.current = progress;
    }
  }, [progress]);

  // Update message based on current phase
  useEffect(() => {
    let phase: "depth" | "gaussian" | "optimization" = "depth";
    
    if (displayedProgress >= 66) {
      phase = "optimization";
    } else if (displayedProgress >= 33) {
      phase = "gaussian";
    }

    const messages = PHASE_MESSAGES[phase];
    const subProgress = phase === "depth" 
      ? displayedProgress / 33 
      : phase === "gaussian" 
        ? (displayedProgress - 33) / 33 
        : (displayedProgress - 66) / 34;
    
    const messageIndex = Math.min(
      Math.floor(subProgress * messages.length),
      messages.length - 1
    );

    if (messageIndex !== messageIndexRef.current || PHASE_MESSAGES[phase][messageIndex] !== currentMessage) {
      messageIndexRef.current = messageIndex;
      setCurrentMessage(messages[messageIndex]);
    }
  }, [displayedProgress, currentMessage]);

  const currentStepIndex = steps.findIndex(
    (step, index) =>
      displayedProgress >= step.threshold &&
      (index === steps.length - 1 || displayedProgress < steps[index + 1].threshold)
  );

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="py-6 sm:py-10 px-4 sm:px-8 text-center">
        <img
          src="https://pub-31178c53271846bd9cb48918a4fdd72e.r2.dev/wordmark.svg"
          alt="Revelium Studio"
          className="h-4 sm:h-10 mx-auto mb-3 sm:mb-4 w-auto"
        />
        <p className="text-muted text-base sm:text-lg font-medium">InstantSplat++</p>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-xl"
        >
          {/* Image preview with Apple Intelligence style BORDER glow */}
          <div className="relative mb-8">
            {/* Outer blurred glow - the "aura" effect */}
            <div 
              className="absolute -inset-6 rounded-3xl"
              style={{
                background: `conic-gradient(
                  from var(--glow-angle, 0deg),
                  #ff6b6b,
                  #feca57,
                  #48dbfb,
                  #ff9ff3,
                  #54a0ff,
                  #5f27cd,
                  #ff6b6b
                )`,
                filter: "blur(40px)",
                opacity: 0.5,
                animation: "rotateGlow 4s linear infinite",
              }}
            />
            
            {/* Inner border glow - tighter, more defined */}
            <div 
              className="absolute -inset-1 rounded-2xl"
              style={{
                background: `conic-gradient(
                  from var(--glow-angle, 0deg),
                  #ff6b6b,
                  #feca57,
                  #48dbfb,
                  #ff9ff3,
                  #54a0ff,
                  #5f27cd,
                  #ff6b6b
                )`,
                filter: "blur(8px)",
                opacity: 0.8,
                animation: "rotateGlow 3s linear infinite",
              }}
            />

            {/* Image container */}
            <div className="relative overflow-hidden bg-card rounded-2xl border border-white/10">
              <div className="relative aspect-[16/10]">
                <img
                  src={imagePreview}
                  alt="Processing"
                  className="w-full h-full object-cover"
                />

                {/* Corner markers */}
                <div className="absolute top-4 left-4 w-5 h-5 border-l-2 border-t-2 border-white/30" />
                <div className="absolute top-4 right-4 w-5 h-5 border-r-2 border-t-2 border-white/30" />
                <div className="absolute bottom-4 left-4 w-5 h-5 border-l-2 border-b-2 border-white/30" />
                <div className="absolute bottom-4 right-4 w-5 h-5 border-r-2 border-b-2 border-white/30" />
              </div>
            </div>
          </div>

          {/* Progress section */}
          <div className="text-center mb-10">
            {/* Message that matches current phase */}
            <div className="flex items-center justify-center gap-2 mb-4 h-6">
              <div className="relative w-2 h-2">
                <div className="absolute inset-0 rounded-full bg-foreground" />
                <div className="absolute inset-0 rounded-full bg-foreground animate-ping opacity-75" />
              </div>
              <AnimatePresence mode="wait">
                <motion.span
                  key={currentMessage}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  transition={{ duration: 0.2 }}
                  className="text-foreground text-sm font-medium"
                >
                  {currentMessage}
                </motion.span>
              </AnimatePresence>
            </div>

            {/* Progress bar */}
            <div className="relative h-1.5 bg-border rounded-full overflow-hidden mb-3">
              <motion.div
                className="absolute inset-y-0 left-0 bg-foreground rounded-full"
                style={{ width: `${displayedProgress}%` }}
              />
            </div>

            {/* Percentage and filename */}
            <div className="flex items-center justify-between text-sm">
              <p className="text-muted">
                Processing{" "}
                <span className="text-foreground font-medium">{fileName}</span>
              </p>
              <span className="text-foreground font-medium tabular-nums">
                {Math.round(displayedProgress)}%
              </span>
            </div>
          </div>

          {/* Step indicators */}
          <div className="flex items-center justify-between px-8">
            {steps.map((step, index) => {
              const isActive = displayedProgress >= step.threshold;
              const isCurrent = index === currentStepIndex;
              const Icon = step.icon;

              return (
                <motion.div
                  key={step.label}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 ${
                      isCurrent
                        ? "bg-foreground text-white"
                        : isActive
                        ? "bg-border text-foreground"
                        : "bg-background text-muted"
                    }`}
                  >
                    <Icon className="w-5 h-5" strokeWidth={1.5} />
                  </div>
                  <span
                    className={`text-xs transition-colors duration-300 ${
                      isActive ? "text-foreground" : "text-muted"
                    }`}
                  >
                    {step.label}
                  </span>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      </main>

      {/* CSS for rotating glow animation */}
      <style jsx>{`
        @property --glow-angle {
          syntax: '<angle>';
          initial-value: 0deg;
          inherits: false;
        }
        
        @keyframes rotateGlow {
          from {
            --glow-angle: 0deg;
          }
          to {
            --glow-angle: 360deg;
          }
        }
      `}</style>
    </div>
  );
}
