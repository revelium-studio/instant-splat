import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir, readdir, readFile } from "fs/promises";
import { spawn } from "child_process";
import path from "path";
import { existsSync } from "fs";

// CORS headers to allow cross-origin requests from router domain
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "https://lab.revelium.studio",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, X-Requested-With",
  "Access-Control-Allow-Credentials": "true",
  "Access-Control-Max-Age": "86400",
};

// Helper to add CORS headers to response
function corsResponse(response: NextResponse): NextResponse {
  Object.entries(CORS_HEADERS).forEach(([key, value]) => {
    response.headers.set(key, value);
  });
  return response;
}

// Check if running on Vercel
const IS_VERCEL = process.env.VERCEL === "1";

// Modal configuration for GPU processing
// Single InstantSplat++ router endpoint (handles process + status)
const MODAL_ENDPOINT =
  process.env.MODAL_ENDPOINT ||
  "https://revelium-studio--instantplus-instantplus-router.modal.run";

// Paths for local development
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const UPLOADS_DIR = path.join(process.cwd(), "public", "uploads");
const OUTPUTS_DIR = path.join(process.cwd(), "public", "outputs");
const JOBS_DIR = path.join(process.cwd(), ".jobs");

// Allow long-running requests (5 minutes for Enterprise, but check Vercel plan limits)
export const maxDuration = 300;

interface JobStatus {
  status: "processing" | "completed" | "failed";
  splatUrl?: string;
  plyBase64?: string;
  error?: string;
  fileName: string;
  startTime: number;
  callId?: string; // Modal call_id for async jobs
}

async function saveJobStatus(jobId: string, status: JobStatus) {
  await mkdir(JOBS_DIR, { recursive: true });
  await writeFile(path.join(JOBS_DIR, `${jobId}.json`), JSON.stringify(status));
}

async function getJobStatus(jobId: string): Promise<JobStatus | null> {
  const filePath = path.join(JOBS_DIR, `${jobId}.json`);
  if (!existsSync(filePath)) return null;
  const data = await readFile(filePath, "utf-8");
  return JSON.parse(data);
}

// Process with Modal (GPU cloud) — supports single or multiple images
async function processWithModal(
  images: Array<{ base64: string; filename: string }>,
  prompt: string = "",
  elevation: number = 20
): Promise<{ callId: string } | { plyBase64: string }> {
  console.log(`🚀 Sending ${images.length} image(s) to Modal InstantSplat++ router endpoint...`);
  console.log(`📍 Modal endpoint: ${MODAL_ENDPOINT}`);

  // Build the request body.  If only one image, use backward-compatible
  // single-image fields.  If multiple, use the "images" array.
  let body: Record<string, unknown>;
  if (images.length === 1) {
    body = {
      op: "process",
      image: images[0].base64,
      filename: images[0].filename,
      prompt,
      elevation,
      async: true,
    };
  } else {
    body = {
      op: "process",
      images: images.map((i) => ({ image: i.base64, filename: i.filename })),
      prompt,
      elevation,
      async: true,
    };
  }

  const response = await fetch(MODAL_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60000),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`❌ Modal request failed: ${response.status} - ${errorText}`);
    throw new Error(`Failed to process with Modal: ${response.status} - ${errorText}`);
  }

  const result = await response.json();
  
  if (result.error) {
    throw new Error(result.error);
  }
  
  if (result.call_id) {
    console.log(`✅ Modal async job created with call_id: ${result.call_id}`);
    return { callId: result.call_id };
  }
  
  if (result.ply) {
    console.log(`✅ Modal returned synchronous result`);
    return { plyBase64: result.ply };
  }
  
  throw new Error("Unexpected response from Modal");
}

// Process locally with a 3D pipeline - for local dev (if set up)
function processLocally(
  inputPath: string,
  outputDir: string,
  jobId: string,
  fileName: string
) {
  const venvActivate = path.join(PROJECT_ROOT, ".venv", "bin", "activate");

  // Note: Local 3D processing requires the full setup
  // This is a placeholder - for local dev, you might want to use Modal InstantSplat++ directly
  const child = spawn(
    "bash",
    [
      "-c",
      `source "${venvActivate}" && python -c "print('Local DiffSplat not configured. Use Modal deployment.')"`,
    ],
    {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PATH: `${PROJECT_ROOT}/.venv/bin:${process.env.PATH}`,
      },
    }
  );

  child.stdout.on("data", (data) => {
    console.log("DiffSplat:", data.toString());
  });

  child.stderr.on("data", (data) => {
    console.log("DiffSplat stderr:", data.toString());
  });

  child.on("close", async () => {
    try {
      const outputFiles = await readdir(outputDir);
      const plyFile = outputFiles.find((f) => f.endsWith(".ply"));

      if (plyFile) {
        const splatUrl = `/outputs/${jobId}/${plyFile}`;
        console.log("Job completed successfully:", jobId, splatUrl);
        await saveJobStatus(jobId, {
          status: "completed",
          splatUrl,
          fileName,
          startTime: Date.now(),
        });
      } else {
        await saveJobStatus(jobId, {
          status: "failed",
          error: "No output file generated. Local DiffSplat processing not configured.",
          fileName,
          startTime: Date.now(),
        });
      }
    } catch (error) {
      await saveJobStatus(jobId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Unknown error",
        fileName,
        startTime: Date.now(),
      });
    }
  });

  child.on("error", async (error) => {
    await saveJobStatus(jobId, {
      status: "failed",
      error: error.message,
      fileName,
      startTime: Date.now(),
    });
  });
}

// Handle CORS preflight requests
export async function OPTIONS(request: NextRequest) {
  const response = new NextResponse(null, { status: 204 });
  return corsResponse(response);
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const prompt = (formData.get("prompt") as string) || "";
    const elevation = parseInt((formData.get("elevation") as string) || "20", 10);

    // Support both single file ("image") and multiple files ("images")
    const singleFile = formData.get("image") as File | null;
    const multiFiles = formData.getAll("images") as File[];

    const files: File[] = [];
    if (multiFiles.length > 0) {
      files.push(...multiFiles);
    } else if (singleFile) {
      files.push(singleFile);
    }

    if (files.length === 0) {
      return corsResponse(NextResponse.json({ error: "No image provided" }, { status: 400 }));
    }

    console.log(`🚀 Starting Modal InstantSplat++ processing with ${files.length} image(s)...`);

    // Convert all files to base64
    const images: Array<{ base64: string; filename: string }> = [];
    for (const file of files) {
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);
      images.push({ base64: buffer.toString("base64"), filename: file.name });
    }

    try {
      const result = await processWithModal(images, prompt, elevation);

      if ("callId" in result) {
        console.log(`✅ Modal async job created with call_id: ${result.callId}`);
        return corsResponse(NextResponse.json({
          success: true,
          jobId: result.callId,
          message: `Processing ${files.length} image(s) on Modal`,
        }));
      } else {
        console.log(`✅ Modal returned synchronous result`);
        return corsResponse(NextResponse.json({
          success: true,
          status: "completed",
          plyBase64: result.plyBase64,
        }));
      }
    } catch (modalError) {
      const errorMessage = modalError instanceof Error ? modalError.message : "Failed to process with Modal";
      console.error("❌ Failed to process with Modal:", errorMessage);
      return corsResponse(NextResponse.json(
        { error: "Failed to start processing job. Please try again in a few minutes." },
        { status: 503 }
      ));
    }
  } catch (error) {
    console.error("Error processing:", error);
    return corsResponse(NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to process image" },
      { status: 500 }
    ));
  }
}

// GET for polling job status (Modal async jobs)
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get("jobId");

  if (!jobId) {
    return corsResponse(NextResponse.json({ error: "No jobId provided" }, { status: 400 }));
  }

  // For Modal InstantSplat++ router, we always POST to the same endpoint for status
  if (IS_VERCEL || jobId.startsWith("fc-")) {
    try {
      console.log(`🔍 Polling Modal InstantSplat++ router status for call_id: ${jobId}`);

      const response = await fetch(MODAL_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          op: "status",
          call_id: jobId,
        }),
        signal: AbortSignal.timeout(30000), // 30 second timeout
      });

      console.log(`📍 Modal InstantSplat++ router status response: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(
          `❌ Modal InstantSplat++ status check failed: ${response.status} - ${errorText}`
        );

        // Return processing status as fallback
        return corsResponse(
          NextResponse.json({
            status: "processing",
            fileName: "",
            startTime: Date.now(),
          })
        );
      }

      const modalStatus = await response.json();
      console.log(`📊 Modal InstantSplat++ status response:`, JSON.stringify(modalStatus, null, 2));

      if (modalStatus.status === "completed") {
        console.log(`✅ Modal InstantSplat++ job completed`);
        return corsResponse(
          NextResponse.json({
            status: "completed",
            plyBase64: modalStatus.ply,
            fileName: "",
            startTime: Date.now(),
          })
        );
      } else if (modalStatus.status === "failed") {
        console.error(`❌ Modal InstantSplat++ job failed:`, modalStatus.error);
        return corsResponse(
          NextResponse.json({
            status: "failed",
            error: modalStatus.error || "Processing failed",
            fileName: "",
            startTime: Date.now(),
          })
        );
      } else {
        // Still processing
        console.log(`⏳ Modal InstantSplat++ job still processing`);
        return corsResponse(
          NextResponse.json({
            status: "processing",
            modalStatus: modalStatus.status,
            fileName: "",
            startTime: Date.now(),
          })
        );
      }
    } catch (error) {
      console.error("❌ Error polling Modal InstantSplat++ router status:", error);
      // If Modal API fails, assume still processing
      return corsResponse(
        NextResponse.json({
          status: "processing",
          fileName: "",
          startTime: Date.now(),
        })
      );
    }
  } else {
    // Local processing - use file-based status
    const status = await getJobStatus(jobId);

    if (!status) {
      return corsResponse(NextResponse.json({ 
        status: "processing", 
        fileName: "",
        startTime: Date.now(),
      }));
    }

    return corsResponse(NextResponse.json(status));
  }
}
