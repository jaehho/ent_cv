import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { readdirSync, statSync, createReadStream, readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PREDICTIONS_DIR = "/mnt/data/ent_cv/predictions";
const RAW_DIR         = "/mnt/data/ent_cv/raw";
const PROJECT_ROOT    = join(__dirname, "..");
const VENV_PYTHON     = join(PROJECT_ROOT, ".venv", "bin", "python");

export default defineConfig({
  plugins: [
    vue(),
    {
      name: "predictions-api",
      configureServer(server) {
        // GET /api/cases → sorted list of case directories
        server.middlewares.use("/api/cases", (req, res, next) => {
          if (req.method !== "GET") return next();
          try {
            const entries = readdirSync(PREDICTIONS_DIR);
            const cases = entries
              .filter(name => {
                try {
                  if (!statSync(join(PREDICTIONS_DIR, name)).isDirectory()) return false;
                  statSync(join(PREDICTIONS_DIR, name, 'detections.json'));
                  return true;
                } catch { return false; }
              })
              .sort()
              .reverse(); // newest first (names are date-prefixed)
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(cases));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: err.message }));
          }
        });

        // POST /api/postprocess/:case → run ent_cv postprocess on demand
        server.middlewares.use("/api/postprocess", (req, res, next) => {
          if (req.method !== "POST") return next();
          let body = "";
          req.on("data", chunk => { body += chunk; });
          req.on("end", () => {
            try {
              const {
                caseName,
                method = "run_length",
                min_duration_sec = 1.0,
                gap_fill_sec = 1.0,
                window_sec = 3.0,
                vote_threshold = 0.5,
                confidence_threshold = 0.0,
              } = JSON.parse(body);

              // Validate inputs to prevent injection / path traversal
              if (!caseName || typeof caseName !== "string" || caseName.includes("..") || caseName.includes("/")) {
                res.statusCode = 400;
                res.setHeader("Content-Type", "application/json");
                return res.end(JSON.stringify({ ok: false, error: "Invalid case name" }));
              }
              const VALID_METHODS = ["run_length", "majority_vote", "gaussian"];
              if (!VALID_METHODS.includes(method)) {
                res.statusCode = 400;
                res.setHeader("Content-Type", "application/json");
                return res.end(JSON.stringify({ ok: false, error: "Invalid method" }));
              }

              const rawJson = join(PREDICTIONS_DIR, caseName, "detections.json");
              const proc = spawnSync(VENV_PYTHON, [
                "-m", "ent_cv.modeling.postprocess",
                "--raw-json", rawJson,
                "--method", method,
                "--min-duration-sec",     String(Number(min_duration_sec)),
                "--gap-fill-sec",         String(Number(gap_fill_sec)),
                "--window-sec",           String(Number(window_sec)),
                "--vote-threshold",       String(Number(vote_threshold)),
                "--confidence-threshold", String(Number(confidence_threshold)),
              ], { cwd: PROJECT_ROOT, encoding: "utf-8", timeout: 120_000 });

              if (proc.status !== 0) {
                res.statusCode = 500;
                res.setHeader("Content-Type", "application/json");
                return res.end(JSON.stringify({ ok: false, error: (proc.stderr || "postprocess failed").slice(0, 2000) }));
              }

              // Read freshly-written filtered_summary.json for display stats
              let summary = null;
              try {
                summary = JSON.parse(readFileSync(join(PREDICTIONS_DIR, caseName, "filtered_summary.json"), "utf-8"));
              } catch { /* not critical */ }

              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: true, summary }));
            } catch (err) {
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: false, error: err.message }));
            }
          });
        });

        // GET /api/raw/:case/:file.mp4 → serve with Range request support
        server.middlewares.use("/api/raw", (req, res, next) => {
          if (req.method !== "GET") return next();
          const safePath = req.url.split("/").filter(s => s && s !== "..").join("/");
          const filePath = join(RAW_DIR, safePath);
          try {
            const stat = statSync(filePath);
            if (!stat.isFile()) return next();
            const fileSize = stat.size;
            const range = req.headers["range"];
            if (range) {
              const [rawStart, rawEnd] = range.replace(/bytes=/, "").split("-");
              const start  = parseInt(rawStart, 10);
              const end    = rawEnd ? parseInt(rawEnd, 10) : fileSize - 1;
              const chunkSize = end - start + 1;
              res.writeHead(206, {
                "Content-Range":  `bytes ${start}-${end}/${fileSize}`,
                "Accept-Ranges":  "bytes",
                "Content-Length": chunkSize,
                "Content-Type":   "video/mp4",
              });
              createReadStream(filePath, { start, end }).pipe(res);
            } else {
              res.writeHead(200, {
                "Content-Length": fileSize,
                "Content-Type":   "video/mp4",
                "Accept-Ranges":  "bytes",
              });
              createReadStream(filePath).pipe(res);
            }
          } catch { next(); }
        });

        // fps cache: raw video path → fps number
        const fpsCacheMap = new Map();
        function probeVideoFps(videoPath) {
          if (fpsCacheMap.has(videoPath)) return fpsCacheMap.get(videoPath);
          const probe = spawnSync("ffprobe", [
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            videoPath,
          ]);
          const raw = probe.stdout?.toString().trim(); // e.g. "30000/1001" or "30/1"
          if (!raw) throw new Error(`ffprobe returned no fps for ${videoPath}`);
          const [num, den] = raw.split("/").map(Number);
          let fps = den ? num / den : num;
          // Snap to nearest standard framerate
          const standards = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];
          fps = standards.reduce((a, b) => Math.abs(b - fps) < Math.abs(a - fps) ? b : a);
          fpsCacheMap.set(videoPath, fps);
          return fps;
        }

        // frame-count cache: raw video path → integer frame count
        const frameCountCacheMap = new Map();
        function probeVideoFrameCount(videoPath) {
          if (frameCountCacheMap.has(videoPath)) return frameCountCacheMap.get(videoPath);
          // Fast path: nb_frames is stored in the MP4 container index (instant read)
          const probe1 = spawnSync("ffprobe", [
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "csv=p=0",
            videoPath,
          ]);
          const raw1 = probe1.stdout?.toString().trim();
          if (raw1 && raw1 !== "N/A" && /^\d+$/.test(raw1)) {
            const count = parseInt(raw1, 10);
            frameCountCacheMap.set(videoPath, count);
            return count;
          }
          // Slow fallback: count every packet (reads the whole file — avoid if possible)
          console.warn(`[predictions-api] nb_frames unavailable for ${videoPath}, falling back to -count_packets (slow)`);
          const probe2 = spawnSync("ffprobe", [
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            videoPath,
          ]);
          const raw2 = probe2.stdout?.toString().trim();
          if (!raw2) throw new Error(`ffprobe returned no frame count for ${videoPath}`);
          const count = parseInt(raw2, 10);
          frameCountCacheMap.set(videoPath, count);
          return count;
        }

        // GET /data/predictions/:case/... → serve detections.json (enriched) and JPEG frames
        server.middlewares.use("/data/predictions", (req, res, next) => {
          if (req.method !== "GET") return next();
          // Prevent path traversal
          const safePath = req.url.split("/").filter(s => s && s !== "..").join("/");
          const filePath = join(PREDICTIONS_DIR, safePath);
          try {
            const stat = statSync(filePath);
            if (!stat.isFile()) return next();
            const ext = filePath.split('.').pop().toLowerCase();
            if (ext === 'jpg' || ext === 'jpeg') {
              res.setHeader("Content-Type", "image/jpeg");
              createReadStream(filePath).pipe(res);
              return;
            }
            // For detections.json: enrich with fps + derived metadata
            if (safePath.endsWith("detections.json")) {
              const caseName = safePath.split("/")[0];
              const rawCaseDir = join(RAW_DIR, caseName);
              const mp4s = readdirSync(rawCaseDir).filter(f => f.toLowerCase().endsWith(".mp4")).sort();
              if (mp4s.length === 0) throw new Error(`No raw .mp4 files found for case ${caseName}`);

              // Try to read fps + total_frames from metadata.json first
              let fps, total_frames, cachedPartFrames;
              const metaPath = join(PREDICTIONS_DIR, caseName, "metadata.json");
              let metaRaw = {};
              try {
                metaRaw = JSON.parse(readFileSync(metaPath, "utf-8"));
                const meta = metaRaw;
                if (meta.fps != null) {
                  // fps may be a fraction string like "30000/1001"
                  const fpsStr = String(meta.fps);
                  if (fpsStr.includes("/")) {
                    const [num, den] = fpsStr.split("/").map(Number);
                    fps = den ? num / den : num;
                  } else {
                    fps = Number(fpsStr);
                  }
                  if (!fps || !isFinite(fps)) fps = null;
                }
                total_frames = meta.total_frames ?? null;
                // Cached per-part frame counts (keyed by filename, not full path)
                cachedPartFrames = meta.part_frames ?? null;
              } catch { /* no metadata */ }
              if (!fps) fps = probeVideoFps(join(rawCaseDir, mp4s[0]));

              let rawDetections = JSON.parse(readFileSync(filePath, "utf-8"));

              // Handle filtered_detections.json wrapper: { _filter: {...}, detections: [...] }
              let filterMetadata = null;
              if (!Array.isArray(rawDetections) && Array.isArray(rawDetections.detections)) {
                filterMetadata = rawDetections._filter ?? null;
                rawDetections = rawDetections.detections;
              }

              // Detect format: flat array {frame, class, name, confidence, box}
              //                vs grouped  [{frame, source, detections:[{class_id,…}]}]
              const isFlat = rawDetections.length > 0 && !Array.isArray(rawDetections[0]?.detections);

              // Derive ordered class list
              const classMap = new Map();
              if (isFlat) {
                for (const d of rawDetections) {
                  if (!classMap.has(d.class)) classMap.set(d.class, d.name);
                }
              } else {
                for (const r of rawDetections) {
                  for (const d of r.detections) {
                    if (!classMap.has(d.class_id)) classMap.set(d.class_id, d.class_name);
                  }
                }
              }
              // For filtered data: supplement classMap from the raw detections.json so
              // classes removed by filtering still appear (greyed out) in the frontend.
              if (filterMetadata !== null) {
                const rawDetPath = join(PREDICTIONS_DIR, caseName, "detections.json");
                try {
                  const rawArr = JSON.parse(readFileSync(rawDetPath, "utf-8"));
                  const arr = Array.isArray(rawArr) ? rawArr : (rawArr.detections ?? []);
                  for (const d of arr) {
                    if (!classMap.has(d.class)) classMap.set(d.class, d.name);
                  }
                } catch { /* raw file unavailable — use filtered classes only */ }
              }
              const sortedClassEntries = [...classMap.entries()].sort((a, b) => a[0] - b[0]);
              const classes = sortedClassEntries.map(([, name]) => name);
              // Remap original (sparse) class IDs → 0-based index into `classes`
              const classIndexRemap = new Map(sortedClassEntries.map(([origId], idx) => [origId, idx]));

              let partBoundaries = null;
              let results;
              if (isFlat) {
                // Build per-part cumulative frame-count boundaries for source assignment
                partBoundaries = []; // [{path, startFrame, endFrame}]
                let cumulative = 0;
                let wroteNewCounts = false;
                const newPartFrames = cachedPartFrames ? { ...cachedPartFrames } : {};
                for (const mp4 of mp4s) {
                  const mp4Path = join(rawCaseDir, mp4);
                  let count;
                  if (cachedPartFrames && cachedPartFrames[mp4] != null) {
                    // Use persisted count — no ffprobe needed
                    count = cachedPartFrames[mp4];
                    frameCountCacheMap.set(mp4Path, count);
                  } else {
                    count = probeVideoFrameCount(mp4Path);
                    newPartFrames[mp4] = count;
                    wroteNewCounts = true;
                  }
                  partBoundaries.push({ path: mp4Path, startFrame: cumulative, endFrame: cumulative + count - 1 });
                  cumulative += count;
                }
                // Persist newly probed counts so future requests skip ffprobe entirely
                if (wroteNewCounts) {
                  try {
                    writeFileSync(metaPath, JSON.stringify({ ...metaRaw, part_frames: newPartFrames }, null, 2));
                  } catch (e) {
                    console.warn("[predictions-api] Could not persist part_frames to metadata.json:", e.message);
                  }
                }
                function sourceForFrame(frame) {
                  for (const p of partBoundaries) {
                    if (frame <= p.endFrame) return p.path;
                  }
                  return partBoundaries[partBoundaries.length - 1].path;
                }

                // Group flat detections by frame
                const frameGroups = new Map();
                for (const d of rawDetections) {
                  if (!frameGroups.has(d.frame)) frameGroups.set(d.frame, []);
                  const b = d.box;
                  frameGroups.get(d.frame).push({
                    class_id: classIndexRemap.get(d.class) ?? d.class,
                    class_name: d.name,
                    confidence: d.confidence,
                    bbox: b ? [b.x1, b.y1, b.x2, b.y2] : null,
                  });
                }
                results = [...frameGroups.entries()]
                  .sort((a, b) => a[0] - b[0])
                  .map(([frame, detections]) => ({
                    frame,
                    source: sourceForFrame(frame),
                    detections,
                  }));
              } else {
                results = rawDetections;
              }

              if (!total_frames) total_frames = results.length;

              // Include part boundaries so the frontend can look up actual part
              // start frames even for frames that have no detection entries.
              const parts = isFlat
                ? partBoundaries.map(b => ({ source: b.path, startFrame: b.startFrame, endFrame: b.endFrame }))
                : null;

              const enriched = { fps, total_frames, classes, results, ...(parts ? { parts } : {}), ...(filterMetadata ? { _filter: filterMetadata } : {}) };
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(enriched));
              return;
            }
            res.setHeader("Content-Type", "application/json");
            createReadStream(filePath).pipe(res);
          } catch (err) { console.error("[predictions-api]", err); next(); }
        });
      },
    },
  ],
  server: {
    allowedHosts: ["entcv.jaehho.com"],
  },
});
