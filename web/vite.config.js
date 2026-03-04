import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { readdirSync, statSync, createReadStream, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PREDICTIONS_DIR = "/mnt/data/ent_cv/predictions";
const RAW_DIR         = "/mnt/data/ent_cv/raw";

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
                try { return statSync(join(PREDICTIONS_DIR, name)).isDirectory(); }
                catch { return false; }
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
            // For detections.json: enrich the flat results array with fps + derived metadata
            if (safePath.endsWith("detections.json")) {
              const results = JSON.parse(readFileSync(filePath, "utf-8"));
              // Derive ordered class list from detection data
              const classMap = new Map();
              for (const r of results) {
                for (const d of r.detections) {
                  if (!classMap.has(d.class_id)) classMap.set(d.class_id, d.class_name);
                }
              }
              const classes = [...classMap.entries()]
                .sort((a, b) => a[0] - b[0])
                .map(([, name]) => name);
              // Probe fps from the first raw video in this case
              const caseName = safePath.split("/")[0];
              const rawCaseDir = join(RAW_DIR, caseName);
              const mp4s = readdirSync(rawCaseDir).filter(f => f.toLowerCase().endsWith(".mp4")).sort();
              if (mp4s.length === 0) throw new Error(`No raw .mp4 files found for case ${caseName}`);
              const fps = probeVideoFps(join(rawCaseDir, mp4s[0]));
              const enriched = { fps, total_frames: results.length, classes, results };
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify(enriched));
              return;
            }
            res.setHeader("Content-Type", "application/json");
            createReadStream(filePath).pipe(res);
          } catch { next(); }
        });
      },
    },
  ],
  server: {
    allowedHosts: ["entcv.jaehho.com"],
  },
});
