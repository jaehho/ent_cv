import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { readdirSync, statSync, createReadStream, existsSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { join, resolve, dirname } from "node:path";
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

        // GET /api/prediction-frame/:case/:file?n=<frame_index> → extract single JPEG from MJPEG AVI
        // Uses frame index (not timestamp) to avoid drift caused by AVI fps != source fps.
        // AVI fps is probed once per file and cached.
        const aviFpsCache = new Map();
        function getAviFps(aviPath) {
          if (aviFpsCache.has(aviPath)) return aviFpsCache.get(aviPath);
          const result = spawnSync("ffprobe", [
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            aviPath,
          ]);
          const raw = result.stdout?.toString().trim(); // e.g. "29/1" or "30000/1001"
          let fps = 30;
          if (raw) {
            const [num, den] = raw.split("/").map(Number);
            fps = den ? num / den : num;
          }
          aviFpsCache.set(aviPath, fps);
          return fps;
        }

        server.middlewares.use("/api/prediction-frame", (req, res, next) => {
          if (req.method !== "GET") return next();
          const [path] = req.url.split("?");
          const safePath = path.split("/").filter(s => s && s !== "..").join("/");
          const aviPath = join(PREDICTIONS_DIR, safePath.replace(/\.(avi|mp4)$/i, "")) + ".avi";
          if (!existsSync(aviPath)) return next();

          // Prefer ?n=FRAME_INDEX (exact); fall back to ?t=SECONDS (legacy)
          const nMatch = req.url.match(/[?&]n=(\d+)/);
          const tMatch = req.url.match(/[?&]t=([\d.]+)/);
          let seekSec;
          if (nMatch) {
            const frameIndex = parseInt(nMatch[1], 10);
            const aviFps = getAviFps(aviPath);
            // Seek to the MIDDLE of the target frame's time window.
            // Frame N spans [N/fps, (N+1)/fps). Seeking to exactly N/fps can
            // land on frame N-1 due to floating-point rounding in ffmpeg's
            // demuxer, causing a 1-frame drift. Adding 0.5 frames of padding
            // places the seek point safely inside frame N's window.
            seekSec = (frameIndex + 0.5) / aviFps;
          } else {
            seekSec = tMatch ? parseFloat(tMatch[1]) : 0;
          }

          res.setHeader("Content-Type", "image/jpeg");
          res.setHeader("Cache-Control", "public, max-age=300");

          const ff = spawn("ffmpeg", [
            "-ss", String(seekSec),
            "-i", aviPath,
            "-frames:v", "1",
            "-f", "image2",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "pipe:1",
          ], { stdio: ["ignore", "pipe", "ignore"] });

          ff.stdout.pipe(res);
          req.on("close", () => ff.kill());
          ff.on("error", () => { if (!res.headersSent) res.end(); });
        });

        // GET /data/predictions/:case/detections.json → serve file
        server.middlewares.use("/data/predictions", (req, res, next) => {
          if (req.method !== "GET") return next();
          // Prevent path traversal
          const safePath = req.url.split("/").filter(s => s && s !== "..").join("/");
          const filePath = join(PREDICTIONS_DIR, safePath);
          try {
            const stat = statSync(filePath);
            if (!stat.isFile()) return next();
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
