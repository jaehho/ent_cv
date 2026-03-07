# Codebase Concerns

**Analysis Date:** 2026-03-07

## Tech Debt

**Dual JSON format support in views.py:**
- Issue: `web/backend/api/views.py` `detections()` view must detect and branch on two mutually incompatible JSON schemas — Polars-format flat rows (from `detections.json`) and grouped-by-frame rows (from `filtered_detections.json`). The `is_flat` flag and dual-format detection logic at lines 227–259 are fragile and will silently misparse data if the format changes.
- Files: `web/backend/api/views.py` (lines 220–259)
- Impact: Any schema change in predict.py or postprocess.py output requires matching updates in two places; mismatches produce wrong detection rendering with no visible error.
- Fix approach: Standardize both `detections.json` and `filtered_detections.json` to the same grouped-by-frame schema, removing the dual-format detection path entirely.

**Module-level in-process caches are not thread-safe:**
- Issue: `_fps_cache` and `_frame_count_cache` (dicts at module level in `web/backend/api/views.py` lines 39–40) are mutated by request handlers without locks. Under a multi-threaded WSGI server (Gunicorn with threads or multiple workers), concurrent writes can corrupt entries or cause stale reads.
- Files: `web/backend/api/views.py` (lines 38–131)
- Impact: Race conditions under concurrent load; incorrect FPS or frame counts fed to the frontend cause wrong seek positions and detection frame alignment.
- Fix approach: Use `threading.Lock` around cache mutations, or replace with a proper cache backend (Django cache framework with LocMemCache or Redis).

**postprocess.py runs synchronously in a web request:**
- Issue: `web/backend/api/views.py` `postprocess_case()` (line 466–478) directly imports and calls `ent_cv.modeling.postprocess.postprocess`, which is a CPU-bound operation that iterates over entire detection arrays. This blocks a Django worker thread for the full duration.
- Files: `web/backend/api/views.py` (lines 464–478), `ent_cv/modeling/postprocess.py`
- Impact: Long-running surgical cases (thousands of frames) will time out or exhaust WSGI workers, making the server unresponsive to other requests.
- Fix approach: Dispatch postprocessing to a background task queue (Celery, Django-Q, or a simple subprocess) and return a job-status polling endpoint.

**`sys.path` mutation in settings.py:**
- Issue: `web/backend/config/settings.py` (lines 7–10) inserts the project root into `sys.path` at import time so `from ent_cv.modeling...` works inside Django. This couples the web app to the local development directory structure and breaks if the package is installed elsewhere.
- Files: `web/backend/config/settings.py` (lines 7–10)
- Impact: Fragile in Docker if the mount paths differ; masks import errors that proper packaging would catch.
- Fix approach: Install `ent_cv` as a proper editable package dependency in the backend's Dockerfile and remove the `sys.path` manipulation.

**`ent_cv/data/extract_frames.py` is 812 lines with no tests:**
- Issue: Largest Python file by line count; handles frame extraction, tiling, and CVAT integration. No corresponding test file found.
- Files: `ent_cv/data/extract_frames.py`
- Impact: High risk of silent regressions in data preparation pipeline.
- Fix approach: Extract distinct responsibilities into smaller modules, add unit tests for edge cases (empty video, partial frames, tiling boundaries).

## Known Bugs

**Unclosed file handles in `predictions_file` and `raw_video` views:**
- Symptoms: `FileResponse(open(file_path, "rb"), ...)` at lines 382 and 438 opens files without a `with` block; the non-range-request path in `raw_video` has the same pattern.
- Files: `web/backend/api/views.py` (lines 382, 438)
- Trigger: Any GET to `/api/cases/<case>/predictions/<path>` or `/api/cases/<case>/raw/<filename>` without a Range header.
- Workaround: Django's `FileResponse` will close the file when the response is consumed in normal operation, but early aborts (client disconnect) can leave the file descriptor open under CPython's GC-based cleanup. Not a crash but leaks descriptors under high load.

**`total_frames` fallback is incorrect for filtered data:**
- Symptoms: In `detections()` at line 331–332, `if not total_frames: total_frames = len(results)`. For filtered detections, `results` contains only frames with detections, so the total frame count equals the detection count rather than the actual video length. This causes the frontend timeline to be shorter than the video.
- Files: `web/backend/api/views.py` (lines 331–332)
- Trigger: Viewing filtered detections for a case where `metadata.json` does not contain `total_frames`.

**`partStartTs` computed from detections only — gaps at part boundaries:**
- Symptoms: In `YOLOVisualizer.vue`, `partStartTs` is built from the first frame number seen per source file. If the first frames of a video part have no detections, the start timestamp is wrong, causing video-seek drift.
- Files: `web/frontend/src/components/YOLOVisualizer.vue` (lines 745–753)
- Trigger: A video part where the first N frames have no instrument detections.
- Workaround: None currently; the `parts` array from the API provides ground-truth `startFrame`/`endFrame` but `partStartTs` ignores it.

## Security Considerations

**Debug mode defaults to `True`:**
- Risk: `DJANGO_DEBUG` defaults to `"True"` in `settings.py` (line 19), so a missing env var in any deployment context enables Django debug mode, exposing full tracebacks and settings.
- Files: `web/backend/config/settings.py` (line 19)
- Current mitigation: Production Docker compose sets `DJANGO_DEBUG=False` explicitly.
- Recommendations: Flip default to `False`; require opt-in for debug mode.

**`ALLOWED_HOSTS` hardcodes the public domain in source:**
- Risk: `entcv.jaehho.com` is hardcoded in `settings.py` (line 22) and `docker-compose.yml` (line 29). This is not a secret but commits the production hostname to source, making it harder to deploy to other environments without code changes.
- Files: `web/backend/config/settings.py` (line 22), `web/docker-compose.yml` (line 29)
- Current mitigation: Env var override is supported.
- Recommendations: Remove hardcoded hostname from source; rely solely on the env var with no default.

**CSRF cookie readable by JavaScript (`CSRF_COOKIE_HTTPONLY = False`):**
- Risk: The CSRF token must be readable by JS (line 115 in settings.py), which is standard for non-SPA Django but means XSS can read the token. The frontend manually extracts it from `document.cookie` in `YOLOVisualizer.vue` (line 1126).
- Files: `web/backend/config/settings.py` (line 115), `web/frontend/src/components/YOLOVisualizer.vue` (line 1126)
- Current mitigation: `SameSite=Lax` limits CSRF exposure; session cookie is HttpOnly.
- Recommendations: Acceptable for this architecture; document why `HttpOnly=False` is required.

## Performance Bottlenecks

**`ffprobe` slow-path frame count scan:**
- Problem: `_probe_video_frame_count()` in `web/backend/api/views.py` (lines 109–132) falls back to `-count_packets` with a 120-second timeout when the container index has no `nb_frames`. Long surgical videos (hours) may hit this on first load.
- Files: `web/backend/api/views.py` (lines 82–132)
- Cause: H.264 in some MP4 containers omits the frame count from the stream header.
- Improvement path: Cache probe results to `metadata.json` (partially done at lines 289–296); add a re-encode step during prediction to ensure frame count is indexed.

**Full detections.json loaded into memory per request:**
- Problem: Both `detections()` and `postprocess_case()` load entire JSON files into memory on each call. For large cases (hours of video at 30fps), `detections.json` can be several hundred MB.
- Files: `web/backend/api/views.py` (lines 211–213)
- Cause: Flat file storage with no indexing; no lazy reading.
- Improvement path: Store detections in a database or binary columnar format (Parquet); serve paginated or range-queried subsets.

**`YOLOVisualizer.vue` is 2,195 lines — single-file component at the limit:**
- Problem: All viewer logic, state management, canvas rendering, and postprocess controls live in one `.vue` file. Hot-reload is slow; scrolling to find code is difficult.
- Files: `web/frontend/src/components/YOLOVisualizer.vue`
- Cause: Incremental feature additions without refactoring.
- Improvement path: Split into focused sub-components: `RasterStrip.vue`, `PostprocessPanel.vue`, `VideoOverlay.vue`, `PlaybackControls.vue`.

## Fragile Areas

**Prediction frame filename convention is Ultralytics-internal:**
- Files: `web/frontend/src/components/YOLOVisualizer.vue` (lines 701–703)
- Why fragile: The prediction mode URL is constructed as `<partName>_frames/<partName>_<localFrame>.jpg` based on an assumed Ultralytics `save_frames` naming convention. Any Ultralytics version upgrade that changes output naming silently breaks prediction frame display.
- Safe modification: Pin Ultralytics version; add an integration test that verifies the directory structure after a predict run.

**`postprocess.py` requires `metadata.json` with an `fps` field:**
- Files: `ent_cv/modeling/postprocess.py` (lines 53–56, 505)
- Why fragile: If `metadata.json` is absent or lacks `fps` (e.g., prediction was run on images, not video), postprocessing raises with an unhelpful error. The web API at line 477–478 catches all exceptions and truncates the message to 2000 characters, potentially hiding the root cause.
- Safe modification: Validate metadata immediately after predict; emit an explicit error in the web response indicating the file is not suitable for postprocessing.

**Module-level `dotenv` load in `ent_cv/utils.py`:**
- Files: `ent_cv/utils.py` (lines 20–21)
- Why fragile: `load_dotenv(find_dotenv())` runs at import time, meaning importing any `ent_cv` module from Django (as done in `postprocess_case`) can silently override Django's already-loaded environment if a `.env` file is present in the project root.
- Safe modification: Call `load_dotenv` only in CLI entry points, not at module level.

## Scaling Limits

**File-system case storage:**
- Current capacity: Unlimited cases, one directory per case under `/mnt/data/ent_cv/predictions/`.
- Limit: `list_cases` calls `PREDICTIONS_DIR.iterdir()` on every request (line 143). With hundreds of cases this is still fast, but with thousands it becomes a blocking I/O call proportional to the number of directories.
- Scaling path: Introduce a database-backed case index; cache the directory listing.

**Single-worker postprocessing:**
- Current capacity: One postprocess job runs per Django worker (synchronous, blocking).
- Limit: Two simultaneous postprocess requests starve each other for WSGI workers.
- Scaling path: Task queue (see tech debt item above).

## Test Coverage Gaps

**`YOLOVisualizer.vue` has no frontend tests:**
- What's not tested: Canvas rendering logic, frame-seek accuracy, part-boundary transitions, postprocess form submission, jump-filter correctness.
- Files: `web/frontend/src/components/YOLOVisualizer.vue`
- Risk: Regressions in core playback and overlay logic are invisible until manual testing.
- Priority: High

**`ent_cv/modeling/postprocess.py` has no unit tests:**
- What's not tested: All three filtering methods (`run_length`, `majority_vote`, `gaussian`), edge cases (zero detections, single-frame runs, all-frames detected).
- Files: `ent_cv/modeling/postprocess.py`
- Risk: Temporal filtering bugs produce incorrect clinical output (wrong instrument presence windows).
- Priority: High

**`ent_cv/data/extract_frames.py` (812 lines) has no tests:**
- What's not tested: Frame extraction, tiling, CVAT integration helpers.
- Files: `ent_cv/data/extract_frames.py`
- Risk: Silent data corruption in the annotation pipeline.
- Priority: Medium

---

*Concerns audit: 2026-03-07*
