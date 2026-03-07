# Architecture

**Analysis Date:** 2026-03-07

## Pattern Overview

**Overall:** Two-subsystem monorepo — a Python ML pipeline (CLI-driven) and a Django+Vue web application. The subsystems share a common filesystem mount at `/mnt/data/ent_cv/` as their integration point; there is no direct in-process coupling between the CLI pipeline and the web server except when the web backend directly imports `ent_cv.modeling.postprocess`.

**Key Characteristics:**
- Filesystem-as-integration-layer: CLI pipeline writes JSON artifacts; Django reads them
- No ORM models for detections — all detection data lives in flat JSON files on disk
- Django used as a thin API server only (no DRF, plain function-based views, no model layer for video data)
- Vue 3 SPA with two routes; all visualization logic lives in a single large component
- Auth is Django session-based; frontend communicates via the same cookie

## Layers

**ML Pipeline (offline):**
- Purpose: Train YOLO models, run inference, post-process temporal detections
- Location: `ent_cv/`
- Contains: Typer CLI commands, YOLO wrappers, Polars-based data manipulation, data pipeline utilities
- Depends on: Ultralytics, Polars, ffprobe (subprocess), `/mnt/data/ent_cv/` filesystem
- Used by: Makefile targets, direct `uv run ent-cv` invocations

**Django API Backend:**
- Purpose: Serve detection JSON and raw video to the frontend; trigger postprocessing on demand
- Location: `web/backend/`
- Contains: Single `api/views.py` with all view functions, URL routing in `api/urls.py`, Django settings in `config/settings.py`
- Depends on: Filesystem at `/mnt/data/ent_cv/predictions/` and `/mnt/data/ent_cv/raw/`, ffprobe (subprocess), ent_cv package (imported at runtime for postprocess)
- Used by: Vue frontend via HTTP

**Vue Frontend:**
- Purpose: Video playback with overlaid bounding boxes and instrument timelines
- Location: `web/frontend/src/`
- Contains: Two route-level components (`CasePicker.vue`, `YOLOVisualizer.vue`), `LoginForm.vue`, shared utils under `src/utils/`
- Depends on: Django API via Vite proxy (dev) or direct requests (prod)
- Used by: End users in browser

## Data Flow

**Inference Pipeline:**
1. Raw `.mp4` video parts placed in `/mnt/data/ent_cv/raw/<case>/`
2. `ent-cv predict` runs Ultralytics YOLO → writes `detections.json` (flat per-detection records) and `metadata.json` (fps, total_frames, part_frames) to `/mnt/data/ent_cv/predictions/<case>/`
3. `ent-cv postprocess` reads `detections.json` → applies temporal filtering (run_length / majority_vote / gaussian) → writes `filtered_detections.json` and `filtered_summary.json`

**Web Viewing Flow:**
1. Frontend `CasePicker` fetches `GET /api/cases/` → lists directories in predictions that have `detections.json`
2. User selects a case → `YOLOVisualizer` fetches `GET /api/cases/<case>/detections/?mode=filtered`
3. Django `detections` view reads JSON files, enriches with fps (from metadata.json or ffprobe) and per-part frame boundaries, remaps class IDs, returns structured JSON
4. Frontend plays raw video via `GET /api/cases/<case>/raw/<filename>` (HTTP Range supported for seeking)
5. On each video frame, frontend draws bounding boxes from the detections payload

**On-Demand Postprocessing:**
1. Frontend sends `POST /api/cases/<case>/postprocess/` with method and parameters
2. Django imports `ent_cv.modeling.postprocess.postprocess` at runtime and calls it directly
3. Response includes `filtered_summary.json` contents

**State Management:**
- No client-side state library; Vue component-local reactive state
- Django module-level `_fps_cache` and `_frame_count_cache` dicts persist across requests (in-process cache, not shared across workers)

## Key Abstractions

**Case:**
- Purpose: A single surgical case; the primary organizational unit
- Examples: `20251113_02`, `20251124_01`
- Pattern: A directory name matching `^[a-zA-Z0-9_-]+$`; must exist under both `/mnt/data/ent_cv/raw/` and `/mnt/data/ent_cv/predictions/`

**Detections JSON (flat format):**
- Purpose: Per-frame, per-instrument detection records from YOLO
- Location: `/mnt/data/ent_cv/predictions/<case>/detections.json`
- Pattern: List of `{frame, class, name, confidence, box: {x1,y1,x2,y2}}` objects

**Enriched Detections (API response):**
- Purpose: Frontend-ready payload combining detections + video metadata + part boundaries
- Pattern: `{fps, total_frames, classes[], results[], parts[], _filter}`

**LABELS list:**
- Purpose: Maps YOLO class ID (index) to human-readable instrument name
- Location: `ent_cv/config.py` — index position is significant and must not be reordered

## Entry Points

**CLI:**
- Location: `ent_cv/modeling/cli.py`
- Triggers: `uv run ent-cv <command>` or Makefile targets
- Responsibilities: Flat Typer commands delegating to module `main()` functions

**Django WSGI:**
- Location: `web/backend/config/wsgi.py`
- Triggers: Gunicorn in production, `manage.py runserver` in dev
- Responsibilities: Serves all API and auth routes

**Vue SPA:**
- Location: `web/frontend/src/main.js`
- Triggers: Browser loading `index.html`
- Responsibilities: Mounts Vue app, sets up router

## Error Handling

**Strategy:** Fail-fast with HTTP status codes in the API; catch-and-log in CLI pipeline.

**Patterns:**
- All API views validate case name with `_validate_case_name()` before any filesystem access
- Views return `HttpResponseBadRequest` (400), `HttpResponseNotFound` (404), or `JsonResponse({error}, 500)` as appropriate
- `@api_login_required` decorator returns `{error: "Authentication required"}` with 401 instead of redirect
- CLI uses loguru for structured logging; `@notify` decorator emails on job completion/failure

## Cross-Cutting Concerns

**Logging:** loguru in Python package; integrated with tqdm.write to avoid progress-bar corruption
**Validation:** Case name regex `^[a-zA-Z0-9_-]+$` enforced in every API view before filesystem access; path traversal checked via `.resolve().relative_to()`
**Authentication:** Django session auth; custom cookie names `entcv_sessionid` / `entcv_csrftoken`; all API endpoints guarded by `@api_login_required`

---

*Architecture analysis: 2026-03-07*
