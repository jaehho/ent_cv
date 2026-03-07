# External Integrations

**Analysis Date:** 2026-03-07

## APIs & External Services

**CVAT (Computer Vision Annotation Tool):**
- Self-hosted fork at `github.com/jaehho/cvat` (git submodule at `cvat/`)
- Used for labeling surgical instrument frames
- SDK: `cvat-sdk` and `cvat-cli` Python packages
- Integration code: `ent_cv/data/cvat/`
- Hosted at: `cvat.jaehho.com` (set via `CVAT_HOST` in `Makefile`)
- Docker Compose: `cvat/docker-compose.yml` + serverless components

**Nuclio (Serverless Functions for CVAT AI):**
- Deployed via `ent_cv/serverless/` for CVAT AI model integration
- Runs inside the CVAT Docker Compose stack

**Gmail SMTP:**
- Used for job completion/failure notifications
- Implementation: `ent_cv/utils.py` — `send_email()` and `@notify` decorator
- Auth: app password (not OAuth)
- Env vars: `NOTIFY_EMAIL`, `NOTIFY_APP_PASSWORD`
- SMTP host: `smtp.gmail.com:465` (SSL)
- Sends email to self (same address as sender)

## Data Storage

**Databases:**
- Development: SQLite at `web/backend/db.sqlite3`
- Production: PostgreSQL 16 (Docker image `postgres:16-alpine`)
  - Connection: `DATABASE_URL` env var (format: `postgres://user:pass@host:port/dbname`)
  - Prod database name: `entcv`, user: `entcv`
  - Client: `psycopg2-binary` (prod uses `psycopg[binary]` via `.[db]` optional dep)
  - Django config: `web/backend/config/settings.py` (manual URL parsing, no dj-database-url)

**File Storage:**
- Local filesystem only — all data at `/mnt/data/ent_cv/`
- `predictions/` — YOLO output JSONs (read-write, bind-mounted in Docker)
- `raw/` — original surgical `.mp4` videos (read-only in production Docker)
- `models/` — YOLO `.pt` weight files
- `datasets/` — labeled training datasets

**Caching:**
- No external cache (Redis/Memcached not used)
- Module-level in-process caches in `web/backend/api/views.py`:
  - `_fps_cache: dict[str, float]` — ffprobe FPS results
  - `_frame_count_cache: dict[str, int]` — per-video frame counts

## Authentication & Identity

**Auth Provider:**
- Django's built-in session auth (no third-party identity provider)
- Implementation: `web/backend/config/views.py` — `login_view`, `logout_view`, `session_view`
- Session cookie: `entcv_sessionid` (custom name, 1-week expiry)
- CSRF cookie: `entcv_csrftoken` (JS-readable, `HttpOnly=False`)
- All API endpoints protected by `@api_login_required` decorator (`web/backend/api/views.py`) — returns 401 JSON instead of redirect

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Datadog, etc.)

**Logs:**
- Python: `loguru` throughout `ent_cv/` package; configured to use `tqdm.write` when tqdm is available (`ent_cv/config.py`)
- Django: standard Django logging (not explicitly configured)
- Production logs: `make web-prod-logs` tails Docker Compose logs

## CI/CD & Deployment

**Hosting:**
- Production: self-hosted Docker Compose stack
- Public access: Cloudflare tunnel to `entcv.jaehho.com`

**CI Pipeline:**
- None detected (no GitHub Actions, CircleCI, etc.)

**Build process:**
- Frontend: `npm run build` (Vite) → static files served by Nginx
- Backend: `pip install .[db]` in `web/backend/Dockerfile` → gunicorn with 2 workers

## Environment Configuration

**Required env vars:**
- `DJANGO_SECRET_KEY` — Django secret (required in production, auto-generated insecure key in dev)
- `POSTGRES_PASSWORD` — PostgreSQL password (required for Docker Compose)
- `DATABASE_URL` — PostgreSQL connection string (omit to use SQLite in dev)

**Optional env vars:**
- `NOTIFY_EMAIL` — Gmail address for job notifications
- `NOTIFY_APP_PASSWORD` — Gmail app password
- `DJANGO_DEBUG` — defaults to `True`
- `DJANGO_ALLOWED_HOSTS` — comma-separated; defaults include `entcv.jaehho.com`
- `CSRF_TRUSTED_ORIGINS` — comma-separated; defaults include Vite dev origin and production domain
- `PREDICTIONS_DIR` — override default `/mnt/data/ent_cv/predictions`
- `RAW_DIR` — override default `/mnt/data/ent_cv/raw`

**Secrets location:**
- `.env` file at repo root (not committed; listed in `.gitignore` implied by absence from git)
- `.envrc` at repo root activates `.venv` only (no secrets)

## Webhooks & Callbacks

**Incoming:**
- `POST /api/cases/<case>/postprocess/` — triggers YOLO post-processing (internal, not external webhook)

**Outgoing:**
- Gmail SMTP notifications on ML job completion/failure (`ent_cv/utils.py`)

## System Tool Dependencies

**ffprobe / ffmpeg:**
- `ffprobe` called via `subprocess` in `web/backend/api/views.py` to extract video FPS and frame counts
- Installed in `web/backend/Dockerfile` via `apt-get install ffmpeg`
- Required on developer machine for local Django dev

---

*Integration audit: 2026-03-07*
