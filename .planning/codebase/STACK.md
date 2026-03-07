# Technology Stack

**Analysis Date:** 2026-03-07

## Languages

**Primary:**
- Python 3.12 — ML pipeline (`ent_cv/`), Django backend (`web/backend/`)
- JavaScript (ES modules, no TypeScript) — Vue 3 frontend (`web/frontend/`)

**Secondary:**
- SQL — SQLite (dev) / PostgreSQL 16 (prod)

## Runtime

**Environment:**
- Python 3.12 (pinned in `.python-version`)
- Node.js 23.11.0 (frontend build and dev server)

**Package Manager:**
- Python: `uv` 0.9.8 — lockfile `uv.lock` present
- JavaScript: `npm` 11.4.2 — `package-lock.json` present

## Frameworks

**Core:**
- Django 6.0.3 — backend API (`web/backend/`)
- Vue 3.4 — frontend SPA (`web/frontend/src/`)
- Ultralytics YOLO — object detection inference (`ent_cv/modeling/predict.py`)

**Testing:**
- `pytest` + `pytest-django` + `pytest-cov` — Python backend tests
- `vitest` 2.x — frontend unit tests
- `@vue/test-utils` 2.4 — Vue component testing

**Build/Dev:**
- Vite 5.x — frontend dev server and production build
- Gunicorn — WSGI server in production (`web/backend/Dockerfile`)
- Nginx (alpine) — static file serving for built frontend (`web/frontend/Dockerfile`)
- Docker Compose — production stack orchestration (`web/docker-compose.yml`)

## Key Dependencies

**Critical:**
- `ultralytics` — YOLO model training and inference; core ML functionality
- `torch` — PyTorch deep learning backend for YOLO
- `polars` — columnar data format for `detections.json` output (`ent_cv/modeling/predict.py`)
- `django` 6.x — web framework for API and auth
- `whitenoise` — static file serving middleware in Django (`web/backend/config/settings.py`)
- `psycopg2-binary` / `psycopg[binary]` — PostgreSQL adapter

**Infrastructure:**
- `typer` — CLI framework for `ent-cv` command (`ent_cv/modeling/cli.py`)
- `loguru` — structured logging throughout `ent_cv/`
- `python-dotenv` — `.env` loading in `ent_cv/config.py` and `ent_cv/utils.py`
- `opencv-python` — video/image processing
- `onnx` + `onnxruntime-gpu` + `onnxslim` — ONNX model export and inference
- `tensorflow` + `tensorrt` — alternate inference backends
- `cvat-cli` + `cvat-sdk` — CVAT annotation platform integration
- `vue-router` 4.6 — client-side routing (`web/frontend/src/router.js`)
- `@vitejs/plugin-vue` 5.x — Vite plugin for `.vue` SFC compilation
- `eslint` 9.x + `eslint-plugin-vue` — JavaScript linting
- `ruff` — Python linting and import sorting

## Configuration

**Environment:**
- `.env` file at repo root (not committed) loaded via `python-dotenv`
- Required: `DJANGO_SECRET_KEY`, `POSTGRES_PASSWORD`
- Optional: `DATABASE_URL`, `NOTIFY_EMAIL`, `NOTIFY_APP_PASSWORD`
- Django env vars: `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `PREDICTIONS_DIR`, `RAW_DIR`
- Dev shortcut: `.envrc` activates the `.venv` via direnv

**Build:**
- `pyproject.toml` — Python project metadata, dependencies, pytest config, ruff config, pyright config
- `web/frontend/vite.config.js` — Vite config with dev proxy to Django
- `web/docker-compose.yml` — production Docker Compose stack
- `web/backend/Dockerfile` — Python 3.12-slim + ffmpeg + gunicorn
- `web/frontend/Dockerfile` — Node 22 build stage → Nginx alpine serve stage

## Platform Requirements

**Development:**
- Linux (data lives at `/mnt/data/ent_cv/`)
- CUDA GPU recommended (inference uses `device=0`)
- `tmux` for `make web-dev` multi-pane session
- `ffmpeg` / `ffprobe` for video metadata in Django views

**Production:**
- Docker + Docker Compose
- Exposed publicly via Cloudflare tunnel at `entcv.jaehho.com`
- Django on `:8787`, Nginx/frontend on `:8050`

---

*Stack analysis: 2026-03-07*
