# Codebase Structure

**Analysis Date:** 2026-03-07

## Directory Layout

```
ent_cv/                          # Root of repo
├── ent_cv/                      # Python package: ML pipeline
│   ├── config.py                # Central constants: paths, LABELS, metric keys
│   ├── utils.py                 # @notify email decorator
│   ├── modeling/                # YOLO modeling commands
│   │   ├── cli.py               # Typer entry point (ent-cv command)
│   │   ├── predict.py           # YOLO inference → detections.json
│   │   ├── postprocess.py       # Temporal filtering → filtered_detections.json
│   │   ├── train.py             # Model training
│   │   ├── val.py               # Model validation
│   │   ├── tune.py              # Hyperparameter tuning
│   │   ├── benchmark.py         # Format benchmarks
│   │   ├── export.py            # Model export
│   │   ├── compare_models.py    # Rank models by metrics
│   │   ├── batch.py             # YAML-driven batch ops
│   │   ├── prepare_dataset.py   # Train/val splitting
│   │   └── configs/             # YOLO config YAML files
│   ├── data/                    # Data pipeline utilities
│   │   ├── extract_frames.py    # Frame extraction from video
│   │   ├── tile_frames.py       # Frame tiling
│   │   ├── cvat/                # CVAT annotation integration
│   │   └── sharepoint/          # SharePoint data sync
│   └── serverless/              # Nuclio serverless functions for CVAT AI
│       └── nuclio/
├── web/                         # Web application
│   ├── backend/                 # Django backend
│   │   ├── config/              # Django project config
│   │   │   ├── settings.py      # All settings (dev/prod branching)
│   │   │   ├── urls.py          # Root URL conf (includes api.urls + auth)
│   │   │   └── wsgi.py          # WSGI entry point
│   │   ├── api/                 # Django app
│   │   │   ├── views.py         # All API view functions (single file)
│   │   │   ├── urls.py          # API URL patterns
│   │   │   ├── models.py        # Minimal (no detection models; only Django auth)
│   │   │   └── tests/           # pytest tests for API
│   │   └── staticfiles/         # Collected static (generated, committed)
│   ├── frontend/                # Vue 3 SPA
│   │   ├── src/
│   │   │   ├── main.js          # Vue app entry point
│   │   │   ├── App.vue          # Root component
│   │   │   ├── router.js        # Vue Router (2 routes)
│   │   │   ├── components/
│   │   │   │   ├── YOLOVisualizer.vue  # Main video + detection overlay viewer
│   │   │   │   ├── CasePicker.vue      # Case listing and selection
│   │   │   │   ├── LoginForm.vue       # Login UI
│   │   │   │   └── index.js            # Component barrel export
│   │   │   ├── utils/           # Shared JS utilities
│   │   │   └── __tests__/       # Vitest tests
│   │   ├── dist/                # Built assets (generated, not in git)
│   │   ├── vite.config.js       # Vite config with /api/ and /auth/ proxy
│   │   └── package.json
│   └── docker-compose.yml       # Production stack (Postgres + Django + Nginx)
├── reports/                     # Analysis outputs
│   └── figures/                 # Matplotlib figures
├── cvat/                        # CVAT git submodule (annotation tool fork)
├── pyproject.toml               # Package definition, dependencies, tool config
├── Makefile                     # All dev and prod commands
└── .planning/                   # GSD planning documents
    └── codebase/
```

## Directory Purposes

**`ent_cv/`:**
- Purpose: Installable Python package containing the entire ML pipeline
- Contains: Config, modeling commands, data utilities, serverless functions
- Key files: `ent_cv/config.py` (all shared constants), `ent_cv/modeling/cli.py` (CLI entry)

**`ent_cv/modeling/`:**
- Purpose: All YOLO-related operations exposed as CLI subcommands
- Contains: One module per command; each exports a `main()` function registered in `cli.py`
- Key files: `predict.py`, `postprocess.py`, `batch.py`

**`web/backend/api/`:**
- Purpose: The single Django app; contains all API logic
- Contains: `views.py` (all endpoints), `urls.py` (routing), minimal `models.py`
- Key files: `web/backend/api/views.py` — this is the only file with API logic

**`web/frontend/src/components/`:**
- Purpose: Vue SFC components
- Contains: `YOLOVisualizer.vue` (main viewer), `CasePicker.vue` (landing), `LoginForm.vue`

**`web/frontend/src/utils/`:**
- Purpose: Shared JavaScript utility functions used across components

## Key File Locations

**Entry Points:**
- `ent_cv/modeling/cli.py`: CLI entry point — all `ent-cv` subcommands registered here
- `web/backend/config/wsgi.py`: Django WSGI entry for production/gunicorn
- `web/frontend/src/main.js`: Vue app bootstrap

**Configuration:**
- `ent_cv/config.py`: All filesystem paths and the `LABELS` class list (order = YOLO class ID)
- `web/backend/config/settings.py`: Django settings, DB selection (SQLite dev / Postgres prod)
- `pyproject.toml`: Package deps, pytest config, ruff config, entry point declaration
- `Makefile`: All runnable commands for development and production

**Core Logic:**
- `ent_cv/modeling/predict.py`: YOLO inference execution
- `ent_cv/modeling/postprocess.py`: Temporal filtering methods
- `web/backend/api/views.py`: All REST API endpoints

**Testing:**
- `web/backend/api/tests/`: pytest tests for Django API
- `web/frontend/src/__tests__/`: Vitest tests for frontend

## Naming Conventions

**Files:**
- Python modules: `snake_case.py`
- Vue components: `PascalCase.vue`
- JS utils: `camelCase.js` or `snake_case.js`

**Directories:**
- Python: `snake_case/`
- Web: `lowercase/` for infrastructure dirs, component dirs match their content

## Where to Add New Code

**New CLI modeling command:**
- Implementation: `ent_cv/modeling/<command_name>.py` with a `main()` function
- Registration: Add `app.command(name="...", ...)(module.main)` in `ent_cv/modeling/cli.py`

**New API endpoint:**
- View function: Add to `web/backend/api/views.py`
- Route: Add to `web/backend/api/urls.py`
- Tests: Add to `web/backend/api/tests/`

**New Vue component:**
- Implementation: `web/frontend/src/components/<ComponentName>.vue`
- Export: Add to `web/frontend/src/components/index.js`
- Route (if page-level): Register in `web/frontend/src/router.js`

**Shared JS utilities:**
- Location: `web/frontend/src/utils/`

**New shared Python constants:**
- Location: `ent_cv/config.py`

## Special Directories

**`cvat/`:**
- Purpose: Git submodule — fork of CVAT annotation tool
- Generated: No (external repo)
- Committed: Submodule reference only

**`web/frontend/dist/`:**
- Purpose: Vite production build output
- Generated: Yes (`make web-build`)
- Committed: No

**`web/backend/staticfiles/`:**
- Purpose: Django `collectstatic` output (admin assets)
- Generated: Yes
- Committed: Yes (admin CSS/JS included for Docker build)

**`/mnt/data/ent_cv/` (external mount):**
- Purpose: All large data — raw videos, predictions, datasets, models
- Structure: `raw/<case>/`, `predictions/<case>/`, `datasets/`, `models/`
- Generated: Yes (runtime data)
- Committed: No

---

*Structure analysis: 2026-03-07*
