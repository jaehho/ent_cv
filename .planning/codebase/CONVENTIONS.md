# Coding Conventions

**Analysis Date:** 2026-03-07

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (e.g., `predict.py`, `postprocess.py`, `extract_frames.py`)
- Vue components: `PascalCase.vue` (e.g., `YOLOVisualizer.vue`, `CasePicker.vue`, `LoginForm.vue`)
- JavaScript utilities: `index.js` barrel files inside named directories (e.g., `src/utils/index.js`)
- Test files (Python): `test_<subject>.py` inside `tests/` subdirectory
- Test files (JS): `<subject>.spec.js` inside `src/__tests__/`

**Functions:**
- Python: `snake_case` for all functions (e.g., `_probe_video_fps`, `_validate_case_name`, `run`)
- Python private/internal helpers: leading underscore prefix (e.g., `_fps_from_file`, `_get_fps`, `_validate_case_name`)
- JavaScript/Vue: `camelCase` (e.g., `formatTime`, `formatRealTime`, `seekFiltered`, `togglePlay`)

**Variables:**
- Python: `snake_case` (e.g., `case_dir`, `predictions_dir`, `raw_dir`)
- Python module-level constants: `UPPER_SNAKE_CASE` (e.g., `PREDICTIONS_DIR`, `RAW_DIR`, `_STANDARD_FPS`)
- JavaScript: `camelCase` (e.g., `leftPanelWidth`, `jumpFilter`, `playbackRate`)

**Types/Classes:**
- Python: `PascalCase` (e.g., `TestListCases`, `TestDetections`)
- Regex constants: `_UPPER_SNAKE_CASE` with leading underscore (e.g., `_CASE_NAME_RE`, `_VALID_METHODS`)

## Code Style

**Formatting (Python):**
- Tool: Ruff (`uv run ruff check .` and `uv run ruff format --check .`)
- Line length: 99 characters (`line-length = 99` in `pyproject.toml`)
- Import sorting: enabled via `extend-select = ["I"]` (isort rules)
- First-party package: `ent_cv`
- `force-sort-within-sections = true`

**Formatting (JavaScript/Vue):**
- Tool: ESLint with `eslint-plugin-vue` (flat config at `web/frontend/eslint.config.js`)
- Base: `@eslint/js` recommended + `eslint-plugin-vue` flat/recommended
- Single-word component names: allowed (`vue/multi-word-component-names` is off)
- Unused vars: warn level, underscore-prefixed args ignored (`argsIgnorePattern: "^_"`)

## Import Organization

**Python order (enforced by Ruff isort):**
1. Standard library (`json`, `os`, `pathlib`, `re`, `subprocess`, `functools`)
2. Third-party (`django`, `pytest`, `polars`, `typer`, `loguru`, `ultralytics`)
3. First-party (`ent_cv.*`)

**JavaScript order (convention observed):**
1. Framework imports (`vitest`, `@vue/test-utils`)
2. Component/local imports (`../App.vue`, `../utils/index.js`)

## Error Handling

**Python backend views:**
- Return `JsonResponse({"error": "..."}, status=<code>)` for all API errors
- Use Django response helpers: `HttpResponseBadRequest`, `HttpResponseNotFound`
- Custom `@api_login_required` decorator returns 401 JSON instead of redirect
- Input validation done at top of each view function before processing

**Python modeling code:**
- `subprocess.run` with `capture_output=True, text=True` for external processes
- `try/except (subprocess.CalledProcessError, FileNotFoundError)` for optional external tool calls (ffprobe)
- `raise RuntimeError(...)` for unrecoverable ffprobe failures

## Logging

**Framework:** `loguru` (Python modeling code)
- Import: `from loguru import logger`
- Usage: `logger.info(...)`, `logger.warning(...)` etc.
- Django views use no explicit logging (relies on Django's default logging)

## Comments

**When to Comment:**
- Section separators using `# ──` dash lines for visual grouping (used in `views.py`)
- Docstrings on functions that have non-obvious behavior
- Inline comments explaining non-obvious logic (e.g., "Snap to nearest standard framerate")
- Fixture docstrings describe what the fixture creates

**JSDoc/TSDoc:**
- Not used (no TypeScript; JavaScript codebase uses plain comments)

**Vue template comments:**
- HTML comment blocks (`<!-- ── Section Name ──... -->`) used for template section grouping in `YOLOVisualizer.vue`

## Function Design

**Size:** Private helper functions are small and single-purpose (e.g., `_validate_case_name`, `_fps_from_file`). Django view functions are larger and encompass full request handling.

**Parameters:** Default values provided for optional parameters (e.g., `conf: float = 0.647`, `iou: float = 0.7`).

**Return Values:**
- Python API views always return a Django response object
- Helper functions return `Optional[...]` with `None` for missing/inapplicable values (e.g., `_get_fps`)
- Boolean validators return `bool` (e.g., `_validate_case_name`)

## Module Design

**Exports (Python):**
- No explicit `__all__`; public API implied by underscore convention
- Module-level caches (`_fps_cache`, `_frame_count_cache`) persist across requests intentionally

**Exports (JavaScript):**
- Named exports from utility modules (`export function`, `export const`)
- Barrel files: `src/utils/index.js` exports multiple named utilities

**Django views:**
- All API logic in a single `api/views.py` file (no DRF, plain function-based views)
- Decorators applied: `@require_GET`, `@require_POST`, `@api_login_required`

## Path Handling

- Python uses `pathlib.Path` throughout for filesystem operations
- Paths read from environment variables with defaults: `Path(os.environ.get("PREDICTIONS_DIR", "/mnt/data/ent_cv/predictions"))`
- `monkeypatch.setattr("api.views.PREDICTIONS_DIR", tmp_path)` pattern used in tests to override module-level path constants

---

*Convention analysis: 2026-03-07*
