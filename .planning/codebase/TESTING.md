# Testing Patterns

**Analysis Date:** 2026-03-07

## Test Framework

**Runner (Python):**
- pytest with pytest-django and pytest-cov
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
- `DJANGO_SETTINGS_MODULE = "config.settings"`
- `pythonpath = ["web/backend"]`
- `testpaths = ["web/backend"]`

**Runner (JavaScript):**
- Vitest (configured inline in `web/frontend/vite.config.js`)
- Environment: `jsdom`
- Globals: `true` (no need to import `describe`/`it`/`expect` in test files, though they are imported explicitly in this project)

**Assertion Library:**
- Python: pytest built-in `assert`
- JavaScript: Vitest built-in `expect`

**Run Commands:**
```bash
make test                                                  # Run all tests (pytest + vitest)
uv run pytest web/backend/ -k test_name                   # Run a single Python test
cd web/frontend && npx vitest run --reporter=verbose       # Run frontend tests verbose
uv run pytest web/backend/                                 # Run all Python tests
```

## Test File Organization

**Location (Python):**
- Separate `tests/` subdirectory inside each Django app: `web/backend/api/tests/`
- Contains `__init__.py`, `conftest.py`, and `test_*.py` files

**Location (JavaScript):**
- Separate `src/__tests__/` directory: `web/frontend/src/__tests__/`
- Not co-located with source files

**Naming (Python):** `test_<subject>.py` (e.g., `test_auth.py`, `test_cases_api.py`)

**Naming (JavaScript):** `<subject>.spec.js` (e.g., `App.spec.js`, `utils.spec.js`)

**Structure:**
```
web/backend/api/tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_auth.py         # Auth endpoint tests
└── test_cases_api.py    # Cases API tests

web/frontend/src/__tests__/
├── App.spec.js          # App component test
└── utils.spec.js        # Utility function tests
```

## Test Structure

**Python suite organization (class-based grouping by endpoint):**
```python
@pytest.mark.django_db
class TestListCases:
    def test_unauthenticated_returns_401(self, anon_client):
        resp = anon_client.get("/api/cases/")
        assert resp.status_code == 401

    def test_returns_cases_with_detections(self, auth_client, predictions_dir):
        resp = auth_client.get("/api/cases/")
        assert resp.status_code == 200
        data = resp.json()
        assert "test_case" in data
```

**JavaScript suite organization:**
```javascript
import { describe, it, expect } from "vitest";
import { formatTime } from "../utils/index.js";

describe("formatTime", () => {
  it("formats 0 seconds", () => {
    expect(formatTime(0)).toBe("0:00.00");
  });
});
```

**Patterns:**
- All Django DB test classes decorated with `@pytest.mark.django_db`
- Tests grouped by class per resource/endpoint (e.g., `TestListCases`, `TestDetections`, `TestRawVideo`)
- Test method names follow `test_<scenario>` pattern (e.g., `test_unauthenticated_returns_401`, `test_invalid_case_name`)
- Each test is focused on a single behavior

## Mocking

**Framework:** pytest `monkeypatch` (built-in)

**Pattern for overriding module-level path constants:**
```python
@pytest.fixture
def predictions_dir(tmp_path, monkeypatch):
    """Create a temporary predictions directory with sample data."""
    case_dir = tmp_path / "test_case"
    case_dir.mkdir()
    # ... create fixture files ...
    monkeypatch.setattr("api.views.PREDICTIONS_DIR", tmp_path)
    return tmp_path
```

**Vue component mocking (stub pattern):**
```javascript
const wrapper = mount(App, {
  global: {
    stubs: {
      LoginForm: { template: '<div class="login-stub" />' },
      "router-view": { template: '<div class="router-stub" />' },
    },
  },
});
```

**What to Mock:**
- Filesystem paths (`PREDICTIONS_DIR`, `RAW_DIR`) via `monkeypatch.setattr` pointing to `tmp_path`
- Child Vue components via `stubs` when testing parent component behavior in isolation

**What NOT to Mock:**
- Django `Client` (use real test client from `django.test`)
- HTTP request/response cycle (test through the real view layer)

## Fixtures and Factories

**Python fixtures in `web/backend/api/tests/conftest.py`:**

```python
@pytest.fixture
def user(db):
    return User.objects.create_user(username="testuser", password="testpass123")

@pytest.fixture
def auth_client(user):
    client = Client()
    client.login(username="testuser", password="testpass123")
    return client

@pytest.fixture
def anon_client():
    return Client()

@pytest.fixture
def predictions_dir(tmp_path, monkeypatch):
    """Create a temporary predictions directory with sample data."""
    case_dir = tmp_path / "test_case"
    case_dir.mkdir()
    detections = [{"frame": 0, "class": 0, "name": "Forceps", "confidence": 0.95, ...}]
    (case_dir / "detections.json").write_text(json.dumps(detections))
    (case_dir / "metadata.json").write_text(json.dumps({"fps": 30, "total_frames": 2}))
    monkeypatch.setattr("api.views.PREDICTIONS_DIR", tmp_path)
    return tmp_path

@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Create a temporary raw directory with a dummy video file."""
    case_dir = tmp_path / "test_case"
    case_dir.mkdir(exist_ok=True)
    dummy_video = case_dir / "part1.mp4"
    dummy_video.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("api.views.RAW_DIR", tmp_path)
    return tmp_path
```

**Location:** `web/backend/api/tests/conftest.py`

**Test data approach:** Created inline in fixtures using `tmp_path`; JSON written to temp files directly.

## Coverage

**Requirements:** No enforced coverage threshold detected.

**View Coverage:**
```bash
uv run pytest web/backend/ --cov=web/backend
```

## Test Types

**Integration Tests (Python):**
- All Python tests are Django integration tests hitting real HTTP endpoints via `django.test.Client`
- Scope: full request/response cycle including middleware, auth, view logic, and filesystem reads
- No unit tests for individual view helper functions

**Unit Tests (JavaScript):**
- `utils.spec.js` — Pure function unit tests for `formatTime` and `formatRealTime`
- `App.spec.js` — Shallow component mount test verifying conditional rendering

**E2E Tests:** Not present.

## Common Patterns

**Auth boundary testing (every endpoint class includes this):**
```python
def test_unauthenticated_returns_401(self, anon_client):
    resp = anon_client.get("/api/cases/")
    assert resp.status_code == 401
```

**Input validation testing:**
```python
def test_invalid_case_name(self, auth_client):
    resp = auth_client.get("/api/cases/bad.name/detections/")
    assert resp.status_code == 400

def test_path_traversal_blocked(self, auth_client, raw_dir):
    resp = auth_client.get("/api/cases/bad..name/raw/part1.mp4")
    assert resp.status_code == 400
```

**JSON body posting:**
```python
resp = auth_client.post(
    "/api/cases/test_case/postprocess/",
    data=json.dumps({"method": "run_length"}),
    content_type="application/json",
)
```

**File content assertion:**
```python
summary = {"instruments": ["Forceps"], "total_time": 10.0}
(predictions_dir / "test_case" / "filtered_summary.json").write_text(json.dumps(summary))
resp = auth_client.get("/api/cases/test_case/filtered-summary/")
assert resp.json()["instruments"] == ["Forceps"]
```

**Range request testing:**
```python
resp = auth_client.get(
    "/api/cases/test_case/raw/part1.mp4",
    HTTP_RANGE="bytes=0-99",
)
assert resp.status_code == 206
assert "Content-Range" in resp
```

---

*Testing analysis: 2026-03-07*
