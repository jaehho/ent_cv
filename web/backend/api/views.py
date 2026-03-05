import json
import os
import re
import subprocess
from functools import wraps
from pathlib import Path

from django.http import (
    FileResponse,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotFound,
    JsonResponse,
    StreamingHttpResponse,
)
from django.views.decorators.http import require_GET, require_POST


def api_login_required(view_func):
    """Return 401 JSON instead of redirecting to a login page."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper

PREDICTIONS_DIR = Path(os.environ.get("PREDICTIONS_DIR", "/mnt/data/ent_cv/predictions"))
RAW_DIR = Path(os.environ.get("RAW_DIR", "/mnt/data/ent_cv/raw"))

# Validation pattern for case names: alphanumeric, underscores, hyphens only
_CASE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_VALID_METHODS = ("run_length", "majority_vote", "gaussian")

# ── Caches (module-level, persist across requests) ──────────────────────
_fps_cache: dict[str, float] = {}
_frame_count_cache: dict[str, int] = {}

# Standard frame rates for snapping
_STANDARD_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]


def _validate_case_name(name: str) -> bool:
    return bool(name) and _CASE_NAME_RE.match(name) and ".." not in name


def _probe_video_fps(video_path: str) -> float:
    if video_path in _fps_cache:
        return _fps_cache[video_path]
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30,
    )
    raw = result.stdout.strip().rstrip(",")
    if not raw:
        raise RuntimeError(f"ffprobe returned no fps for {video_path}")
    parts = raw.split("/")
    num, den = float(parts[0]), float(parts[1]) if len(parts) > 1 else 1.0
    fps = num / den if den else num
    # Snap to nearest standard framerate
    fps = min(_STANDARD_FPS, key=lambda s: abs(s - fps))
    _fps_cache[video_path] = fps
    return fps


def _probe_video_frame_count(video_path: str) -> int:
    if video_path in _frame_count_cache:
        return _frame_count_cache[video_path]
    # Fast path: nb_frames from container index
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30,
    )
    raw = result.stdout.strip().rstrip(",")
    if raw and raw != "N/A" and raw.isdigit():
        count = int(raw)
        _frame_count_cache[video_path] = count
        return count
    # Slow fallback: count every packet
    result2 = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=120,
    )
    raw2 = result2.stdout.strip().rstrip(",")
    if not raw2:
        raise RuntimeError(f"ffprobe returned no frame count for {video_path}")
    count = int(raw2)
    _frame_count_cache[video_path] = count
    return count


# ── Views ───────────────────────────────────────────────────────────────


@require_GET
@api_login_required
def list_cases(request):
    """List case directories that contain detections.json, newest first."""
    try:
        cases = sorted(
            [
                d.name for d in PREDICTIONS_DIR.iterdir()
                if d.is_dir() and (d / "detections.json").exists()
            ],
            reverse=True,
        )
        return JsonResponse(cases, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
@api_login_required
def detections(request, case):
    """Serve enriched detections.json (with fps, class remapping, part boundaries).
    
    Query params:
        mode=filtered  → read filtered_detections.json instead of detections.json
    """
    if not _validate_case_name(case):
        return HttpResponseBadRequest("Invalid case name")

    mode = request.GET.get("mode", "raw")
    if mode == "filtered":
        det_path = PREDICTIONS_DIR / case / "filtered_detections.json"
    else:
        det_path = PREDICTIONS_DIR / case / "detections.json"
    if not det_path.exists():
        return HttpResponseNotFound(f"{det_path.name} not found")

    raw_case_dir = RAW_DIR / case
    try:
        mp4s = sorted(f.name for f in raw_case_dir.iterdir() if f.suffix.lower() == ".mp4")
    except FileNotFoundError:
        return HttpResponseNotFound(f"No raw directory for case {case}")
    if not mp4s:
        return JsonResponse({"error": f"No raw .mp4 files for case {case}"}, status=500)

    # Read metadata.json if available
    meta_path = PREDICTIONS_DIR / case / "metadata.json"
    meta_raw = {}
    fps = None
    total_frames = None
    cached_part_frames = None
    try:
        with open(meta_path) as f:
            meta_raw = json.load(f)
        fps_val = meta_raw.get("fps")
        if fps_val is not None:
            fps_str = str(fps_val)
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) else None
            else:
                fps = float(fps_str)
            if fps is None or not (fps > 0):
                fps = None
        total_frames = meta_raw.get("total_frames")
        cached_part_frames = meta_raw.get("part_frames")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if fps is None:
        fps = _probe_video_fps(str(raw_case_dir / mp4s[0]))

    # Load detections
    try:
        with open(det_path) as f:
            raw_detections = json.load(f)
    except json.JSONDecodeError:
        return JsonResponse(
            {"error": "Detections file is corrupted — re-run postprocessing."},
            status=503,
        )

    # Handle filtered_detections.json wrapper
    filter_metadata = None
    if not isinstance(raw_detections, list) and isinstance(raw_detections.get("detections"), list):
        filter_metadata = raw_detections.get("_filter")
        raw_detections = raw_detections["detections"]

    # Detect format: flat vs grouped
    is_flat = len(raw_detections) > 0 and not isinstance(raw_detections[0].get("detections"), list)

    # Derive ordered class list
    class_map = {}  # class_id -> class_name
    if is_flat:
        for d in raw_detections:
            cid = d.get("class")
            if cid not in class_map:
                class_map[cid] = d.get("name", "")
    else:
        for r in raw_detections:
            for d in r.get("detections", []):
                cid = d.get("class_id")
                if cid not in class_map:
                    class_map[cid] = d.get("class_name", "")

    # For filtered data: supplement class_map from raw detections
    if filter_metadata is not None:
        raw_det_path = PREDICTIONS_DIR / case / "detections.json"
        try:
            with open(raw_det_path) as f:
                raw_arr = json.load(f)
            arr = raw_arr if isinstance(raw_arr, list) else raw_arr.get("detections", [])
            for d in arr:
                cid = d.get("class")
                if cid not in class_map:
                    class_map[cid] = d.get("name", "")
        except Exception:
            pass

    sorted_class_entries = sorted(class_map.items(), key=lambda x: x[0])
    classes = [name for _, name in sorted_class_entries]
    class_index_remap = {orig_id: idx for idx, (orig_id, _) in enumerate(sorted_class_entries)}

    part_boundaries = None
    results = None

    if is_flat:
        # Build per-part cumulative frame-count boundaries
        part_boundaries = []
        cumulative = 0
        wrote_new_counts = False
        new_part_frames = dict(cached_part_frames) if cached_part_frames else {}

        for mp4 in mp4s:
            mp4_path = str(raw_case_dir / mp4)
            if cached_part_frames and mp4 in cached_part_frames:
                count = cached_part_frames[mp4]
                _frame_count_cache[mp4_path] = count
            else:
                count = _probe_video_frame_count(mp4_path)
                new_part_frames[mp4] = count
                wrote_new_counts = True
            part_boundaries.append({
                "path": mp4_path,
                "startFrame": cumulative,
                "endFrame": cumulative + count - 1,
            })
            cumulative += count

        # Persist newly probed counts
        if wrote_new_counts:
            try:
                meta_raw["part_frames"] = new_part_frames
                with open(meta_path, "w") as f:
                    json.dump(meta_raw, f, indent=2)
            except Exception:
                pass

        def source_for_frame(frame):
            for p in part_boundaries:
                if frame <= p["endFrame"]:
                    return p["path"]
            return part_boundaries[-1]["path"]

        # Group flat detections by frame
        frame_groups: dict[int, list] = {}
        for d in raw_detections:
            fr = d.get("frame")
            if fr not in frame_groups:
                frame_groups[fr] = []
            b = d.get("box")
            frame_groups[fr].append({
                "class_id": class_index_remap.get(d.get("class"), d.get("class")),
                "class_name": d.get("name", ""),
                "confidence": d.get("confidence"),
                "bbox": [b["x1"], b["y1"], b["x2"], b["y2"]] if b else None,
            })

        results = [
            {
                "frame": fr,
                "source": source_for_frame(fr),
                "detections": dets,
            }
            for fr, dets in sorted(frame_groups.items())
        ]
    else:
        results = raw_detections

    if not total_frames:
        total_frames = len(results)

    parts = (
        [{"source": b["path"], "startFrame": b["startFrame"], "endFrame": b["endFrame"]}
         for b in part_boundaries]
        if is_flat and part_boundaries else None
    )

    enriched = {
        "fps": fps,
        "total_frames": total_frames,
        "classes": classes,
        "results": results,
    }
    if parts:
        enriched["parts"] = parts
    if filter_metadata:
        enriched["_filter"] = filter_metadata

    return JsonResponse(enriched, safe=False)


@require_GET
@api_login_required
def predictions_file(request, case, path):
    """Serve JPEG frames or other prediction files."""
    if not _validate_case_name(case):
        return HttpResponseBadRequest("Invalid case name")
    # Sanitize path
    clean_parts = [p for p in path.split("/") if p and p != ".."]
    file_path = PREDICTIONS_DIR / case / "/".join(clean_parts)
    if not file_path.exists() or not file_path.is_file():
        return HttpResponseNotFound("File not found")
    # Verify it's under PREDICTIONS_DIR (prevent traversal)
    try:
        file_path.resolve().relative_to(PREDICTIONS_DIR.resolve())
    except ValueError:
        return HttpResponseBadRequest("Invalid path")

    ext = file_path.suffix.lower()
    content_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".json": "application/json",
    }
    content_type = content_types.get(ext, "application/octet-stream")
    return FileResponse(open(file_path, "rb"), content_type=content_type)


@require_GET
@api_login_required
def raw_video(request, case, filename):
    """Serve raw video files with HTTP Range support."""
    if not _validate_case_name(case):
        return HttpResponseBadRequest("Invalid case name")
    # Validate filename
    if ".." in filename or "/" in filename:
        return HttpResponseBadRequest("Invalid filename")

    file_path = RAW_DIR / case / filename
    if not file_path.exists() or not file_path.is_file():
        return HttpResponseNotFound("File not found")
    try:
        file_path.resolve().relative_to(RAW_DIR.resolve())
    except ValueError:
        return HttpResponseBadRequest("Invalid path")

    file_size = file_path.stat().st_size
    range_header = request.META.get("HTTP_RANGE", "")

    if range_header:
        # Parse Range: bytes=start-end
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not match:
            return HttpResponse(status=416)  # Range Not Satisfiable
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
        if start >= file_size:
            return HttpResponse(status=416)
        end = min(end, file_size - 1)
        length = end - start + 1

        f = open(file_path, "rb")
        f.seek(start)

        def read_chunk():
            remaining = length
            while remaining > 0:
                chunk_size = min(8192, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                remaining -= len(data)
                yield data
            f.close()

        response = StreamingHttpResponse(read_chunk(), content_type="video/mp4", status=206)
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Accept-Ranges"] = "bytes"
        response["Content-Length"] = length
        return response
    else:
        response = FileResponse(open(file_path, "rb"), content_type="video/mp4")
        response["Content-Length"] = file_size
        response["Accept-Ranges"] = "bytes"
        return response


@require_POST
@api_login_required
def postprocess_case(request, case):
    """Run postprocessing on a case's detections."""
    if not _validate_case_name(case):
        return HttpResponseBadRequest("Invalid case name")

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    method = body.get("method", "run_length")
    if method not in _VALID_METHODS:
        return JsonResponse({"ok": False, "error": "Invalid method"}, status=400)

    raw_json = PREDICTIONS_DIR / case / "detections.json"
    if not raw_json.exists():
        return JsonResponse({"ok": False, "error": "detections.json not found"}, status=404)

    try:
        # Import postprocess function directly
        from ent_cv.modeling.postprocess import postprocess

        postprocess(
            raw_json=raw_json,
            method=method,
            min_duration_sec=float(body.get("min_duration_sec", 1.0)),
            gap_fill_sec=float(body.get("gap_fill_sec", 1.0)),
            window_sec=float(body.get("window_sec", 3.0)),
            vote_threshold=float(body.get("vote_threshold", 0.5)),
            confidence_threshold=float(body.get("confidence_threshold", 0.0)),
        )
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)[:2000]}, status=500)

    # Read the freshly-written filtered_summary.json
    summary = None
    summary_path = PREDICTIONS_DIR / case / "filtered_summary.json"
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except Exception:
        pass

    return JsonResponse({"ok": True, "summary": summary})


@require_GET
@api_login_required
def filtered_summary(request, case):
    """Serve filtered_summary.json for a case."""
    if not _validate_case_name(case):
        return HttpResponseBadRequest("Invalid case name")

    summary_path = PREDICTIONS_DIR / case / "filtered_summary.json"
    if not summary_path.exists():
        return HttpResponseNotFound("No filtered summary found")

    with open(summary_path) as f:
        data = json.load(f)
    return JsonResponse(data, safe=False)
