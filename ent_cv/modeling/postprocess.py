"""Post-process YOLO temporal detections: filter noise and extract segments.

Three temporal filtering methods are provided so you can compare results on the
same raw detections.json:

  run_length   — drop class-present runs shorter than ``min_duration_sec``;
                 optionally fill short gaps first.  Most interpretable: it does
                 exactly what you'd do by hand.

  majority_vote — symmetric rolling window; keep class at frame t only when it
                  appears in >``vote_threshold`` fraction of the window.
                  Naturally smooths both spikes and gaps.

  gaussian      — Gaussian-weighted version of majority_vote.  Frames near the
                  edge of the window contribute less, giving smoother
                  onset/offset boundaries than a hard rectangular window.

Other post-processing directions worth exploring (not yet implemented):
  - Confidence-gated filtering: require avg confidence > threshold per run.
  - Cross-run merging: join two runs of the same class if the gap between them
    is shorter than min_duration_sec (they were likely one continuous use).
  - Multi-class conflict detection: flag spans where >1 instrument is present
    simultaneously for more than N seconds.
  - Segment timeline: already emitted as segments.json / filtered_segments.json.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml
from loguru import logger

app = typer.Typer(add_completion=False)

METHODS = ("run_length", "majority_vote", "gaussian")


# ── I/O ───────────────────────────────────────────────────────────────────

def load_detections(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Presence matrix ───────────────────────────────────────────────────────

def build_presence_matrix(data: dict) -> np.ndarray:
    """Return bool array of shape (total_frames, num_classes).

    Only frames listed in *results* are marked; any frame absent from the JSON
    (e.g. a gap in sparse output) remains False.
    """
    T = data["total_frames"]
    C = len(data["classes"])
    cls_idx = {name: i for i, name in enumerate(data["classes"])}
    M = np.zeros((T, C), dtype=bool)
    for r in data["results"]:
        frame = r["frame"]
        if frame >= T:
            continue
        for d in r["detections"]:
            ci = cls_idx.get(d["class_name"], -1)
            if ci >= 0:
                M[frame, ci] = True
    return M


# ── Run-length filter ─────────────────────────────────────────────────────

def _find_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive ranges where *arr* is True."""
    runs: list[tuple[int, int]] = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            j = i
            while j < n and arr[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _run_length_filter_1d(arr: np.ndarray, min_len: int, gap_fill: int) -> np.ndarray:
    out = arr.copy()
    # Step 1 — fill short gaps (connect runs that are almost contiguous)
    if gap_fill > 0:
        i, n = 0, len(out)
        while i < n:
            if not out[i]:
                j = i
                while j < n and not out[j]:
                    j += 1
                gap = j - i
                # Only fill if the gap is flanked by True on both sides
                if gap <= gap_fill and i > 0 and j < n:
                    out[i:j] = True
                i = j
            else:
                i += 1
    # Step 2 — remove runs that are still too short
    for start, end in _find_runs(out):
        if end - start + 1 < min_len:
            out[start : end + 1] = False
    return out


def filter_run_length(
    M: np.ndarray,
    min_frames: int,
    gap_fill_frames: int = 0,
) -> np.ndarray:
    """Apply run-length smoothing independently to each class column."""
    out = np.empty_like(M)
    for c in range(M.shape[1]):
        out[:, c] = _run_length_filter_1d(M[:, c], min_frames, gap_fill_frames)
    return out


# ── Majority-vote filter ──────────────────────────────────────────────────

def _rolling_mean_matrix(M: np.ndarray, window: int) -> np.ndarray:
    """Vectorised O(T·C) rolling mean using cumsum with edge-aware counts."""
    T = M.shape[0]
    half = window // 2
    cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M.astype(np.float32), axis=0)])
    lo = np.maximum(0, np.arange(T) - half)
    hi = np.minimum(T, np.arange(T) + half + 1)
    counts = (hi - lo).reshape(-1, 1).astype(np.float32)
    return (cs[hi] - cs[lo]) / counts


def filter_majority_vote(
    M: np.ndarray,
    window: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Symmetric rectangular rolling window majority vote.

    A class is kept at frame t when its fraction of present frames inside
    the window [t-half, t+half] is strictly greater than *threshold*.
    """
    return _rolling_mean_matrix(M, window) > threshold


# ── Gaussian filter ───────────────────────────────────────────────────────

def _gaussian_kernel(window: int) -> np.ndarray:
    """Normalised 1-D Gaussian kernel; sigma = window / 6 (3σ = half-window)."""
    sigma = max(window / 6.0, 1e-6)
    x = np.arange(window) - window // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def filter_gaussian(
    M: np.ndarray,
    window: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Gaussian-weighted majority vote.

    Smoothly penalises frames at the edges of the window; gives softer
    onset/offset boundaries compared to a hard rectangular window.
    """
    kernel = _gaussian_kernel(window)
    T, C = M.shape
    out = np.zeros_like(M)
    for c in range(C):
        smoothed = np.convolve(M[:, c].astype(np.float32), kernel, mode="same")
        out[:, c] = smoothed > threshold
    return out


# ── Apply filter to JSON ──────────────────────────────────────────────────

def apply_filter(data: dict, M_filtered: np.ndarray) -> dict:
    """Return a deep copy of *data* with detections removed where M_filtered is False."""
    cls_idx = {name: i for i, name in enumerate(data["classes"])}
    out = deepcopy(data)
    for r in out["results"]:
        frame = r["frame"]
        if frame >= M_filtered.shape[0]:
            continue
        row = M_filtered[frame]
        r["detections"] = [
            d for d in r["detections"]
            if row[cls_idx.get(d["class_name"], -1)]
        ]
    return out


# ── Segment extraction ────────────────────────────────────────────────────

def extract_segments(data: dict) -> list[dict]:
    """Convert frame-level detections into a structured segment timeline.

    A *segment* is a maximal continuous span of frames where a particular class
    is present.  Segments are annotated with confidence statistics drawn from
    the individual box confidences in each frame.

    Returns a list sorted by start_frame, suitable for downstream analysis such
    as "forceps used from 00:12 to 01:45 (avg conf 0.82)".
    """
    fps = data.get("fps") or data.get("source_fps") or 30.0
    T = data["total_frames"]
    C = len(data["classes"])
    cls_idx = {name: i for i, name in enumerate(data["classes"])}

    M = np.zeros((T, C), dtype=bool)
    # conf_sum / conf_max / conf_count per (frame, class)
    conf_sum = np.zeros((T, C), dtype=np.float64)
    conf_max = np.zeros((T, C), dtype=np.float64)
    conf_cnt = np.zeros((T, C), dtype=np.int32)

    for r in data["results"]:
        frame = r["frame"]
        if frame >= T:
            continue
        for d in r["detections"]:
            ci = cls_idx.get(d["class_name"], -1)
            if ci < 0:
                continue
            M[frame, ci] = True
            conf = d["confidence"]
            conf_sum[frame, ci] += conf
            conf_cnt[frame, ci] += 1
            if conf > conf_max[frame, ci]:
                conf_max[frame, ci] = conf

    segments: list[dict] = []
    for c, cls_name in enumerate(data["classes"]):
        for start, end in _find_runs(M[:, c]):
            # Aggregate confidence over the segment
            total_cnt = int(conf_cnt[start : end + 1, c].sum())
            total_sum = float(conf_sum[start : end + 1, c].sum())
            seg_max = float(conf_max[start : end + 1, c].max())
            avg_conf = total_sum / total_cnt if total_cnt > 0 else 0.0
            segments.append({
                "class_name": cls_name,
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start / fps, 3),
                "end_sec": round(end / fps, 3),
                "duration_sec": round((end - start + 1) / fps, 3),
                "frame_count": end - start + 1,
                "avg_confidence": round(avg_conf, 4),
                "max_confidence": round(seg_max, 4),
            })

    return sorted(segments, key=lambda s: s["start_frame"])


# ── Summary recomputation ─────────────────────────────────────────────────

def compute_summary(data: dict) -> dict:
    fps = data.get("fps") or data.get("source_fps") or 30.0
    T = data["total_frames"]
    sec_per_frame = 1.0 / fps

    class_frame_counts: dict[str, int] = {}
    label_changes = 0
    prev: set[str] = set()

    for r in data["results"]:
        present = {d["class_name"] for d in r["detections"]}
        for cls in present:
            class_frame_counts[cls] = class_frame_counts.get(cls, 0) + 1
        if r["frame"] > 0 and present != prev:
            label_changes += 1
        prev = present

    total_sec = round(T * sec_per_frame, 3)
    return {
        "total_frames": T,
        "label_change_count": label_changes,
        "source_fps": fps,
        "total_case_time_sec": total_sec,
        "class_frame_counts": class_frame_counts,
        "class_frame_percent": {
            cls: round(cnt / T * 100, 2) for cls, cnt in class_frame_counts.items()
        } if T else {},
        "class_time_sec": {
            cls: round(cnt * sec_per_frame, 3) for cls, cnt in class_frame_counts.items()
        },
    }


# ── Main public API ───────────────────────────────────────────────────────

def postprocess(
    raw_json: Path,
    method: str,
    min_duration_sec: float,
    gap_fill_sec: float,
    window_sec: float,
    vote_threshold: float,
    output_dir: Optional[Path] = None,
) -> dict:
    """Filter temporal noise from YOLO detections and write filtered outputs.

    Reads ``raw_json`` (detections.json), applies the chosen temporal filter,
    and writes three files into *output_dir* (default: same directory):

      filtered_detections.json  — full frame-level detections after filtering
      filtered_summary.json     — recomputed summary statistics
      filtered_segments.json    — continuous class-usage segments with confidence

    The raw files are untouched so you can re-run with different parameters and
    compare.  ``segments.json`` is also written from the raw data if it does not
    already exist.

    Args:
        raw_json:         Path to raw ``detections.json``.
        output_dir:       Where to write filtered outputs (default: raw_json parent).
        method:           ``"run_length"`` | ``"majority_vote"`` | ``"gaussian"``.
        min_duration_sec: [run_length] A class must be continuously present for
                          at least this many seconds to survive the filter.
                          Default 3 s (roughly the fastest plausible instrument
                          change in the OR).  Make this smaller if short genuine
                          uses are expected.
        gap_fill_sec:     [run_length] Gaps shorter than this are filled *before*
                          the min_duration check, so two runs separated by a brief
                          dropout are treated as one.  Set to 0 to disable.
        window_sec:       [majority_vote / gaussian] Full window width in seconds.
        vote_threshold:   [majority_vote / gaussian] Fraction of the window that
                          must contain the class for it to be kept (default 0.5).

    Returns:
        dict: Per-class change stats::

            {class_name: {raw_frames, filtered_frames, dropped_frames}}
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}, got {method!r}")

    data = load_detections(raw_json)
    output_dir = Path(output_dir) if output_dir else raw_json.parent
    fps = data.get("fps") or data.get("source_fps") or 30.0

    M = build_presence_matrix(data)

    if method == "run_length":
        min_frames = max(1, round(min_duration_sec * fps))
        gap_frames = max(0, round(gap_fill_sec * fps))
        M_filtered = filter_run_length(M, min_frames, gap_frames)
        params: dict = {
            "min_duration_sec": min_duration_sec,
            "gap_fill_sec": gap_fill_sec,
            "min_frames": min_frames,
            "gap_fill_frames": gap_frames,
        }
    elif method == "majority_vote":
        window_frames = max(1, round(window_sec * fps))
        if window_frames % 2 == 0:
            window_frames += 1  # odd window for symmetry
        M_filtered = filter_majority_vote(M, window_frames, vote_threshold)
        params = {
            "window_sec": window_sec,
            "window_frames": window_frames,
            "vote_threshold": vote_threshold,
        }
    elif method == "gaussian":
        window_frames = max(1, round(window_sec * fps))
        if window_frames % 2 == 0:
            window_frames += 1
        M_filtered = filter_gaussian(M, window_frames, vote_threshold)
        params = {
            "window_sec": window_sec,
            "window_frames": window_frames,
            "vote_threshold": vote_threshold,
        }

    filtered_data = apply_filter(data, M_filtered)
    filtered_data["_filter"] = {"method": method, "source": str(raw_json), **params}

    save_json(filtered_data, output_dir / "filtered_detections.json")
    save_json(compute_summary(filtered_data), output_dir / "filtered_summary.json")
    save_json({"segments": extract_segments(filtered_data)}, output_dir / "filtered_segments.json")

    # Also emit raw segments the first time (cheap, useful reference)
    raw_segs_path = raw_json.parent / "segments.json"
    if not raw_segs_path.exists():
        save_json({"segments": extract_segments(data)}, raw_segs_path)
        logger.info(f"Raw segments written to {raw_segs_path}")

    # Change stats
    raw_counts = M.sum(axis=0).astype(int)
    filt_counts = M_filtered.sum(axis=0).astype(int)
    changes = {
        data["classes"][i]: {
            "raw_frames": int(raw_counts[i]),
            "filtered_frames": int(filt_counts[i]),
            "dropped_frames": int(raw_counts[i] - filt_counts[i]),
        }
        for i in range(len(data["classes"]))
        if raw_counts[i] > 0
    }

    logger.success(f"[{method}] Written to {output_dir}")
    for cls, s in changes.items():
        pct = 100 * s["dropped_frames"] / s["raw_frames"] if s["raw_frames"] > 0 else 0.0
        logger.info(
            f"  {cls:<44} {s['raw_frames']:>6} → {s['filtered_frames']:>6} frames"
            f"  ({pct:.1f}% dropped)"
        )
    return changes


# ── Config dataclass ──────────────────────────────────────────────────────

@dataclass
class PostprocessConfig:
    raw_json: Path
    method: str
    min_duration_sec: float
    gap_fill_sec: float
    window_sec: float
    vote_threshold: float
    output_dir: Optional[Path] = None

    def __post_init__(self):
        self.raw_json = Path(self.raw_json)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


def _load_config(config_file: Path) -> PostprocessConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in PostprocessConfig.__dataclass_fields__}
    if unknown := set(d) - set(PostprocessConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return PostprocessConfig(**known)


# ── CLI ───────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/postprocess.yaml")


@app.command()
def main(
    config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config"),
) -> None:
    """Post-process YOLO temporal detections to remove classification noise.

    Reads a YAML config (default: ent_cv/modeling/configs/postprocess.yaml) and
    writes filtered_detections.json, filtered_summary.json, and
    filtered_segments.json alongside the raw detections.json.  Raw files are
    never modified.
    """
    cfg = _load_config(config_file)
    postprocess(
        raw_json=cfg.raw_json,
        output_dir=cfg.output_dir,
        method=cfg.method,
        min_duration_sec=cfg.min_duration_sec,
        gap_fill_sec=cfg.gap_fill_sec,
        window_sec=cfg.window_sec,
        vote_threshold=cfg.vote_threshold,
    )


if __name__ == "__main__":
    app()
