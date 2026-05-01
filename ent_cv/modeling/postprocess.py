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

from fractions import Fraction
import json
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import polars as pl
import typer

app = typer.Typer(add_completion=False)

METHODS = ("run_length", "majority_vote", "gaussian")


# ── I/O ───────────────────────────────────────────────────────────────────

def load_detections(path: Path) -> pl.DataFrame:
    """Load a Polars-format detections.json."""
    return pl.read_json(path)


def load_metadata(detections_path: Path) -> dict:
    """Load the sidecar metadata.json next to detections.json."""
    meta_path = detections_path.parent / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Sidecar metadata.json not found next to {detections_path}. "
            "Re-run predict to generate it."
        )
    with open(meta_path) as f:
        return json.load(f)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# ── Presence matrix ───────────────────────────────────────────────────────

def build_presence_matrix(
    df: pl.DataFrame, total_frames: int, classes: list[str],
) -> np.ndarray:
    """Return bool array of shape (total_frames, num_classes).

    Only frames present in *df* are marked; any frame absent remains False.
    """
    T = total_frames
    C = len(classes)
    cls_idx = {name: i for i, name in enumerate(classes)}
    M = np.zeros((T, C), dtype=bool)
    for row in df.iter_rows(named=True):
        frame = row["frame"]
        if frame >= T:
            continue
        ci = cls_idx.get(row["name"], -1)
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


# ── Apply filter to DataFrame ─────────────────────────────────────────────

def apply_filter(
    df: pl.DataFrame, M_filtered: np.ndarray, classes: list[str],
) -> pl.DataFrame:
    """Return a filtered copy of *df*, dropping rows where M_filtered is False."""
    cls_idx = {name: i for i, name in enumerate(classes)}

    keep_mask = []
    for row in df.iter_rows(named=True):
        frame = row["frame"]
        ci = cls_idx.get(row["name"], -1)
        if frame >= M_filtered.shape[0] or ci < 0:
            keep_mask.append(False)
        else:
            keep_mask.append(bool(M_filtered[frame, ci]))

    return df.filter(pl.Series(keep_mask))


# ── Bounding-box interpolation ────────────────────────────────────────────

def interpolate_gaps(
    filtered_df: pl.DataFrame,
    M_original: np.ndarray,
    M_filtered: np.ndarray,
    classes: list[str],
) -> pl.DataFrame:
    """Insert linearly interpolated rows for frames filled by gap-fill.

    When ``gap_fill_sec > 0`` the run-length filter bridges short gaps in the
    presence matrix, but those gap frames have no real detection rows —
    ``apply_filter`` only drops rows, it never creates them.  This function
    synthesises one row per (gap_frame, class) by linearly interpolating the
    bounding-box coordinates and confidence between the nearest real detections
    on either side of the gap.  If a gap frame has no real detection on one side
    (edge of clip), the nearest available detection is copied as-is.
    """
    if filtered_df.is_empty():
        return filtered_df

    # Frames that were gap-filled: present in filtered but absent in original
    gap_mask = M_filtered & ~M_original  # (T, C) bool
    if not gap_mask.any():
        return filtered_df

    cls_idx = {name: i for i, name in enumerate(classes)}

    # Per-class lookup: frame -> best (highest-confidence) detection row
    best_by_frame: dict[int, dict[int, dict]] = {c: {} for c in range(len(classes))}
    for row in filtered_df.iter_rows(named=True):
        ci = cls_idx.get(row["name"], -1)
        if ci < 0:
            continue
        f = row["frame"]
        if f not in best_by_frame[ci] or row["confidence"] > best_by_frame[ci][f]["confidence"]:
            best_by_frame[ci][f] = row

    new_rows: list[dict] = []

    for ci, cls_name in enumerate(classes):
        gap_frames_arr = np.where(gap_mask[:, ci])[0]
        if len(gap_frames_arr) == 0:
            continue

        real_frames = sorted(best_by_frame[ci].keys())
        if not real_frames:
            continue

        real_arr = np.array(real_frames)

        for gf in map(int, gap_frames_arr):
            bi = int(np.searchsorted(real_arr, gf, side="left")) - 1  # last frame <= gf
            ai = int(np.searchsorted(real_arr, gf, side="right"))     # first frame > gf

            has_before = bi >= 0
            has_after = ai < len(real_arr)

            if not has_before and not has_after:
                continue

            if not has_before:
                ref = best_by_frame[ci][int(real_arr[ai])]
                new_rows.append({**ref, "frame": gf})
                continue

            if not has_after:
                ref = best_by_frame[ci][int(real_arr[bi])]
                new_rows.append({**ref, "frame": gf})
                continue

            r0 = best_by_frame[ci][int(real_arr[bi])]
            r1 = best_by_frame[ci][int(real_arr[ai])]
            t = (gf - int(real_arr[bi])) / (int(real_arr[ai]) - int(real_arr[bi]))

            b0, b1 = r0["box"], r1["box"]
            new_rows.append({
                "name": cls_name,
                "class": r0["class"],
                "confidence": round(
                    r0["confidence"] + t * (r1["confidence"] - r0["confidence"]), 5
                ),
                "box": {
                    "x1": b0["x1"] + t * (b1["x1"] - b0["x1"]),
                    "y1": b0["y1"] + t * (b1["y1"] - b0["y1"]),
                    "x2": b0["x2"] + t * (b1["x2"] - b0["x2"]),
                    "y2": b0["y2"] + t * (b1["y2"] - b0["y2"]),
                },
                "frame": gf,
            })

    if not new_rows:
        return filtered_df

    new_df = pl.from_dicts(new_rows, schema=filtered_df.schema)
    return pl.concat([filtered_df, new_df]).sort("frame")


# ── Segment extraction ────────────────────────────────────────────────────

def extract_segments(
    df: pl.DataFrame, total_frames: int, classes: list[str], fps: float,
) -> list[dict]:
    """Convert frame-level detections into a structured segment timeline.

    A *segment* is a maximal continuous span of frames where a particular class
    is present.  Segments are annotated with confidence statistics drawn from
    the individual box confidences in each frame.

    Returns a list sorted by start_frame, suitable for downstream analysis such
    as "forceps used from 00:12 to 01:45 (avg conf 0.82)".
    """
    T = total_frames
    C = len(classes)
    cls_idx = {name: i for i, name in enumerate(classes)}

    M = np.zeros((T, C), dtype=bool)
    conf_sum = np.zeros((T, C), dtype=np.float64)
    conf_max = np.zeros((T, C), dtype=np.float64)
    conf_cnt = np.zeros((T, C), dtype=np.int32)

    for row in df.iter_rows(named=True):
        frame = row["frame"]
        if frame >= T:
            continue
        ci = cls_idx.get(row["name"], -1)
        if ci < 0:
            continue
        M[frame, ci] = True
        conf = row["confidence"]
        conf_sum[frame, ci] += conf
        conf_cnt[frame, ci] += 1
        if conf > conf_max[frame, ci]:
            conf_max[frame, ci] = conf

    segments: list[dict] = []
    for c, cls_name in enumerate(classes):
        for start, end in _find_runs(M[:, c]):
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

def compute_summary(
    df: pl.DataFrame, total_frames: int, fps: float,
) -> dict:
    sec_per_frame = 1.0 / fps
    T = total_frames

    class_frame_counts: dict[str, int] = {}
    onset_counts: dict[str, int] = {}
    transition_matrix: dict[str, dict[str, int]] = {}

    # Group detections by frame number
    if df.is_empty():
        frame_groups: dict[int, set[str]] = {}
    else:
        frame_groups = {}
        for row in df.iter_rows(named=True):
            frame = row["frame"]
            frame_groups.setdefault(frame, set()).add(row["name"])

    # Frame-count accumulation (all labeled frames)
    for present in frame_groups.values():
        for cls in present:
            class_frame_counts[cls] = class_frame_counts.get(cls, 0) + 1

    # Onset + transition matrix: iterate only over sorted labeled frames.
    # Unlabeled gaps are discarded — consecutive labeled frames are treated as adjacent
    # for transition purposes regardless of the gap size between them.
    sorted_labeled = sorted(frame_groups.keys())
    class_last_frame: dict[str, int] = {}  # cls -> last frame index seen

    for i, frame_idx in enumerate(sorted_labeled):
        present = frame_groups[frame_idx]

        # Onset: class starts a new run (was absent in the immediately preceding frame).
        # Collect onset classes BEFORE updating class_last_frame so the check is accurate.
        onset_classes: set[str] = set()
        for cls in present:
            if class_last_frame.get(cls, -2) < frame_idx - 1:
                onset_counts[cls] = onset_counts.get(cls, 0) + 1
                onset_classes.add(cls)
            class_last_frame[cls] = frame_idx

        # Transitions: only record when a class starts a new run (onset).
        # One transition per onset, from each class in the preceding labeled frame.
        # This ensures sum(matrix[A][B] for all A) == onset_count[B] (minus the
        # very first labeled frame which has no preceding state).
        # Gaps with no detections are naturally collapsed — frames absent from
        # frame_groups never appear in sorted_labeled, so None is never its own state.
        if i > 0 and onset_classes:
            prev_present = frame_groups[sorted_labeled[i - 1]]
            for to_cls in onset_classes:
                for from_cls in prev_present:
                    row = transition_matrix.setdefault(from_cls, {})
                    row[to_cls] = row.get(to_cls, 0) + 1

    total_sec = round(T * sec_per_frame, 3)
    return {
        "total_frames": T,
        "source_fps": fps,
        "total_case_time_sec": total_sec,
        "class_frame_counts": class_frame_counts,
        "class_frame_percent": {
            cls: round(cnt / T * 100, 2) for cls, cnt in class_frame_counts.items()
        } if T else {},
        "class_time_sec": {
            cls: round(cnt * sec_per_frame, 3) for cls, cnt in class_frame_counts.items()
        },
        "onset_count": onset_counts,
        "transition_matrix": transition_matrix,
    }


# ── Main public API ───────────────────────────────────────────────────────

def postprocess(
    raw_json: Path,
    method: str,
    min_duration_sec: float,
    gap_fill_sec: float,
    window_sec: float,
    vote_threshold: float,
    confidence_threshold: float = 0.0,
    output_dir: Optional[Path] = None,
) -> dict:
    """Filter temporal noise from YOLO detections and write filtered outputs.

    Reads ``raw_json`` (detections.json in Polars JSON format) plus a sidecar
    ``metadata.json`` in the same directory, applies the chosen temporal filter,
    and writes three files into *output_dir* (default: same directory):

      filtered_detections.json  — filtered detections (standard JSON)
      filtered_summary.json     — recomputed summary statistics
      filtered_segments.json    — continuous class-usage segments with confidence

    Args:
        raw_json:         Path to raw ``detections.json``.
        output_dir:       Where to write filtered outputs (default: raw_json parent).
        method:           ``"run_length"`` | ``"majority_vote"`` | ``"gaussian"``.
        min_duration_sec: [run_length] Min duration in seconds.
        gap_fill_sec:     [run_length] Gaps shorter than this are filled first.
        window_sec:       [majority_vote / gaussian] Full window width in seconds.
        vote_threshold:   [majority_vote / gaussian] Fraction threshold (default 0.5).

    Returns:
        dict: Per-class change stats::

            {class_name: {raw_frames, filtered_frames, dropped_frames}}
    """
    raw_json = Path(raw_json)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}, got {method!r}")

    df = load_detections(raw_json)
    if confidence_threshold > 0.0:
        df = df.filter(pl.col("confidence") >= confidence_threshold)
    meta = load_metadata(raw_json)
    output_dir = Path(output_dir) if output_dir else raw_json.parent

    total_frames: int = meta["total_frames"]
    raw_fps = meta.get("fps")
    if raw_fps is None:
        raise ValueError(
            "metadata.json has no 'fps' field — re-run predict on a video source."
        )
    fps: float = float(Fraction(raw_fps))

    # Derive classes from the data
    classes = sorted(df["name"].unique().to_list()) if not df.is_empty() else []

    M = build_presence_matrix(df, total_frames, classes)

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

    filtered_df = apply_filter(df, M_filtered, classes)

    # For run_length with gap-fill, synthesise detections for gap frames
    if method == "run_length" and gap_frames > 0:
        filtered_df = interpolate_gaps(filtered_df, M, M_filtered, classes)

    # Write filtered outputs as standard JSON for human readability
    filter_info = {"method": method, "source": str(raw_json), "confidence_threshold": confidence_threshold, **params}

    # Convert filtered DataFrame to list-of-dicts JSON
    filtered_records = filtered_df.to_dicts() if not filtered_df.is_empty() else []
    save_json(
        {"_filter": filter_info, "detections": filtered_records},
        output_dir / "filtered_detections.json",
    )
    save_json(
        compute_summary(filtered_df, total_frames, fps),
        output_dir / "filtered_summary.json",
    )
    save_json(
        {"segments": extract_segments(filtered_df, total_frames, classes, fps)},
        output_dir / "filtered_segments.json",
    )

    # Also emit raw segments the first time (cheap, useful reference)
    raw_segs_path = raw_json.parent / "segments.json"
    if not raw_segs_path.exists():
        save_json(
            {"segments": extract_segments(df, total_frames, classes, fps)},
            raw_segs_path,
        )
        logger.info(f"Raw segments written to {raw_segs_path}")

    # Change stats
    raw_counts = M.sum(axis=0).astype(int)
    filt_counts = M_filtered.sum(axis=0).astype(int)
    changes = {
        classes[i]: {
            "raw_frames": int(raw_counts[i]),
            "filtered_frames": int(filt_counts[i]),
            "dropped_frames": int(raw_counts[i] - filt_counts[i]),
        }
        for i in range(len(classes))
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


# ── CLI ───────────────────────────────────────────────────────────────────

@app.command()
def main(
    raw_json: Path = typer.Option(..., help="Path to raw detections.json"),
    method: str = typer.Option("run_length", help="Filter method: run_length | majority_vote | gaussian"),
    min_duration_sec: float = typer.Option(1.0, help="[run_length] Min duration in seconds"),
    gap_fill_sec: float = typer.Option(1.0, help="[run_length] Fill gaps shorter than this (sec)"),
    window_sec: float = typer.Option(3.0, help="[majority_vote/gaussian] Window width in seconds"),
    vote_threshold: float = typer.Option(0.5, help="[majority_vote/gaussian] Vote fraction threshold"),
    confidence_threshold: float = typer.Option(0.0, help="Min detection confidence (0 = keep all)"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory (default: same as raw_json)"),
) -> None:
    """Post-process YOLO temporal detections to remove classification noise."""
    postprocess(
        raw_json=raw_json,
        output_dir=output_dir,
        method=method,
        min_duration_sec=min_duration_sec,
        gap_fill_sec=gap_fill_sec,
        window_sec=window_sec,
        vote_threshold=vote_threshold,
        confidence_threshold=confidence_threshold,
    )


if __name__ == "__main__":
    app()
