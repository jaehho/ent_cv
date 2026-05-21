"""Unit tests for min_box_distance and compute_in_use in ent_cv.modeling.postprocess."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ent_cv.modeling.postprocess import compute_in_use, min_box_distance


def _box(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


# ── min_box_distance ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_min_box_distance_overlap_is_zero() -> None:
    a = _box(0, 0, 10, 10)
    b = _box(5, 5, 15, 15)
    assert min_box_distance(a, b) == 0.0


@pytest.mark.unit
def test_min_box_distance_one_inside_other_is_zero() -> None:
    a = _box(0, 0, 100, 100)
    b = _box(40, 40, 60, 60)
    assert min_box_distance(a, b) == 0.0


@pytest.mark.unit
def test_min_box_distance_touching_edge_is_zero() -> None:
    a = _box(0, 0, 10, 10)
    b = _box(10, 0, 20, 10)  # shares x=10 edge
    assert min_box_distance(a, b) == 0.0


@pytest.mark.unit
def test_min_box_distance_horizontal_gap() -> None:
    a = _box(0, 0, 10, 10)
    b = _box(15, 0, 25, 10)  # 5 px gap in x, full overlap in y
    assert min_box_distance(a, b) == pytest.approx(5.0)


@pytest.mark.unit
def test_min_box_distance_diagonal_gap() -> None:
    a = _box(0, 0, 10, 10)
    b = _box(13, 14, 20, 20)  # gap (3, 4) → 5
    assert min_box_distance(a, b) == pytest.approx(5.0)


@pytest.mark.unit
def test_min_box_distance_symmetric() -> None:
    a = _box(0, 0, 10, 10)
    b = _box(50, 50, 60, 60)
    assert min_box_distance(a, b) == min_box_distance(b, a)


# ── compute_in_use ────────────────────────────────────────────────────────

CLASSES = ["Forceps", "Patient", "Empty Hand"]
WIDTH, HEIGHT = 1000, 1000
# frame diagonal = sqrt(2)*1000 ≈ 1414. threshold_frac=0.02 → ~28 px.
THRESH = 0.02


def _row(frame: int, name: str, box: dict, conf: float = 0.9) -> dict:
    return {"frame": frame, "name": name, "box": box, "confidence": conf, "class": 0}


@pytest.mark.unit
def test_compute_in_use_overlap_is_in_use() -> None:
    rows = [
        _row(0, "Patient", _box(0, 0, 500, 500)),
        _row(0, "Forceps", _box(100, 100, 200, 200)),  # fully inside patient
    ]
    df = pl.from_dicts(rows)
    matrix, flags = compute_in_use(df, 1, CLASSES, WIDTH, HEIGHT, THRESH)
    assert matrix[0, 0] is True or bool(matrix[0, 0])
    # The forceps row was the 2nd in the dict order
    forceps_idx = next(i for i, r in enumerate(rows) if r["name"] == "Forceps")
    assert flags[forceps_idx] is True


@pytest.mark.unit
def test_compute_in_use_far_apart_not_in_use() -> None:
    rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(0, "Forceps", _box(800, 800, 900, 900)),  # ~990 px away >> 28
    ]
    df = pl.from_dicts(rows)
    matrix, flags = compute_in_use(df, 1, CLASSES, WIDTH, HEIGHT, THRESH)
    assert not bool(matrix[0, 0])
    forceps_idx = next(i for i, r in enumerate(rows) if r["name"] == "Forceps")
    assert flags[forceps_idx] is False


@pytest.mark.unit
def test_compute_in_use_within_threshold_in_use() -> None:
    rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(0, "Forceps", _box(110, 0, 200, 100)),  # 10 px gap < 28
    ]
    df = pl.from_dicts(rows)
    matrix, _ = compute_in_use(df, 1, CLASSES, WIDTH, HEIGHT, THRESH)
    assert bool(matrix[0, 0])


@pytest.mark.unit
def test_compute_in_use_no_patient_is_not_in_use() -> None:
    rows = [
        _row(0, "Forceps", _box(100, 100, 200, 200)),
    ]
    df = pl.from_dicts(rows)
    matrix, flags = compute_in_use(df, 1, CLASSES, WIDTH, HEIGHT, THRESH)
    assert not bool(matrix[0, 0])
    assert flags[0] is False


@pytest.mark.unit
def test_compute_in_use_non_instrument_rows_get_none_flag() -> None:
    rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(0, "Empty Hand", _box(120, 0, 200, 100)),
    ]
    df = pl.from_dicts(rows)
    _, flags = compute_in_use(df, 1, CLASSES, WIDTH, HEIGHT, THRESH)
    assert flags[0] is None  # Patient
    assert flags[1] is None  # Empty Hand (not an instrument)


@pytest.mark.unit
def test_compute_in_use_carry_forward_within_window() -> None:
    # Patient seen at frame 0, missing at frames 1-2, instrument near at frame 2.
    rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(2, "Forceps", _box(110, 0, 200, 100)),
    ]
    df = pl.from_dicts(rows)
    # carry up to 5 frames → frame 2 still has a valid patient ref
    _, flags = compute_in_use(
        df, total_frames=3, classes=CLASSES, width=WIDTH, height=HEIGHT,
        threshold_frac=THRESH, patient_carry_frames=5,
    )
    forceps_idx = next(i for i, r in enumerate(rows) if r["name"] == "Forceps")
    assert flags[forceps_idx] is True


@pytest.mark.unit
def test_compute_in_use_carry_forward_beyond_window_drops() -> None:
    rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(10, "Forceps", _box(110, 0, 200, 100)),
    ]
    df = pl.from_dicts(rows)
    # carry only 3 frames → frame 10 has no patient ref
    _, flags = compute_in_use(
        df, total_frames=11, classes=CLASSES, width=WIDTH, height=HEIGHT,
        threshold_frac=THRESH, patient_carry_frames=3,
    )
    forceps_idx = next(i for i, r in enumerate(rows) if r["name"] == "Forceps")
    assert flags[forceps_idx] is False


@pytest.mark.unit
def test_compute_in_use_no_carry_forward_default() -> None:
    # With patient_carry_frames=0, instrument in a frame without a same-frame
    # patient cannot be in-use.
    rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(1, "Forceps", _box(110, 0, 200, 100)),
    ]
    df = pl.from_dicts(rows)
    _, flags = compute_in_use(
        df, total_frames=2, classes=CLASSES, width=WIDTH, height=HEIGHT,
        threshold_frac=THRESH, patient_carry_frames=0,
    )
    forceps_idx = next(i for i, r in enumerate(rows) if r["name"] == "Forceps")
    assert flags[forceps_idx] is False


@pytest.mark.unit
def test_compute_in_use_uses_external_patient_df() -> None:
    # Eval df has no patient rows; patient comes from the raw df.
    eval_rows = [_row(0, "Forceps", _box(110, 0, 200, 100))]
    raw_rows = [
        _row(0, "Patient", _box(0, 0, 100, 100)),
        _row(0, "Forceps", _box(110, 0, 200, 100)),
    ]
    eval_df = pl.from_dicts(eval_rows)
    raw_df = pl.from_dicts(raw_rows)
    _, flags = compute_in_use(
        eval_df, total_frames=1, classes=CLASSES, width=WIDTH, height=HEIGHT,
        threshold_frac=THRESH, patient_df=raw_df,
    )
    assert flags[0] is True


@pytest.mark.unit
def test_compute_in_use_threshold_at_frame_diagonal() -> None:
    # Sanity: with threshold = 1.0, everything qualifies regardless of distance.
    rows = [
        _row(0, "Patient", _box(0, 0, 10, 10)),
        _row(0, "Forceps", _box(990, 990, 999, 999)),
    ]
    df = pl.from_dicts(rows)
    _, flags = compute_in_use(
        df, total_frames=1, classes=CLASSES, width=WIDTH, height=HEIGHT,
        threshold_frac=1.0,
    )
    forceps_idx = next(i for i, r in enumerate(rows) if r["name"] == "Forceps")
    assert flags[forceps_idx] is True
    # And that the geometric assumption holds:
    assert math.hypot(WIDTH, HEIGHT) > 1000.0
