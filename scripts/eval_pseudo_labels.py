"""Compute precision/recall of patient_v0 pseudo-labels vs. a hand-labeled test set.

Inputs:
  - A CVAT task ID (typically the output of create_patient_test_set.py) whose
    annotations are hand-labeled Patient boxes on frames the bootstrap detector
    has never seen.
  - On-disk pseudo-labels under predictions/patient_v0_pseudo/labels/.

For each test frame the script matches pseudo-label boxes to GT boxes via
greedy IoU; reports precision, recall, and F1 at IoU∈{0.5, 0.75, 0.9}, both
overall and per-case. The 0.5 number tells you "does the box hit the right
region"; 0.75/0.9 stress test geometry.

Why this matters: patient_v0's reported mAP50≈0.995 is on 30 frames from one
case. That's not a credible quality bar for merging 1,796 pseudo-labels into
the training corpus. This script gives you a real number.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import tempfile
import zipfile

from cvat_sdk import make_client
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

PSEUDO_LABELS_DIR = DATA_DIR / "predictions" / "patient_v0_pseudo" / "labels"
PATIENT_CLASS = LABELS.index("Patient")
IOU_THRESHOLDS = (0.5, 0.75, 0.9)
CASE_RE = re.compile(r"^(.+?)_Part\d+_f\d+$")


def _case_of(stem: str) -> str:
    m = CASE_RE.match(stem)
    return m.group(1) if m else stem


def _read_yolo_boxes_text(
    text: str, only_class: int
) -> list[tuple[float, float, float, float]]:
    """Parse YOLO label text. Returns (cx, cy, w, h) tuples matching ``only_class``."""
    out: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        if int(parts[0]) != only_class:
            continue
        cx, cy, w, h = (float(x) for x in parts[1:])
        out.append((cx, cy, w, h))
    return out


def _read_yolo_boxes(path: Path, only_class: int) -> list[tuple[float, float, float, float]]:
    """Return list of (cx, cy, w, h) for boxes matching ``only_class``."""
    if not path.exists():
        return []
    return _read_yolo_boxes_text(path.read_text(), only_class)


def _gt_patient_class(export_dir: Path) -> int:
    """Find Patient's class index in the CVAT YOLO export's data.yaml.

    CVAT indexes YOLO classes by *project label order*, which is not guaranteed
    to match ``LABELS`` in ent_cv.config. Read it from the export rather than
    assuming.
    """
    data_yaml = export_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"data.yaml not found in CVAT export: {data_yaml}")
    spec = yaml.safe_load(data_yaml.read_text())
    names = spec.get("names")
    if isinstance(names, dict):
        for idx, name in names.items():
            if name == "Patient":
                return int(idx)
    elif isinstance(names, list):
        for idx, name in enumerate(names):
            if name == "Patient":
                return idx
    raise RuntimeError(f"'Patient' not found in data.yaml names: {names}")


def _to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _match(
    preds: list[tuple[float, float, float, float]],
    gts: list[tuple[float, float, float, float]],
    iou_thr: float,
) -> tuple[int, int, int]:
    """Greedy IoU matching. Returns (TP, FP, FN)."""
    if not gts and not preds:
        return (0, 0, 0)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    # Sort prediction-GT pairs by IoU desc, accept until threshold violated.
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = _iou(p, g)
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    for _iou_v, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return (tp, fp, fn)


def _flatten_labels(export_dir: Path) -> dict[str, Path]:
    """CVAT preserves SHARE-relative paths inside the zip; flatten by stem."""
    out: dict[str, Path] = {}
    for p in (export_dir / "labels").rglob("*.txt"):
        out[p.stem] = p
    return out


def _fmt_pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) and not (p != p or r != r) else float("nan")
    return p, r, f1


@app.command()
def main(
    task_id: int = typer.Argument(..., help="CVAT task ID with hand-labeled Patient boxes."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    pseudo_dir: Path = typer.Option(PSEUDO_LABELS_DIR, help="Path to pseudo-label .txt files."),
):
    """Compute pseudo-label P/R against the hand-labeled CVAT task."""
    # Read GT into memory *inside* the with block — the temp dir is gone the
    # moment we exit, and lazy file reads later would silently return [].
    gt_boxes_by_stem: dict[str, list[tuple[float, float, float, float]]] = {}
    with tempfile.TemporaryDirectory(prefix=f"eval_{task_id}_") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "test.zip"
        with make_client(
            host=host, port=port, credentials=(username, password)
        ) as client:
            task = client.tasks.retrieve(task_id)
            logger.info(f"Task {task_id}: '{task.name}' (project {task.project_id})")
            task.export_dataset(
                format_name="Ultralytics YOLO Detection 1.0",
                filename=str(zip_path),
                include_images=False,
            )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        gt_class = _gt_patient_class(tmp)
        logger.info(f"  CVAT exports Patient as class {gt_class}")
        for stem, path in _flatten_labels(tmp).items():
            gt_boxes_by_stem[stem] = _read_yolo_boxes_text(path.read_text(), gt_class)

    if not gt_boxes_by_stem:
        raise typer.Exit("No GT label files in the export — did you label any frames?")

    # Per-frame matching at each IoU threshold.
    overall: dict[float, dict[str, int]] = {
        thr: {"tp": 0, "fp": 0, "fn": 0} for thr in IOU_THRESHOLDS
    }
    by_case: dict[str, dict[float, dict[str, int]]] = defaultdict(
        lambda: {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in IOU_THRESHOLDS}
    )

    n_frames_labeled = 0
    n_frames_with_gt_boxes = 0
    n_frames_with_pseudo_only = 0
    n_frames_empty_both = 0

    for stem, gts in gt_boxes_by_stem.items():
        case = _case_of(stem)
        preds = _read_yolo_boxes(pseudo_dir / f"{stem}.txt", PATIENT_CLASS)

        n_frames_labeled += 1
        if gts:
            n_frames_with_gt_boxes += 1
        if not gts and preds:
            n_frames_with_pseudo_only += 1
        if not gts and not preds:
            n_frames_empty_both += 1

        for thr in IOU_THRESHOLDS:
            tp, fp, fn = _match(preds, gts, thr)
            overall[thr]["tp"] += tp
            overall[thr]["fp"] += fp
            overall[thr]["fn"] += fn
            by_case[case][thr]["tp"] += tp
            by_case[case][thr]["fp"] += fp
            by_case[case][thr]["fn"] += fn

    typer.echo()
    typer.echo(f"Test set: {n_frames_labeled} frames, {len(by_case)} cases")
    typer.echo(f"  frames with ≥1 GT Patient box: {n_frames_with_gt_boxes}")
    typer.echo(
        f"  frames with no GT but pseudo box: {n_frames_with_pseudo_only} (potential false positives)"
    )
    typer.echo(f"  frames with no boxes either side: {n_frames_empty_both}")

    typer.echo()
    typer.echo("=" * 78)
    typer.echo(f"{'Scope':<22}{'IoU':>5}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'P':>6}  {'R':>6}  {'F1':>6}")
    typer.echo("=" * 78)
    for case in sorted(by_case):
        for thr in IOU_THRESHOLDS:
            s = by_case[case][thr]
            p, r, f1 = _fmt_pr(s["tp"], s["fp"], s["fn"])
            typer.echo(
                f"{case:<22}{thr:>5.2f}  {s['tp']:>5}  {s['fp']:>5}  {s['fn']:>5}  "
                f"{p:>6.3f}  {r:>6.3f}  {f1:>6.3f}"
            )
        typer.echo("-" * 78)
    for thr in IOU_THRESHOLDS:
        s = overall[thr]
        p, r, f1 = _fmt_pr(s["tp"], s["fp"], s["fn"])
        typer.echo(
            f"{'OVERALL':<22}{thr:>5.2f}  {s['tp']:>5}  {s['fp']:>5}  {s['fn']:>5}  "
            f"{p:>6.3f}  {r:>6.3f}  {f1:>6.3f}"
        )
    typer.echo("=" * 78)

    typer.echo()
    typer.echo("Interpretation:")
    typer.echo("  IoU=0.50 captures 'is the box in roughly the right place?'")
    typer.echo("  IoU=0.75 stress-tests geometry")
    typer.echo("  IoU=0.90 is very strict — drops a lot with fuzzy class definitions")
    typer.echo()
    typer.echo(
        "Decision heuristic: if IoU=0.50 P>0.90 and R>0.85 across cases, "
        "the pseudo-labels are mergeable as-is. Lower → investigate failures."
    )


if __name__ == "__main__":
    app()
