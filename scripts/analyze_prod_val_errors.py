"""Analyze prod model errors on the val set.

Runs the prod model on every val frame at a low conf threshold (so we see what
the model considered, not just what survived deployment NMS), matches each
prediction to the GT box with highest IoU (greedy, IoU >= 0.5), and tallies:

  - True positive       (matched, same class)
  - Class confusion     (matched, wrong class)
  - False positive      (pred with no IoU>=0.5 GT match)
  - False negative      (GT with no IoU>=0.5 pred match)

Outputs (written under repo's gitignored reports/ dir per project layout):
  reports/prod_val_errors/
    per_class_metrics.md     P/R/F1 at multiple conf thresholds, per class
    confusion_matrix.csv     Raw counts: rows=GT class (incl 'background' row
                             for FPs), cols=pred class (incl 'background' col
                             for FNs)
    worst_frames.md          Top 25 frames by total error count
    summary.md               Headline numbers + the patterns I'd act on
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import LABELS
from ent_cv.gpu import gpu_yield

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path("/mnt/data/ent_cv/models/best/weights/best.pt")
DEFAULT_DATASET = Path("/mnt/data/ent_cv/datasets/current_full_v2")
DEFAULT_OUT = REPO_ROOT / "reports" / "prod_val_errors"
LOW_CONF = 0.05  # capture more candidates than deployment uses
IOU_MATCH_THRESHOLD = 0.5
DEPLOY_CONF_THRESHOLDS = (0.25, 0.5, 0.647)  # 0.647 is the CLI default
TOP_WORST_FRAMES = 25
BACKGROUND_LABEL = "<background>"


def _xywhn_to_xyxy(box: tuple[float, float, float, float], w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = box
    return np.array([
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, np.ndarray]] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        box = tuple(float(p) for p in parts[1:])  # type: ignore[assignment]
        out.append((cls, _xywhn_to_xyxy(box, w, h)))  # type: ignore[arg-type]
    return out


def _greedy_match(
    preds: list[tuple[int, float, np.ndarray]],  # (cls, conf, xyxy)
    gts: list[tuple[int, np.ndarray]],            # (cls, xyxy)
) -> tuple[list[tuple[int, int, int, float, float]], list[int], list[int]]:
    """Return (matches, unmatched_pred_idxs, unmatched_gt_idxs).

    matches: list of (pred_idx, gt_idx, gt_cls, pred_cls/conf via lookup, iou)
              -- actually we return (pred_idx, gt_idx, gt_cls, pred_cls, iou)
              with conf available by indexing preds. We pack conf in the tuple
              for convenience.
    Match is greedy by descending IoU subject to IoU >= IOU_MATCH_THRESHOLD,
    regardless of class. Class equality is reported in the result for the
    caller to tally TPs vs class-confusion separately.
    """
    pred_used = [False] * len(preds)
    gt_used = [False] * len(gts)
    pairs: list[tuple[int, int, float]] = []
    for p_idx, (p_cls, p_conf, p_box) in enumerate(preds):
        for g_idx, (g_cls, g_box) in enumerate(gts):
            iou = _iou(p_box, g_box)
            if iou >= IOU_MATCH_THRESHOLD:
                pairs.append((p_idx, g_idx, iou))
    pairs.sort(key=lambda t: -t[2])
    matches: list[tuple[int, int, int, float, float]] = []
    for p_idx, g_idx, iou in pairs:
        if pred_used[p_idx] or gt_used[g_idx]:
            continue
        pred_used[p_idx] = True
        gt_used[g_idx] = True
        matches.append((p_idx, g_idx, gts[g_idx][0], iou, preds[p_idx][1]))
    unmatched_pred = [i for i, u in enumerate(pred_used) if not u]
    unmatched_gt = [i for i, u in enumerate(gt_used) if not u]
    return matches, unmatched_pred, unmatched_gt


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(s) for s in col) for col in cols]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    out = [fmt.format(*headers), sep]
    out += [fmt.format(*row) for row in rows]
    return "\n".join(out)


def main(
    weights: Path,
    dataset: Path,
    out_dir: Path,
    low_conf: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    val_split = dataset / "val_split.txt"
    labels_dir = dataset / "labels" / "train"  # single-dir layout
    if not val_split.exists():
        raise FileNotFoundError(val_split)
    images = [Path(ln.strip()) for ln in val_split.read_text().splitlines() if ln.strip()]
    logger.info(f"Loaded {len(images)} val images")

    logger.info(f"Loading weights: {weights}")
    model = cast("YOLO", YOLO(str(weights)))

    # Pred records: per-frame list of (cls, conf)
    # Match records (matched preds + unmatched preds + unmatched gts) for analysis
    # Confusion: rows = GT class index (or BACKGROUND_LABEL for unmatched preds),
    #            cols = pred class index (or BACKGROUND_LABEL for unmatched gts)
    n_classes = len(LABELS)
    conf_mat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # For threshold-swept P/R/F1, accumulate per class:
    #   tp_by_conf[cls][thr] = TPs at conf >= thr
    #   fp_by_conf[cls][thr] = FPs at conf >= thr
    #   fn_by_conf[cls][thr] = GT count (constant for cls); FNs = gt_count - TPs at thr
    gt_counts: Counter[int] = Counter()
    per_frame_rows: list[dict] = []

    # Per-class predictions list: (conf, is_tp, match_gt_class) — we'll sort and
    # sweep for precision/recall at each deploy threshold.
    pred_records: list[dict] = []

    logger.info(f"Running inference at conf={low_conf}, iou=0.7 on {len(images)} frames")
    frame_state: list[tuple[Path, int, int, list[tuple[int, float, np.ndarray]]]] = []
    with gpu_yield("0"):
        for idx, img_path in enumerate(images):
            if idx % 100 == 0:
                logger.info(f"  inference: {idx}/{len(images)}")
            result = model.predict(
                source=str(img_path),
                conf=low_conf,
                iou=0.7,
                imgsz=1024,
                device="0",
                verbose=False,
            )[0]
            h, w = result.orig_shape
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy().astype(int)
                preds = [(int(c), float(cf), boxes_xyxy[i]) for i, (c, cf) in enumerate(zip(clses, confs))]
            else:
                preds = []
            frame_state.append((img_path, h, w, preds))

    for img_path, h, w, preds in frame_state:
        label_path = labels_dir / f"{img_path.stem}.txt"
        gts = _load_gt(label_path, w, h)
        for g_cls, _ in gts:
            gt_counts[g_cls] += 1

        matches, unmatched_preds, unmatched_gts = _greedy_match(preds, gts)

        n_tp = 0
        n_class_conf = 0
        n_fp = len(unmatched_preds)
        n_fn = len(unmatched_gts)

        for p_idx, g_idx, g_cls, iou, p_conf in matches:
            p_cls = preds[p_idx][0]
            conf_mat[LABELS[g_cls]][LABELS[p_cls]] += 1
            if p_cls == g_cls:
                n_tp += 1
                pred_records.append({
                    "cls": p_cls,
                    "conf": p_conf,
                    "is_tp": True,
                    "iou": iou,
                })
            else:
                n_class_conf += 1
                pred_records.append({
                    "cls": p_cls,
                    "conf": p_conf,
                    "is_tp": False,
                    "iou": iou,
                })

        for p_idx in unmatched_preds:
            p_cls, p_conf, _ = preds[p_idx]
            conf_mat[BACKGROUND_LABEL][LABELS[p_cls]] += 1
            pred_records.append({
                "cls": p_cls,
                "conf": p_conf,
                "is_tp": False,
                "iou": 0.0,
            })

        for g_idx in unmatched_gts:
            g_cls = gts[g_idx][0]
            conf_mat[LABELS[g_cls]][BACKGROUND_LABEL] += 1

        per_frame_rows.append({
            "frame": img_path.stem,
            "n_gt": len(gts),
            "n_pred": len(preds),
            "tp": n_tp,
            "class_confusion": n_class_conf,
            "fp": n_fp,
            "fn": n_fn,
            "total_errors": n_class_conf + n_fp + n_fn,
        })

    # ----- Confusion matrix CSV -----
    all_classes = list(LABELS) + [BACKGROUND_LABEL]
    rows: list[list[str]] = []
    for row_name in all_classes:
        row = [row_name]
        for col_name in all_classes:
            row.append(str(conf_mat[row_name].get(col_name, 0)))
        rows.append(row)
    (out_dir / "confusion_matrix.csv").write_text(
        ",".join(["gt\\pred"] + all_classes) + "\n"
        + "\n".join(",".join(r) for r in rows) + "\n"
    )

    # ----- Per-class metrics at each conf threshold -----
    metrics_lines = [
        "# Per-class precision / recall / F1 at multiple conf thresholds",
        "",
        f"Inference conf={low_conf}, IoU match threshold={IOU_MATCH_THRESHOLD}",
        f"Class-confusion (matched IoU, wrong class) is counted as FP for the predicted class and FN for the true class.",
        "",
    ]
    for thr in DEPLOY_CONF_THRESHOLDS:
        metrics_lines.append(f"\n## Threshold = {thr}")
        per_class_table: list[list[str]] = []
        for cls_idx in range(n_classes):
            tp_at = sum(1 for r in pred_records if r["cls"] == cls_idx and r["is_tp"] and r["conf"] >= thr)
            fp_at = sum(1 for r in pred_records if r["cls"] == cls_idx and not r["is_tp"] and r["conf"] >= thr)
            gt_at = gt_counts.get(cls_idx, 0)
            fn_at = gt_at - tp_at
            p = tp_at / (tp_at + fp_at) if (tp_at + fp_at) > 0 else 0.0
            r_ = tp_at / gt_at if gt_at > 0 else 0.0
            f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0
            per_class_table.append([
                LABELS[cls_idx],
                str(gt_at), str(tp_at), str(fp_at), str(fn_at),
                f"{p:.3f}", f"{r_:.3f}", f"{f1:.3f}",
            ])
        metrics_lines.append(_format_table(
            per_class_table,
            ["class", "GT", "TP", "FP", "FN", "P", "R", "F1"],
        ))
    (out_dir / "per_class_metrics.md").write_text("\n".join(metrics_lines) + "\n")

    # ----- Worst frames -----
    df = pl.DataFrame(per_frame_rows)
    worst = df.sort("total_errors", descending=True).head(TOP_WORST_FRAMES)
    worst_lines = [
        f"# Top {TOP_WORST_FRAMES} val frames by error count (inference conf={low_conf})",
        "",
        "Errors = class_confusion + FP + FN. Pred conf threshold not applied — these are raw at low-conf inference.",
        "",
    ]
    worst_lines.append(_format_table(
        [
            [r["frame"], str(r["n_gt"]), str(r["n_pred"]), str(r["tp"]),
             str(r["class_confusion"]), str(r["fp"]), str(r["fn"]), str(r["total_errors"])]
            for r in worst.iter_rows(named=True)
        ],
        ["frame", "GT", "pred", "TP", "class_conf", "FP", "FN", "errors"],
    ))
    (out_dir / "worst_frames.md").write_text("\n".join(worst_lines) + "\n")

    # ----- Summary -----
    total_gt = sum(gt_counts.values())
    total_pred = sum(1 for r in pred_records)
    total_tp = sum(1 for r in pred_records if r["is_tp"])
    total_fp = sum(1 for r in pred_records if not r["is_tp"])
    total_fn = total_gt - total_tp

    overall_p = total_tp / total_pred if total_pred > 0 else 0.0
    overall_r = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    # Top confusable class pairs (gt_class -> pred_class, excluding background)
    pair_counts: list[tuple[str, str, int]] = []
    for gt_cls_name, row in conf_mat.items():
        for pred_cls_name, n in row.items():
            if gt_cls_name == pred_cls_name:
                continue
            pair_counts.append((gt_cls_name, pred_cls_name, n))
    pair_counts.sort(key=lambda t: -t[2])
    top_pairs = pair_counts[:15]

    summary = [
        "# Prod val error summary",
        "",
        f"Weights:  {weights}",
        f"Dataset:  {dataset}",
        f"Val:      {len(images)} frames, {total_gt} GT boxes",
        f"Pred (conf>={low_conf}):  {total_pred} boxes",
        "",
        "## Overall (matched-IoU, all conf >= low_conf)",
        "",
        f"- Total GT boxes:       {total_gt}",
        f"- TP (correct class):   {total_tp}",
        f"- FP / class-confusion: {total_fp}",
        f"- FN (missed GT):       {total_fn}",
        f"- Overall precision:    {overall_p:.3f}",
        f"- Overall recall:       {overall_r:.3f}",
        f"- Overall F1:           {overall_f1:.3f}",
        "",
        "## Top error transitions (gt -> pred)",
        "",
        "Each row counts boxes whose IoU-matched (or unmatched) pred has the other class label.",
        "`<background>` as gt = false positive (model invented a box). `<background>` as pred = false negative (model missed a GT box).",
        "",
    ]
    summary.append(_format_table(
        [[gt, pred, str(n)] for gt, pred, n in top_pairs],
        ["gt class", "pred class", "count"],
    ))
    summary.append("\n## Per-class GT box counts in val")
    summary.append("")
    summary.append(_format_table(
        [[LABELS[i], str(gt_counts.get(i, 0))] for i in range(n_classes)],
        ["class", "GT in val"],
    ))
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")

    logger.success(
        f"Wrote {out_dir}/\n"
        f"  - summary.md\n"
        f"  - per_class_metrics.md\n"
        f"  - confusion_matrix.csv\n"
        f"  - worst_frames.md"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--low-conf", type=float, default=LOW_CONF)
    args = ap.parse_args()
    main(args.weights, args.dataset, args.out_dir, args.low_conf)
