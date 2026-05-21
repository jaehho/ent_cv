"""Render GT-vs-prediction overlays for one case's val frames.

For each frame in the val split that belongs to the target case, draws:
  - GT boxes in GREEN with class label
  - Prod prediction boxes (conf >= --conf) in RED with class label + conf

Outputs (under repo's gitignored reports/):
  reports/prod_val_errors/audit_<case>/
    worst/<rank>_<frame>.png   # top N frames by error count
    sample/<frame>.png         # random sample of "low-error" frames for contrast
    index.md                   # links + per-frame notes (error counts)

Designed for the audit step before deciding whether a case has noisy GT or just
needs more training data.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import LABELS
from ent_cv.gpu import gpu_yield

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path("/mnt/data/ent_cv/models/best/weights/best.pt")
DEFAULT_DATASET = Path("/mnt/data/ent_cv/datasets/current_full_v2")
DEFAULT_OUT = REPO_ROOT / "reports" / "prod_val_errors"
IOU_MATCH_THRESHOLD = 0.5
GT_COLOR = (0, 200, 0)            # BGR — green
PRED_COLOR = (0, 0, 230)          # BGR — red
THICKNESS = 3
LABEL_BG_ALPHA = 0.7


def _xywhn_to_xyxy(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    return (
        int((cx - bw / 2) * w),
        int((cy - bh / 2) * h),
        int((cx + bw / 2) * w),
        int((cy + bh / 2) * h),
    )


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        box = tuple(float(p) for p in parts[1:])  # type: ignore[assignment]
        out.append((cls, _xywhn_to_xyxy(box, w, h)))  # type: ignore[arg-type]
    return out


def _draw_box(
    img: np.ndarray,
    xyxy: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    align_bottom: bool = False,
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.7
    fthick = 2
    (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
    pad = 4
    if align_bottom:
        ty = min(y2 + th + pad * 2, img.shape[0] - 2)
        ly = ty - th - pad
    else:
        ty = max(y1 - pad, th + pad)
        ly = ty - th - pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, ly), (x1 + tw + pad * 2, ly + th + pad * 2), color, -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, img, 1 - LABEL_BG_ALPHA, 0, img)
    cv2.putText(img, label, (x1 + pad, ty), font, fscale, (255, 255, 255), fthick, cv2.LINE_AA)


def _render(
    image_path: Path,
    gts: list[tuple[int, tuple[int, int, int, int]]],
    preds: list[tuple[int, float, tuple[int, int, int, int]]],
) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    for cls, box in gts:
        _draw_box(img, box, f"GT: {LABELS[cls]}", GT_COLOR, align_bottom=False)
    for cls, conf, box in preds:
        _draw_box(img, box, f"P: {LABELS[cls]} {conf:.2f}", PRED_COLOR, align_bottom=True)
    return img


def main(
    case: str,
    weights: Path,
    dataset: Path,
    out_root: Path,
    conf: float,
    n_worst: int,
    n_sample: int,
    sample_seed: int,
) -> None:
    val_split = dataset / "val_split.txt"
    labels_dir = dataset / "labels" / "train"
    images_all = [Path(ln.strip()) for ln in val_split.read_text().splitlines() if ln.strip()]
    images = [p for p in images_all if p.stem.startswith(case + "_") or p.stem.startswith(case + ".")]
    logger.info(f"Case {case}: {len(images)} val frames")

    out_dir = out_root / f"audit_{case}"
    (out_dir / "worst").mkdir(parents=True, exist_ok=True)
    (out_dir / "sample").mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading weights: {weights}")
    model = cast("YOLO", YOLO(str(weights)))

    per_frame: list[dict] = []

    with gpu_yield("0"):
        for idx, img_path in enumerate(images):
            if idx % 50 == 0:
                logger.info(f"  inference: {idx}/{len(images)}")
            result = model.predict(
                source=str(img_path),
                conf=conf,
                iou=0.7,
                imgsz=1024,
                device="0",
                verbose=False,
            )[0]
            h, w = result.orig_shape
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy().astype(int)
                preds = [
                    (int(clses[i]), float(confs[i]), tuple(boxes_xyxy[i].tolist()))
                    for i in range(len(clses))
                ]
            else:
                preds = []

            label_path = labels_dir / f"{img_path.stem}.txt"
            gts = _load_gt(label_path, w, h)

            # Compute simple per-frame errors via greedy match
            used_pred = [False] * len(preds)
            used_gt = [False] * len(gts)
            pairs: list[tuple[int, int, float]] = []
            for pi, (_, _, pbox) in enumerate(preds):
                for gi, (_, gbox) in enumerate(gts):
                    iou = _iou_xyxy(pbox, gbox)
                    if iou >= IOU_MATCH_THRESHOLD:
                        pairs.append((pi, gi, iou))
            pairs.sort(key=lambda t: -t[2])
            tp = class_conf = 0
            for pi, gi, _ in pairs:
                if used_pred[pi] or used_gt[gi]:
                    continue
                used_pred[pi] = True
                used_gt[gi] = True
                if preds[pi][0] == gts[gi][0]:
                    tp += 1
                else:
                    class_conf += 1
            fp = sum(1 for u in used_pred if not u)
            fn = sum(1 for u in used_gt if not u)
            per_frame.append({
                "image_path": img_path,
                "gts": gts,
                "preds": preds,
                "tp": tp,
                "class_conf": class_conf,
                "fp": fp,
                "fn": fn,
                "errors": class_conf + fp + fn,
            })

    per_frame.sort(key=lambda r: (-r["errors"], r["image_path"].stem))
    worst = per_frame[:n_worst]
    rest = per_frame[n_worst:]
    rng = random.Random(sample_seed)
    sample = rng.sample(rest, min(n_sample, len(rest))) if rest else []

    logger.info(f"Rendering {len(worst)} worst and {len(sample)} sample frames")
    index_lines = [
        f"# Audit: case {case}",
        "",
        f"Weights:   {weights}",
        f"Dataset:   {dataset}",
        f"Frames:    {len(images)} val frames from this case",
        f"Conf:      preds shown if conf >= {conf}",
        "",
        "GT boxes = GREEN. Pred boxes = RED. Box errors = class_confusion + FP + FN.",
        "",
        f"## Worst {len(worst)} frames",
        "",
        "| rank | frame | GT | pred | TP | class_conf | FP | FN | errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(worst, 1):
        stem = row["image_path"].stem
        out_path = out_dir / "worst" / f"{rank:02d}_{stem}.png"
        cv2.imwrite(str(out_path), _render(row["image_path"], row["gts"], row["preds"]))
        index_lines.append(
            f"| {rank} | [{stem}](worst/{out_path.name}) | "
            f"{len(row['gts'])} | {len(row['preds'])} | {row['tp']} | "
            f"{row['class_conf']} | {row['fp']} | {row['fn']} | {row['errors']} |"
        )

    index_lines += [
        "",
        f"## Sample of {len(sample)} non-worst frames (for contrast)",
        "",
        "| frame | GT | pred | TP | class_conf | FP | FN | errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sample:
        stem = row["image_path"].stem
        out_path = out_dir / "sample" / f"{stem}.png"
        cv2.imwrite(str(out_path), _render(row["image_path"], row["gts"], row["preds"]))
        index_lines.append(
            f"| [{stem}](sample/{out_path.name}) | "
            f"{len(row['gts'])} | {len(row['preds'])} | {row['tp']} | "
            f"{row['class_conf']} | {row['fp']} | {row['fn']} | {row['errors']} |"
        )

    # Class-error breakdown across the whole case
    gt_class_counts: dict[int, int] = defaultdict(int)
    miscls_pairs: dict[tuple[int, int], int] = defaultdict(int)
    fp_class_counts: dict[int, int] = defaultdict(int)
    fn_class_counts: dict[int, int] = defaultdict(int)

    for row in per_frame:
        for cls, _ in row["gts"]:
            gt_class_counts[cls] += 1
        # rebuild matches to attribute errors per class
        preds = row["preds"]
        gts = row["gts"]
        pairs: list[tuple[int, int, float]] = []
        for pi, (_, _, pbox) in enumerate(preds):
            for gi, (_, gbox) in enumerate(gts):
                iou = _iou_xyxy(pbox, gbox)
                if iou >= IOU_MATCH_THRESHOLD:
                    pairs.append((pi, gi, iou))
        pairs.sort(key=lambda t: -t[2])
        used_pred = [False] * len(preds)
        used_gt = [False] * len(gts)
        for pi, gi, _ in pairs:
            if used_pred[pi] or used_gt[gi]:
                continue
            used_pred[pi] = True
            used_gt[gi] = True
            p_cls, _, _ = preds[pi]
            g_cls, _ = gts[gi]
            if p_cls != g_cls:
                miscls_pairs[(g_cls, p_cls)] += 1
        for pi, used in enumerate(used_pred):
            if not used:
                fp_class_counts[preds[pi][0]] += 1
        for gi, used in enumerate(used_gt):
            if not used:
                fn_class_counts[gts[gi][0]] += 1

    index_lines += [
        "",
        "## Whole-case error breakdown",
        "",
        f"Total GT boxes:       {sum(gt_class_counts.values())}",
        f"Total preds rendered: {sum(len(r['preds']) for r in per_frame)}",
        "",
        "### Class confusions (gt -> pred, different class)",
        "",
        "| gt | pred | count |",
        "| --- | --- | --- |",
    ]
    for (gt_cls, pred_cls), n in sorted(miscls_pairs.items(), key=lambda kv: -kv[1]):
        index_lines.append(f"| {LABELS[gt_cls]} | {LABELS[pred_cls]} | {n} |")

    index_lines += [
        "",
        "### False positives by class (pred without matching GT)",
        "",
        "| class | count |",
        "| --- | --- |",
    ]
    for cls, n in sorted(fp_class_counts.items(), key=lambda kv: -kv[1]):
        index_lines.append(f"| {LABELS[cls]} | {n} |")

    index_lines += [
        "",
        "### False negatives by class (GT with no matching pred)",
        "",
        "| class | count |",
        "| --- | --- |",
    ]
    for cls, n in sorted(fn_class_counts.items(), key=lambda kv: -kv[1]):
        index_lines.append(f"| {LABELS[cls]} | {n} |")

    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n")

    logger.success(
        f"Wrote {out_dir}/\n"
        f"  - index.md   (start here)\n"
        f"  - worst/     {len(worst)} PNGs\n"
        f"  - sample/    {len(sample)} PNGs"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="Case ID, e.g. 20251218_02")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n-worst", type=int, default=30)
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--sample-seed", type=int, default=42)
    args = ap.parse_args()
    main(
        case=args.case,
        weights=args.weights,
        dataset=args.dataset,
        out_root=args.out_root,
        conf=args.conf,
        n_worst=args.n_worst,
        n_sample=args.n_sample,
        sample_seed=args.sample_seed,
    )
