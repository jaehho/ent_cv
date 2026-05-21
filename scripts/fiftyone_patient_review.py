"""Spot-check Patient pseudo-labels against existing instrument GT in FiftyOne.

The dataset shows three fields per frame:
  - ``ground_truth``  : existing multi-class instrument boxes (no Patient).
  - ``predictions``   : patient_v0 pseudo-labels (Patient only).
  - ``no_pseudo``     : boolean tag — True when the detector emitted no box
                       (worth a look: probably a true negative, sometimes a miss).

Filter tips in the App:
  * ``no_pseudo == True``  — frames the detector skipped
  * ``predictions.detections.confidence < 0.7``  — borderline pseudo-labels
  * Group by ``case`` to scan one surgical case at a time

Usage:
    uv run python scripts/fiftyone_patient_review.py            # build + launch
    uv run python scripts/fiftyone_patient_review.py --no-app   # build only
    uv run python scripts/fiftyone_patient_review.py --reuse    # reuse existing dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import fiftyone as fo
from tqdm import tqdm

from ent_cv.config import DATA_DIR, DATASETS_DIR, LABELS

DATASET_NAME = "ent_cv_patient_v0"
IMAGES_DIR = DATASETS_DIR / "current" / "images" / "train"
GT_LABELS_DIR = DATASETS_DIR / "current" / "labels" / "train"
PSEUDO_LABELS_DIR = DATA_DIR / "predictions" / "patient_v0_pseudo" / "labels"

CASE_RE = re.compile(r"^(?P<case>\d{8}_\d{2})_Part(?P<part>\d+)_f(?P<frame>\d+)$")


def _yolo_to_fo(line: str, names: list[str]) -> fo.Detection | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(parts[0])
    cx, cy, w, h = (float(x) for x in parts[1:])
    return fo.Detection(
        label=names[cls] if 0 <= cls < len(names) else f"cls_{cls}",
        bounding_box=[cx - w / 2, cy - h / 2, w, h],
    )


def _read_yolo_file(p: Path, names: list[str]) -> list[fo.Detection]:
    if not p.exists():
        return []
    dets: list[fo.Detection] = []
    for line in p.read_text().splitlines():
        d = _yolo_to_fo(line, names)
        if d is not None:
            dets.append(d)
    return dets


def build_dataset() -> fo.Dataset:
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)

    images = sorted(IMAGES_DIR.glob("*.png"))
    print(f"Loading {len(images)} images...")
    ds = fo.Dataset(name=DATASET_NAME, persistent=True)

    samples: list[fo.Sample] = []
    for img in tqdm(images, desc="building"):
        gt = _read_yolo_file(GT_LABELS_DIR / f"{img.stem}.txt", list(LABELS))
        pseudo = _read_yolo_file(PSEUDO_LABELS_DIR / f"{img.stem}.txt", list(LABELS))

        s = fo.Sample(filepath=str(img))
        m = CASE_RE.match(img.stem)
        if m:
            s["case"] = m["case"]
            s["part"] = int(m["part"])
            s["frame"] = int(m["frame"])
        s["ground_truth"] = fo.Detections(detections=gt)
        s["predictions"] = fo.Detections(detections=pseudo)
        s["no_pseudo"] = not pseudo
        s["box_count"] = len(pseudo)

        # Pre-tag triage views so the user can jump straight to them in the
        # App. Tagging is the only edit primitive we trust (in-App detection
        # edits have a buggy history through 1.13/1.14; we round-trip through
        # CVAT for geometry fixes instead).
        sample_tags: list[str] = []
        if not pseudo:
            sample_tags.append("needs_fix")  # all 67 misses surface here
        if len(pseudo) >= 2:
            sample_tags.append("multi_box")  # 29 suspect frames
        s.tags = sample_tags

        samples.append(s)

    ds.add_samples(samples)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-app", action="store_true", help="Build dataset, do not launch App.")
    ap.add_argument("--reuse", action="store_true", help="Reuse existing dataset if present.")
    ap.add_argument("--port", type=int, default=5151)
    args = ap.parse_args()

    if args.reuse and fo.dataset_exists(DATASET_NAME):
        print(f"Reusing existing dataset '{DATASET_NAME}'")
        ds = fo.load_dataset(DATASET_NAME)
    else:
        ds = build_dataset()

    print(f"\nSamples: {len(ds)}")
    print(f"  Cases: {len(ds.distinct('case'))}")
    n_no_pseudo = ds.match({"no_pseudo": True}).count()
    print(f"  Frames with no pseudo-label: {n_no_pseudo}")
    print("  Pre-tagged views:")
    for tag in ("needs_fix", "multi_box"):
        n = ds.match_tags(tag).count()
        print(f"    {tag:<12} {n}")
    print(
        "  Tip: tag your own bad-geometry frames with the 't' key in the App, "
        "then run scripts/upload_patient_fixes_to_cvat.py."
    )

    gt_counts = ds.count_values("ground_truth.detections.label")
    if gt_counts:
        print("  Ground-truth label counts:")
        for label, n in sorted(gt_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n:5d}  {label}")
    pred_counts = ds.count_values("predictions.detections.label")
    if pred_counts:
        print("  Prediction label counts:")
        for label, n in sorted(pred_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n:5d}  {label}")

    if args.no_app:
        return
    print(f"\nLaunching App on port {args.port}...")
    session = fo.launch_app(ds, port=args.port, address="0.0.0.0")
    session.wait()


if __name__ == "__main__":
    main()
