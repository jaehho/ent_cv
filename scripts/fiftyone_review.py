"""Load combined YOLO dataset + per-case predictions into FiftyOne for review.

Usage:
    uv run python scripts/fiftyone_review.py            # build + launch App
    uv run python scripts/fiftyone_review.py --no-app   # build only
    uv run python scripts/fiftyone_review.py --raw      # use raw detections.json instead of filtered

Open the App at http://localhost:5151 once it launches. Filter by label, sort by
confidence, or use Actions -> Mistakenness to find suspicious annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import fiftyone as fo
from PIL import Image
from tqdm import tqdm

from ent_cv.config import DATASETS_DIR, PREDICTIONS_DIR

DATASET_DIR = DATASETS_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data_with_val.yaml"

# 20251113_02_Part1_f000000  -> case=20251113_02, part=1, local_frame=0
STEM_RE = re.compile(r"^(?P<case>\d{8}_\d{2})_Part(?P<part>\d+)_f(?P<frame>\d+)$")
PART_RE = re.compile(r".*_Part(\d+)\.mp4$")

DATASET_NAME = "ent_cv"


def parse_stem(stem: str) -> tuple[str, int, int] | None:
    m = STEM_RE.match(stem)
    if not m:
        return None
    return m["case"], int(m["part"]), int(m["frame"])


def load_part_offsets(case: str) -> dict[int, int]:
    """Return {part_number: global_frame_offset_at_part_start}."""
    meta_path = PREDICTIONS_DIR / case / "metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    parts = sorted(meta["part_frames"].items())
    offsets: dict[int, int] = {}
    cumulative = 0
    for filename, count in parts:
        m = PART_RE.match(filename)
        if not m:
            continue
        offsets[int(m[1])] = cumulative
        cumulative += count
    return offsets


def load_predictions(case: str, kind: str) -> dict[int, list[dict]]:
    """Return {global_frame: [det, ...]} for the case. Empty if missing."""
    pred_path = PREDICTIONS_DIR / case / f"{kind}.json"
    if not pred_path.exists():
        return {}
    raw = json.loads(pred_path.read_text())
    dets = raw["detections"] if isinstance(raw, dict) else raw
    by_frame: dict[int, list[dict]] = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)
    return by_frame


def to_fo_detections(dets: list[dict], img_w: int, img_h: int) -> fo.Detections:
    out: list[fo.Detection] = []
    for d in dets:
        b = d["box"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        out.append(
            fo.Detection(
                label=d["name"],
                bounding_box=[
                    x1 / img_w,
                    y1 / img_h,
                    (x2 - x1) / img_w,
                    (y2 - y1) / img_h,
                ],
                confidence=float(d.get("confidence", 0.0)),
            )
        )
    return fo.Detections(detections=out)


def build_dataset(pred_kind: str) -> fo.Dataset:
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)

    print(f"Loading YOLOv5 dataset from {DATA_YAML}...")
    ds = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path=str(DATA_YAML),
        split="train",
        tags=["train"],
        name=DATASET_NAME,
    )
    ds.add_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path=str(DATA_YAML),
        split="val",
        tags=["val"],
    )
    ds.persistent = True

    case_offsets: dict[str, dict[int, int]] = {}
    case_preds: dict[str, dict[int, list[dict]]] = {}
    img_size_cache: dict[str, tuple[int, int]] = {}

    matched = 0
    missing_meta: set[str] = set()

    print(f"Attaching {pred_kind} predictions...")
    for sample in tqdm(ds.iter_samples(autosave=True), total=len(ds)):
        stem = Path(sample.filepath).stem
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        case, part, local_frame = parsed
        sample["case"] = case
        sample["part"] = part
        sample["local_frame"] = local_frame

        if case not in case_offsets:
            offsets = load_part_offsets(case)
            if not offsets:
                missing_meta.add(case)
            case_offsets[case] = offsets
            case_preds[case] = load_predictions(case, pred_kind)

        offsets = case_offsets[case]
        if part not in offsets:
            continue
        global_frame = offsets[part] + local_frame
        sample["global_frame"] = global_frame

        dets = case_preds[case].get(global_frame, [])
        if not dets:
            continue

        if sample.filepath not in img_size_cache:
            with Image.open(sample.filepath) as im:
                img_size_cache[sample.filepath] = im.size
        w, h = img_size_cache[sample.filepath]
        sample["predictions"] = to_fo_detections(dets, w, h)
        matched += 1

    if missing_meta:
        print(f"WARNING: missing metadata.json for cases: {sorted(missing_meta)}")
    print(f"Attached predictions to {matched}/{len(ds)} samples")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw detections.json instead of filtered_detections.json",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Build dataset only; do not launch the App",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for the FiftyOne App (default 5151)",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing 'ent_cv' dataset if present (skip rebuild)",
    )
    args = parser.parse_args()

    if args.reuse and fo.dataset_exists(DATASET_NAME):
        print(f"Reusing existing dataset '{DATASET_NAME}'")
        ds = fo.load_dataset(DATASET_NAME)
    else:
        pred_kind = "detections" if args.raw else "filtered_detections"
        ds = build_dataset(pred_kind)

    print(f"\nDataset summary: {len(ds)} samples")
    print(f"  Cases: {len(ds.distinct('case'))}")
    print(f"  Splits: {ds.distinct('tags')}")
    counts = ds.count_values("ground_truth.detections.label")
    if counts:
        print("  Ground-truth label counts:")
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
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
