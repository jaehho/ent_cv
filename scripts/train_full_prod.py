"""Production training: yolo11m, imgsz=1024, 120 epochs on current_full_v2.

The bigger model + higher resolution targets the strong-class plateau seen
in full_v1 (mAP50 ~0.82 on the 9 base classes) and the small-instance
classes that benefit most from imgsz=1024. Same dataset as full_v2 so the
"v2-yolo11s-640 → prod-yolo11m-1024" delta isolates capacity + resolution.

Patience is left at default (100); with 120 epochs that's effectively
"no early stop" so the cosine schedule (if enabled) or natural plateau
plays out. Set --dry-run to print plan without launching.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from ent_cv.config import MODELS_DIR
from ent_cv.modeling.train import run as train_run

DATA_YAML = Path("/mnt/data/ent_cv/datasets/current_full_v2/data.yaml")
BASE_MODEL = "yolo11m.pt"
EPOCHS = 120
IMGSZ = 1024


def main(dry_run: bool = False) -> None:
    project = MODELS_DIR / f"full_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project.mkdir(parents=True, exist_ok=True)
    logger.info(f"Project dir: {project}")
    logger.info(f"Data:        {DATA_YAML}")
    logger.info(f"Model:       {BASE_MODEL}  imgsz={IMGSZ}  epochs={EPOCHS}")

    if dry_run:
        logger.info("--- DRY RUN ---")
        return

    result = train_run(
        data=DATA_YAML,
        model=BASE_MODEL,
        epochs=EPOCHS,
        project=project,
        name_suffix="full_prod_img1024",
        imgsz=IMGSZ,
    )
    rd = getattr(result, "results_dict", {}) or {}
    save_dir = str(getattr(result, "save_dir", "") or "")
    logger.success(
        "Done.\n"
        f"  weights:  {Path(save_dir) / 'weights' / 'best.pt'}\n"
        f"  mAP50:    {float(rd.get('metrics/mAP50(B)', -1.0)):.4f}\n"
        f"  mAP50-95: {float(rd.get('metrics/mAP50-95(B)', -1.0)):.4f}\n"
        f"  P:        {float(rd.get('metrics/precision(B)', -1.0)):.3f}\n"
        f"  R:        {float(rd.get('metrics/recall(B)', -1.0)):.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print plan without training.")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
