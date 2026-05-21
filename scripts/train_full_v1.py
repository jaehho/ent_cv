"""Train the v1 full 13-class detector on the HN-extended corpus.

Same single-config setup as train_full_v0 (yolo11s, img640, 80 epochs, AdamW
auto-optimizer) — keeping everything identical except the dataset so the v0
vs v1 metric delta is interpretable as "what HN mining + 3 new cases bought."

For 13 classes the auto-optimizer heuristic picks lr_fit ≈ 0.0006
(0.002 * 5 / (4 + nc)), same as v0.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from ent_cv.config import MODELS_DIR
from ent_cv.modeling.train import run as train_run

DATA_YAML = Path("/mnt/data/ent_cv/datasets/current_full_v1/data.yaml")
BASE_MODEL = "yolo11s.pt"
EPOCHS = 80
IMGSZ = 640


def main(dry_run: bool = False) -> None:
    project = MODELS_DIR / f"full_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        name_suffix="full_v1_img640",
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
