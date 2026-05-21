"""Train the first full 13-class detector on the patient-merged corpus.

Same single-config setup as patient_v2 (yolo11s, img640, 80 epochs, AdamW
auto-optimizer) — this is a fast first pass to confirm the merge produced
trainable data and to establish a baseline before considering larger configs.

For 13 classes the auto-optimizer heuristic picks a lower lr0 than the
1-class patient runs (lr_fit = 0.002 * 5 / (4 + nc) ≈ 0.0006 here vs 0.002
when nc=1), so training dynamics will look different from the patient bootstraps.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from ent_cv.config import MODELS_DIR
from ent_cv.modeling.train import run as train_run

DATA_YAML = Path("/mnt/data/ent_cv/datasets/current_with_patient/data.yaml")
BASE_MODEL = "yolo11s.pt"
EPOCHS = 80
IMGSZ = 640


def main(dry_run: bool = False) -> None:
    project = MODELS_DIR / f"full_v0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        name_suffix="full_v0_img640",
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
