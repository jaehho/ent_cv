"""Train patient_v1 on the v0 corpus + 90 hand-corrected frames from task 106.

Single config, no sweep:
  - yolo11s (won v0, plenty of capacity for "any visible facial tissue")
  - imgsz=640 (beat 1024 on v0)
  - 80 epochs (up from v0's 60 — 1.6x more data, longer ramp to convergence)

Why no lr0 sweep: v0's lr0 grid produced byte-identical train losses because
Ultralytics's optimizer="auto" silently overrides lr0 with a per-class-count
heuristic (lr_fit = 0.002 * 5 / (4 + nc)). Sweeping it again would be theatre.

The eval target afterward is a fresh held-out test set on cases neither v0 nor
v1 has seen — see scripts/create_patient_test_set.py once you decide which
unseen cases to pull.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from ent_cv.config import MODELS_DIR
from ent_cv.modeling.train import run as train_run

DATA_YAML = Path("/mnt/data/ent_cv/datasets/patient_v1/data.yaml")
BASE_MODEL = "yolo11s.pt"
EPOCHS = 80
IMGSZ = 640


def main(dry_run: bool = False) -> None:
    project = MODELS_DIR / f"patient_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        name_suffix="patient_v1_img640",
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
