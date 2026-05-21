"""Tight 1-class Patient bootstrap sweep — imgsz × lr0 on yolo11s.

Grid: 2 imgsz × 3 lr0 = 6 configs. 60 epochs each. Single architecture (yolo11s)
since the task ("any visible facial tissue") is easy and the dataset is small
(~144 boxes / 4 train cases / 1 val case).

Results CSV: /mnt/data/ent_cv/models/patient_v0_sweep_<ts>/results.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from loguru import logger

from ent_cv.config import MODELS_DIR
from ent_cv.modeling.train import run as train_run

DATA_YAML = Path("/mnt/data/ent_cv/datasets/patient_v0/data_with_val.yaml")
BASE_MODEL = "yolo11s.pt"
EPOCHS = 60

IMG_SIZES: list[int] = [640, 1024]
LR0_VALUES: list[float] = [0.001, 0.005, 0.01]


@dataclass
class Variant:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)


VARIANTS: list[Variant] = [
    Variant(
        label=f"img{imgsz}_lr{lr0}",
        overrides={"imgsz": imgsz, "lr0": lr0},
    )
    for imgsz in IMG_SIZES
    for lr0 in LR0_VALUES
]

CSV_FIELDS = [
    "label", "model", "epochs", "imgsz", "lr0",
    "mAP50_95", "mAP50", "precision", "recall",
    "duration_min", "save_dir", "error",
]


def run_variant(v: Variant, *, project: Path) -> dict[str, Any]:
    logger.info(f"[{v.label}] starting — {v.overrides}")
    t0 = time.time()
    try:
        result = train_run(
            data=DATA_YAML,
            model=BASE_MODEL,
            epochs=EPOCHS,
            project=project,
            name_suffix=f"patient_v0_{v.label}",
            **v.overrides,
        )
        rd = getattr(result, "results_dict", {}) or {}
        save_dir = str(getattr(result, "save_dir", "") or "")
        row = {
            "label": v.label,
            "model": BASE_MODEL,
            "epochs": EPOCHS,
            "imgsz": v.overrides["imgsz"],
            "lr0": v.overrides["lr0"],
            "mAP50_95": float(rd.get("metrics/mAP50-95(B)", -1.0)),
            "mAP50": float(rd.get("metrics/mAP50(B)", -1.0)),
            "precision": float(rd.get("metrics/precision(B)", -1.0)),
            "recall": float(rd.get("metrics/recall(B)", -1.0)),
            "duration_min": round((time.time() - t0) / 60, 1),
            "save_dir": save_dir,
            "error": "",
        }
        logger.success(
            f"[{v.label}] mAP50={row['mAP50']:.4f}  "
            f"mAP50-95={row['mAP50_95']:.4f}  ({row['duration_min']} min)"
        )
        return row
    except Exception as exc:
        logger.error(f"[{v.label}] FAILED: {exc}")
        return {
            "label": v.label,
            "model": BASE_MODEL,
            "epochs": EPOCHS,
            "imgsz": v.overrides["imgsz"],
            "lr0": v.overrides["lr0"],
            "mAP50_95": -1.0,
            "mAP50": -1.0,
            "precision": -1.0,
            "recall": -1.0,
            "duration_min": round((time.time() - t0) / 60, 1),
            "save_dir": "",
            "error": str(exc)[:200],
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def main(dry_run: bool = False) -> None:
    sweep_dir = MODELS_DIR / f"patient_v0_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "results.csv"
    logger.info(f"Sweep dir: {sweep_dir}")
    logger.info(f"Results CSV: {csv_path}")
    logger.info(f"Data: {DATA_YAML}")
    logger.info(f"Configs ({len(VARIANTS)}):")
    for v in VARIANTS:
        logger.info(f"  {v.label}  -> {v.overrides}")

    if dry_run:
        logger.info("--- DRY RUN ---")
        return

    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        rows.append(run_variant(v, project=sweep_dir))
        write_csv(csv_path, rows)

    # Rank
    ok = sorted([r for r in rows if not r["error"]], key=lambda r: r["mAP50_95"], reverse=True)
    failed = [r for r in rows if r["error"]]
    logger.info("--- Final ranking (by mAP50-95) ---")
    for r in ok:
        logger.info(
            f"  {r['label']:>16}  mAP50-95={r['mAP50_95']:.4f}  mAP50={r['mAP50']:.4f}  "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  ({r['duration_min']}m)"
        )
    for r in failed:
        logger.info(f"  {r['label']:>16}  FAILED: {r['error']}")

    if ok:
        winner = ok[0]
        logger.success(
            f"Winner: {winner['label']}  mAP50-95={winner['mAP50_95']:.4f}\n"
            f"  weights: {Path(winner['save_dir']) / 'weights' / 'best.pt'}"
        )
    logger.success(f"Sweep complete. Results: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print plan without training")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
