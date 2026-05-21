"""Focused hparam sweep tuned for the grasp-coupled dataset.

Background: a broad arch/size sweep (ent-cv sweep) is overkill here — we already
have a strong yolo26s baseline (mAP50=0.819 / mAP50-95=0.741 at 200 ep). This
script targets the knobs most likely to move the needle for a small, imbalanced,
fine-detail dataset:

  Phase 1  (yolo26s, 100 ep each) — 8 hparam variants:
    A baseline
    B schedule:                cos_lr=True, close_mosaic=50
    C imbalance/regularize:    B + cls=0.8, mixup=0.15
    D aug cocktail:            B + degrees=10, perspective=0.0005, hsv stronger
    E freeze backbone:         B + freeze=10
    F weight decay:            B + weight_decay=1e-3
    G higher resolution:       B + imgsz=1280
    H multi-scale:             B + multi_scale=0.5, rect=False

  Phase 2 (100 ep) — winner config on yolo26m and yolo11s (arch sanity).
  Phase 3 (300 ep) — overall winner with patience=50.

Ranking is by mAP50-95 (more discriminating than mAP50 at this performance level).

Results CSV: /mnt/data/ent_cv/models/smart_sweep_<ts>/results.csv

Run: uv run python scripts/smart_sweep.py
     uv run python scripts/smart_sweep.py --dry-run
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

DATA_YAML = Path("/mnt/data/ent_cv/datasets/current/data_with_val.yaml")


@dataclass
class Variant:
    label: str
    description: str
    model: str
    epochs: int
    overrides: dict[str, Any] = field(default_factory=dict)


# --- Phase 1: hparam variants at yolo26s -------------------------------------
P1_VARIANTS: list[Variant] = [
    Variant("A", "baseline", "yolo26s.pt", 100, {}),
    Variant("B", "schedule (cos_lr + close_mosaic=50)", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50,
    }),
    Variant("C", "B + imbalance (cls=0.8, mixup=0.15)", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50, "cls": 0.8, "mixup": 0.15,
    }),
    Variant("D", "B + aug cocktail (degrees, perspective, hsv)", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50,
        "degrees": 10.0, "perspective": 0.0005,
        "hsv_h": 0.03, "hsv_s": 0.8, "hsv_v": 0.5,
    }),
    Variant("E", "B + freeze=10 (backbone)", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50, "freeze": 10,
    }),
    Variant("F", "B + weight_decay=1e-3", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50, "weight_decay": 1e-3,
    }),
    Variant("G", "B + imgsz=1280", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50, "imgsz": 1280,
    }),
    Variant("H", "B + multi_scale=0.5 + rect=False", "yolo26s.pt", 100, {
        "cos_lr": True, "close_mosaic": 50, "multi_scale": 0.5, "rect": False,
    }),
]

CSV_FIELDS = [
    "phase", "label", "description", "model", "epochs",
    "mAP50_95", "mAP50", "precision", "recall",
    "duration_min", "save_dir", "overrides_json", "error",
]


def run_variant(v: Variant, *, project: Path, phase: str) -> dict[str, Any]:
    """Run one training; capture results or error. Never raises."""
    logger.info(f"[{phase}] {v.label} — {v.description}  ({v.model}, {v.epochs} ep)")
    logger.info(f"        overrides: {v.overrides}")
    t0 = time.time()
    try:
        result = train_run(
            data=DATA_YAML,
            model=v.model,
            epochs=v.epochs,
            project=project,
            name_suffix=v.label,
            **v.overrides,
        )
        rd = getattr(result, "results_dict", {}) or {}
        save_dir = str(getattr(result, "save_dir", "") or "")
        row = {
            "phase": phase,
            "label": v.label,
            "description": v.description,
            "model": v.model,
            "epochs": v.epochs,
            "mAP50_95": float(rd.get("metrics/mAP50-95(B)", -1.0)),
            "mAP50": float(rd.get("metrics/mAP50(B)", -1.0)),
            "precision": float(rd.get("metrics/precision(B)", -1.0)),
            "recall": float(rd.get("metrics/recall(B)", -1.0)),
            "duration_min": round((time.time() - t0) / 60, 1),
            "save_dir": save_dir,
            "overrides_json": json.dumps(v.overrides, default=str),
            "error": "",
        }
        logger.success(
            f"[{phase}] {v.label}: mAP50={row['mAP50']:.4f}  "
            f"mAP50-95={row['mAP50_95']:.4f}  ({row['duration_min']} min)"
        )
        return row
    except Exception as exc:
        logger.error(f"[{phase}] {v.label} FAILED: {exc}")
        return {
            "phase": phase,
            "label": v.label,
            "description": v.description,
            "model": v.model,
            "epochs": v.epochs,
            "mAP50_95": -1.0,
            "mAP50": -1.0,
            "precision": -1.0,
            "recall": -1.0,
            "duration_min": round((time.time() - t0) / 60, 1),
            "save_dir": "",
            "overrides_json": json.dumps(v.overrides, default=str),
            "error": str(exc)[:200],
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def pick_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if not r["error"] and r["mAP50_95"] >= 0]
    if not ok:
        return None
    return max(ok, key=lambda r: r["mAP50_95"])


def print_ranked(rows: list[dict[str, Any]], title: str) -> None:
    logger.info(f"--- {title} ---")
    ok = sorted([r for r in rows if not r["error"]], key=lambda r: r["mAP50_95"], reverse=True)
    failed = [r for r in rows if r["error"]]
    for r in ok:
        logger.info(
            f"  {r['label']:>5}  mAP50-95={r['mAP50_95']:.4f}  mAP50={r['mAP50']:.4f}  "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  "
            f"({r['duration_min']:.1f}m)  {r['description']}"
        )
    for r in failed:
        logger.info(f"  {r['label']:>5}  FAILED ({r['duration_min']:.1f}m): {r['error']}")


def main(dry_run: bool = False) -> None:
    sweep_dir = MODELS_DIR / f"smart_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "results.csv"
    logger.info(f"Sweep dir: {sweep_dir}")
    logger.info(f"Results CSV: {csv_path}")

    if dry_run:
        logger.info("--- DRY RUN ---")
        for v in P1_VARIANTS:
            logger.info(f"  P1 {v.label}: {v.description}  overrides={v.overrides}")
        logger.info("  P2: <winner config> on yolo26m and yolo11s")
        logger.info("  P3: <overall winner> at 300 ep, patience=50")
        logger.info(f"Total runs: {len(P1_VARIANTS) + 2 + 1}")
        return

    rows: list[dict[str, Any]] = []

    # Phase 1
    for v in P1_VARIANTS:
        rows.append(run_variant(v, project=sweep_dir, phase="P1"))
        write_csv(csv_path, rows)
    print_ranked([r for r in rows if r["phase"] == "P1"], "Phase 1 ranked")

    # Phase 2: winner config on different archs
    p1_winner = pick_winner([r for r in rows if r["phase"] == "P1"])
    if not p1_winner:
        logger.error("Phase 1 had no successful runs — aborting.")
        return
    winner_overrides = next(v.overrides for v in P1_VARIANTS if v.label == p1_winner["label"])
    logger.info(
        f"Phase 1 winner: {p1_winner['label']} ({p1_winner['description']}) "
        f"mAP50-95={p1_winner['mAP50_95']:.4f}"
    )

    p2_variants = [
        Variant(
            "P2-26m",
            f"P1 winner ({p1_winner['label']}) on yolo26m",
            "yolo26m.pt", 100, dict(winner_overrides),
        ),
        Variant(
            "P2-11s",
            f"P1 winner ({p1_winner['label']}) on yolo11s",
            "yolo11s.pt", 100, dict(winner_overrides),
        ),
    ]
    for v in p2_variants:
        rows.append(run_variant(v, project=sweep_dir, phase="P2"))
        write_csv(csv_path, rows)
    print_ranked(rows, "After Phase 2")

    # Phase 3: overall winner at 300 epochs + patience=50
    overall = pick_winner(rows)
    if not overall:
        logger.error("No successful runs to advance to Phase 3 — aborting.")
        return
    # Look up the source Variant (P1 or P2)
    all_variants = {v.label: v for v in P1_VARIANTS + p2_variants}
    src = all_variants[overall["label"]]
    final = Variant(
        "FINAL",
        f"overall winner ({overall['label']}) at 300 ep, patience=50",
        src.model,
        300,
        dict(src.overrides, patience=50),
    )
    rows.append(run_variant(final, project=sweep_dir, phase="P3"))
    write_csv(csv_path, rows)

    print_ranked(rows, "Final ranking")
    logger.success(f"Sweep complete. Results: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print plan without training")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
