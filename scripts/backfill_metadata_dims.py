"""Backfill width/height into older predictions metadata.json files.

For each ``PREDICTIONS_DIR/*/metadata.json`` that lacks width/height, ffprobe
the source video and write the dimensions in place. Idempotent — directories
that already have both fields are skipped.

Run: uv run python scripts/backfill_metadata_dims.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from ent_cv.config import PREDICTIONS_DIR
from ent_cv.modeling.postprocess import _ffprobe_dims


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _probe_source(source: str) -> tuple[int, int] | None:
    src_path = Path(source)
    if src_path.is_file():
        return _ffprobe_dims(src_path)
    if src_path.is_dir():
        for candidate in sorted(src_path.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() == ".mp4":
                dims = _ffprobe_dims(candidate)
                if dims is not None:
                    return dims
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions-dir", type=Path, default=PREDICTIONS_DIR,
        help=f"Root predictions directory (default: {PREDICTIONS_DIR})",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report only; don't write")
    args = ap.parse_args()

    root: Path = args.predictions_dir
    if not root.exists():
        logger.error(f"Predictions dir not found: {root}")
        raise SystemExit(1)

    stats = {"already_set": 0, "filled": 0, "no_source": 0, "probe_failed": 0}
    for meta_path in sorted(root.glob("*/metadata.json")):
        case = meta_path.parent.name
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning(f"[{case}] invalid metadata.json: {e}")
            continue

        w, h = meta.get("width"), meta.get("height")
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            stats["already_set"] += 1
            continue

        source = meta.get("source")
        if not source:
            logger.warning(f"[{case}] no 'source' field — cannot probe")
            stats["no_source"] += 1
            continue

        dims = _probe_source(source)
        if dims is None:
            logger.warning(f"[{case}] ffprobe failed for source: {source}")
            stats["probe_failed"] += 1
            continue

        meta["width"], meta["height"] = dims
        if args.dry_run:
            logger.info(f"[{case}] would set width={dims[0]} height={dims[1]}")
        else:
            _write_atomic(meta_path, meta)
            logger.success(f"[{case}] set width={dims[0]} height={dims[1]}")
        stats["filled"] += 1

    logger.info(
        f"Done. already_set={stats['already_set']} filled={stats['filled']} "
        f"no_source={stats['no_source']} probe_failed={stats['probe_failed']}"
    )


if __name__ == "__main__":
    main()
