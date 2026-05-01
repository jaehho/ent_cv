"""Shared helpers for modeling subcommands."""

from pathlib import Path

import yaml
from loguru import logger


def read_imgsz(weights: Path) -> int | None:
    """Read the ``imgsz`` value from the ``args.yaml`` sidecar next to *weights*."""
    args_yaml = weights.parent.parent / "args.yaml"
    if not args_yaml.exists():
        return None
    try:
        with args_yaml.open() as f:
            val = yaml.safe_load(f).get("imgsz")
        return int(val) if val is not None else None
    except Exception as exc:
        logger.warning(f"Could not read {args_yaml}: {exc}")
        return None
