"""Validate a YOLO model on a dataset split."""
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import typer
import yaml
from loguru import logger
from ultralytics import YOLO

from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


@dataclass
class ValConfig:
    # required
    weights: Path
    data: Path
    # optional
    conf: Optional[float] = None
    iou: Optional[float] = None
    device: Optional[str] = None
    half: Optional[bool] = None
    split: Optional[str] = None  # "val", "test", or "train"
    save_json: Optional[bool] = None
    verbose: Optional[bool] = None
    imgsz: Optional[int] = None
    batch: Optional[int] = None

    def __post_init__(self):
        self.weights = Path(self.weights)
        self.data = Path(self.data)


def _load_config(config_file: Path) -> ValConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in ValConfig.__dataclass_fields__}
    if unknown := set(d) - set(ValConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return ValConfig(**known)


def _read_imgsz(weights: Path) -> Optional[int]:
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


_VAL_METRIC_KEYS = [
    ("mAP50",     "metrics/mAP50(B)"),
    ("mAP50-95",  "metrics/mAP50-95(B)"),
    ("Precision", "metrics/precision(B)"),
    ("Recall",    "metrics/recall(B)"),
]


def run(cfg: ValConfig) -> Any:
    """Validate a YOLO model. Returns the metrics object."""
    if not cfg.weights.exists():
        raise FileNotFoundError(f"Weights not found: {cfg.weights}")
    if not cfg.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {cfg.data}")

    imgsz = cfg.imgsz or _read_imgsz(cfg.weights) or 640

    logger.info(f"Loading weights: {cfg.weights}")
    model = cast(Any, YOLO(str(cfg.weights), task="detect"))

    kwargs = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in dataclasses.asdict(cfg).items()
        if k != "weights" and v is not None
    }
    kwargs["imgsz"] = imgsz  # use resolved value

    logger.info(
        f"Validating — split={cfg.split}  conf={cfg.conf}  iou={cfg.iou}  "
        f"imgsz={imgsz}  half={cfg.half}"
    )
    metrics = model.val(**kwargs)
    logger.success("Validation complete.")
    return metrics


def _results_fn(metrics: Any):
    if metrics is None:
        return "", []
    rd = getattr(metrics, "results_dict", {})
    save_dir = Path(str(getattr(metrics, "save_dir", "")))
    lines = []
    for label, key in _VAL_METRIC_KEYS:
        val = rd.get(key)
        if val is not None:
            lines.append(f"  {label:<12} {val:.4f}")
    if save_dir.exists():
        lines.append(f"\n  Results dir: {save_dir}")
    attachments = [f for f in [
        save_dir / "confusion_matrix_normalized.png",
        save_dir / "confusion_matrix.png",
        save_dir / "PR_curve.png",
        save_dir / "F1_curve.png",
        save_dir / "P_curve.png",
        save_dir / "R_curve.png",
    ] if f.exists()]
    return "\n".join(lines), attachments


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/val.yaml")


@app.command()
@notify("Validation", results_fn=_results_fn)
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Validate a YOLO model on a dataset split."""
    return run(_load_config(config_file))


if __name__ == "__main__":
    app()
