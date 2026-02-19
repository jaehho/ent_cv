"""Train a YOLO model."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typer
import yaml
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import MODELS_DIR
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


@dataclass
class TrainConfig:
    # required
    dataset_yaml: Path
    model: str
    epochs: int
    batch: int
    imgsz: int
    rect: bool
    scale: float
    device: int

    def __post_init__(self):
        self.dataset_yaml = Path(self.dataset_yaml)


def _load_config(config_file: Path) -> TrainConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in TrainConfig.__dataclass_fields__}
    if unknown := set(d) - set(TrainConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return TrainConfig(**known)


def run(cfg: TrainConfig) -> Any:
    """Train a YOLO model. Returns the YOLO results object."""
    if not cfg.dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {cfg.dataset_yaml}")

    model_p = Path(cfg.model)
    if model_p.suffix == ".pt":
        yolo_model_path = str(model_p)
        base_name = model_p.parents[1].name if len(model_p.parts) >= 3 else model_p.stem
    else:
        yolo_model_path = f"{cfg.model}.pt"
        base_name = cfg.model

    model_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Loading {yolo_model_path}…")
    model_obj = cast(Any, YOLO(yolo_model_path))
    logger.info(f"Dataset:  {cfg.dataset_yaml}")
    logger.info(f"Output:   {MODELS_DIR / model_name}")

    results = model_obj.train(
        data=str(cfg.dataset_yaml),
        epochs=cfg.epochs, batch=cfg.batch, imgsz=cfg.imgsz,
        rect=cfg.rect, device=cfg.device, scale=cfg.scale,
        project=str(MODELS_DIR), name=model_name,
        pretrained=False, verbose=True,
    )

    logger.success(f"Training complete — {MODELS_DIR / model_name / 'weights' / 'best.pt'}")
    return results


_TRAIN_METRIC_KEYS = [
    ("mAP50",     "metrics/mAP50(B)"),
    ("mAP50-95",  "metrics/mAP50-95(B)"),
    ("Precision", "metrics/precision(B)"),
    ("Recall",    "metrics/recall(B)"),
]


def _results_fn(results: Any):
    rd = getattr(results, "results_dict", {})
    save_dir = Path(str(getattr(results, "save_dir", "")))
    lines = []
    for label, key in _TRAIN_METRIC_KEYS:
        val = rd.get(key)
        if val is not None:
            lines.append(f"  {label:<12} {val:.4f}")
    if save_dir.exists():
        lines.append(f"\n  Results dir: {save_dir}")
    attachments = [f for f in [
        save_dir / "results.png",
        save_dir / "confusion_matrix_normalized.png",
        save_dir / "confusion_matrix.png",
        save_dir / "args.yaml",
    ] if f.exists()]
    return "\n".join(lines), attachments


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/train.yaml")


@app.command()
@notify("Training", results_fn=_results_fn)
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Train a YOLO model."""
    return run(_load_config(config_file))


if __name__ == "__main__":
    app()
