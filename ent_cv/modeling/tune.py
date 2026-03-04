"""Run Ultralytics evolutionary hyperparameter tuning."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import typer
import yaml
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import MODELS_DIR
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


@dataclass
class TuneConfig:
    # required
    data: Path
    model: str
    # optional
    epochs: Optional[int] = None
    iterations: Optional[int] = None
    imgsz: Optional[int] = None
    batch: Optional[int] = None
    device: Optional[int] = None
    optimizer: Optional[str] = None

    def __post_init__(self):
        self.data = Path(self.data)


def _load_config(config_file: Path) -> TuneConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in TuneConfig.__dataclass_fields__}
    if unknown := set(d) - set(TuneConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return TuneConfig(**known)


def run(cfg: TuneConfig) -> None:
    """Run Ultralytics evolutionary hyperparameter tuning."""
    if not cfg.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {cfg.data}")

    model_p = Path(cfg.model)
    yolo_path = str(model_p) if model_p.suffix == ".pt" else f"{cfg.model}.pt"
    tune_dir = MODELS_DIR / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {yolo_path}…")
    model_obj = cast(Any, YOLO(yolo_path))
    logger.info(f"Dataset: {cfg.data} | Iterations: {cfg.iterations} | Epochs/run: {cfg.epochs}")

    model_obj.tune(
        data=str(cfg.data),
        project=str(tune_dir), name=cfg.model,
        plots=True, save=True, val=True,
        **{k: v for k, v in {
            "epochs": cfg.epochs, "iterations": cfg.iterations,
            "imgsz": cfg.imgsz, "batch": cfg.batch,
            "device": cfg.device, "optimizer": cfg.optimizer,
        }.items() if v is not None},
    )

    best = tune_dir / cfg.model / "best_hyperparameters.yaml"
    logger.success(f"Tuning complete — best: {best}")


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/tune.yaml")


@app.command()
@notify("Tuning")
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Run Ultralytics evolutionary hyperparameter tuning."""
    return run(_load_config(config_file))


if __name__ == "__main__":
    app()
