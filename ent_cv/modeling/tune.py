"""Run Ultralytics evolutionary hyperparameter tuning."""
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import typer
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import MODELS_DIR
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


def run(
    data: Path,
    model: str,
    epochs: int = 30,
    iterations: int = 300,
    imgsz: int = 1024,
    batch: int = -1,
    device: str = "0",
    optimizer: Optional[str] = None,
) -> None:
    """Run Ultralytics evolutionary hyperparameter tuning."""
    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    model_p = Path(model)
    yolo_path = str(model_p) if model_p.suffix == ".pt" else f"{model}.pt"
    tune_dir = MODELS_DIR / f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tune_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {yolo_path}…")
    model_obj = cast(Any, YOLO(yolo_path))
    logger.info(f"Dataset: {data} | Iterations: {iterations} | Epochs/run: {epochs}")

    kwargs: dict[str, Any] = {
        "data": str(data),
        "project": str(tune_dir),
        "name": model,
        "plots": True,
        "save": True,
        "val": True,
        "epochs": epochs,
        "iterations": iterations,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
    }
    if optimizer is not None:
        kwargs["optimizer"] = optimizer

    model_obj.tune(**kwargs)

    best = tune_dir / model / "best_hyperparameters.yaml"
    logger.success(f"Tuning complete — best: {best}")


@app.command()
@notify("Tuning")
def main(
    data: Path = typer.Option(..., help="Path to dataset YAML"),
    model: str = typer.Option(..., help="Model name or .pt path"),
    epochs: int = typer.Option(30, help="Epochs per tuning iteration"),
    iterations: int = typer.Option(300, help="Number of tuning iterations"),
    imgsz: int = typer.Option(1024, help="Image size"),
    batch: int = typer.Option(-1, help="Batch size (-1 = auto)"),
    device: str = typer.Option("0", help="Device (e.g. '0', 'cpu')"),
    optimizer: Optional[str] = typer.Option(None, help="Optimizer override"),
) -> None:
    """Run Ultralytics evolutionary hyperparameter tuning."""
    run(
        data=data,
        model=model,
        epochs=epochs,
        iterations=iterations,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    app()
