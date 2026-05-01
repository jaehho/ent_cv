"""Train a YOLO model."""
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import typer
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import MODELS_DIR, TRAIN_METRIC_KEYS
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


def run(
    data: Path,
    model: str,
    epochs: int = 200,
    batch: int = -1,
    imgsz: int = 1024,
    rect: bool = True,
    scale: float = 0.0,
    device: str = "0",
    project: Path | None = None,
) -> Any:
    """Train a YOLO model. Returns the YOLO results object."""
    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    out_dir = Path(project) if project else MODELS_DIR

    model_p = Path(model)
    if model_p.suffix == ".pt":
        yolo_model_path = str(model_p)
        base_name = model_p.parents[1].name if len(model_p.parts) >= 3 else model_p.stem
    else:
        yolo_model_path = f"{model}.pt"
        base_name = model

    model_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Loading {yolo_model_path}…")
    model_obj = cast(Any, YOLO(yolo_model_path))
    logger.info(f"Dataset:  {data}")
    logger.info(f"Output:   {out_dir / model_name}")

    results = model_obj.train(
        data=str(data),
        project=str(out_dir),
        name=model_name,
        pretrained=False,
        verbose=True,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        rect=rect,
        device=device,
        scale=scale,
    )

    logger.success(f"Training complete — {out_dir / model_name / 'weights' / 'best.pt'}")
    return results


def _results_fn(results: Any):
    rd = getattr(results, "results_dict", {})
    save_dir = Path(str(getattr(results, "save_dir", "")))
    lines = []
    for label, key in TRAIN_METRIC_KEYS:
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


@app.command()
@notify("Training", results_fn=_results_fn)
def main(
    data: Path = typer.Option(..., help="Path to dataset YAML"),
    model: str = typer.Option(..., help="Model name or path to .pt weights"),
    epochs: int = typer.Option(200, help="Number of training epochs"),
    batch: int = typer.Option(-1, help="Batch size (-1 = auto)"),
    imgsz: int = typer.Option(1024, help="Image size"),
    rect: bool = typer.Option(True, help="Rectangular training"),
    scale: float = typer.Option(0.0, help="Scale augmentation"),
    device: str = typer.Option("0", help="Device (e.g. '0', '0,1', 'cpu')"),
):
    """Train a YOLO model."""
    return run(
        data=data, model=model, epochs=epochs, batch=batch,
        imgsz=imgsz, rect=rect, scale=scale, device=device,
    )


if __name__ == "__main__":
    app()
