"""Validate a YOLO model on a dataset split."""
from pathlib import Path
from typing import Any, Optional, cast

from loguru import logger
import typer
from ultralytics import YOLO

from ent_cv.config import PREDICTIONS_DIR, VAL_METRIC_KEYS
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


def run(
    weights: Path,
    data: Path,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: str = "0",
    half: bool = False,
    split: str = "val",
    save_json: bool = False,
    verbose: bool = False,
    imgsz: Optional[int] = None,
    batch: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    """Validate a YOLO model. Returns the metrics object."""
    weights, data = Path(weights), Path(data)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    derived_name = name or weights.parent.parent.name

    logger.info(f"Loading weights: {weights}")
    model = cast(Any, YOLO(str(weights), task="detect"))

    kwargs: dict[str, Any] = {
        "data": str(data),
        "device": device,
        "half": half,
        "split": split,
        "save_json": save_json,
        "verbose": verbose,
        "project": str(PREDICTIONS_DIR),
        "name": derived_name,
    }
    if conf is not None:
        kwargs["conf"] = conf
    if iou is not None:
        kwargs["iou"] = iou
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if batch is not None:
        kwargs["batch"] = batch

    logger.info(
        f"Validating — split={split}  conf={conf}  iou={iou}  "
        f"imgsz={imgsz}  half={half}"
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
    for label, key in VAL_METRIC_KEYS:
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


@app.command()
@notify("Validation", results_fn=_results_fn)
def main(
    weights: Path = typer.Option(..., help="Path to model weights (.pt)"),
    data: Path = typer.Option(..., help="Path to dataset YAML"),
    conf: Optional[float] = typer.Option(None, help="Confidence threshold"),
    iou: Optional[float] = typer.Option(None, help="IoU threshold for NMS"),
    device: str = typer.Option("0", help="Device (e.g. '0', 'cpu')"),
    half: bool = typer.Option(False, help="Use FP16 half precision"),
    split: str = typer.Option("val", help="Dataset split: val, test, or train"),
    save_json: bool = typer.Option(False, help="Save results as COCO JSON"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    imgsz: Optional[int] = typer.Option(None, help="Image size"),
    batch: Optional[int] = typer.Option(None, help="Batch size"),
    name: Optional[str] = typer.Option(None, help="Run name (default: derive from weights path)"),
) -> Any:
    """Validate a YOLO model on a dataset split."""
    return run(
        weights=weights,
        data=data,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        split=split,
        save_json=save_json,
        verbose=verbose,
        imgsz=imgsz,
        batch=batch,
        name=name,
    )


if __name__ == "__main__":
    app()
