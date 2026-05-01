"""Benchmark a YOLO model across export formats."""
from pathlib import Path
from typing import Any, Optional, cast

from loguru import logger
import typer
from ultralytics import YOLO

from ent_cv.modeling._utils import read_imgsz
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


def run(
    weights: Path,
    data: Path,
    device: str = "0",
    half: bool = False,
    int8: bool = False,
    verbose: bool = False,
    imgsz: Optional[int] = None,
    format: Optional[str] = None,
) -> Any:
    """Benchmark a YOLO model across export formats. Returns a DataFrame of results."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    resolved_imgsz = imgsz or read_imgsz(weights) or 640

    logger.info(f"Loading weights: {weights}")
    model = cast(Any, YOLO(str(weights), task="detect"))

    kwargs: dict[str, Any] = {
        "data": str(data),
        "device": device,
        "half": half,
        "int8": int8,
        "verbose": verbose,
        "imgsz": resolved_imgsz,
    }
    if format is not None:
        kwargs["format"] = format

    logger.info(
        f"Benchmarking — imgsz={resolved_imgsz}  half={half}  int8={int8}  "
        f"device={device}  format={format}  verbose={verbose}"
    )
    df = model.benchmark(**kwargs)
    logger.success("Benchmark complete.")
    return df


def _results_fn(df: Any):
    if df is None:
        return "", []
    try:
        return f"\n{df.to_string()}", []
    except Exception:
        return str(df), []


@app.command()
@notify("Benchmark", results_fn=_results_fn)
def main(
    weights: Path = typer.Option(..., help="Path to model weights (.pt)"),
    data: Path = typer.Option(..., help="Path to dataset YAML"),
    device: str = typer.Option("0", help="Device (e.g. '0', 'cpu')"),
    half: bool = typer.Option(False, help="Use FP16 half precision"),
    int8: bool = typer.Option(False, help="Use INT8 quantization"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    imgsz: Optional[int] = typer.Option(None, help="Image size (auto-read from args.yaml if omitted)"),
    format: Optional[str] = typer.Option(None, help="Export format to benchmark (default: all)"),
) -> Any:
    """Benchmark a YOLO model across export formats."""
    return run(
        weights=weights,
        data=data,
        device=device,
        half=half,
        int8=int8,
        verbose=verbose,
        imgsz=imgsz,
        format=format,
    )


if __name__ == "__main__":
    app()
