"""Export a YOLO model to various formats."""
from pathlib import Path
from typing import Any, Optional

from loguru import logger
import typer
from ultralytics import YOLO

from ent_cv.modeling._utils import read_imgsz

app = typer.Typer(add_completion=False)


def run(
    weights: Path,
    format: str = "engine",
    device: str = "0",
    imgsz: Optional[int] = None,
    half: Optional[bool] = None,
    int8: Optional[bool] = None,
    dynamic: Optional[bool] = None,
    simplify: Optional[bool] = None,
    optimize: Optional[bool] = None,
    keras: Optional[bool] = None,
    nms: Optional[bool] = None,
    end2end: Optional[bool] = None,
    opset: Optional[int] = None,
    workspace: Optional[float] = None,
    data: Optional[str] = None,
    fraction: Optional[float] = None,
    batch: Optional[int] = None,
) -> Path:
    """Export a YOLO model to the specified format. Returns the path to the exported file."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    resolved_imgsz = imgsz or read_imgsz(weights) or 640
    logger.info(f"Loading weights: {weights}")
    model = YOLO(str(weights))

    kwargs: dict[str, Any] = {
        "format": format,
        "device": device,
        "imgsz": resolved_imgsz,
    }
    # Only pass rare options when explicitly set
    for name, val in [
        ("half", half), ("int8", int8), ("dynamic", dynamic),
        ("simplify", simplify), ("optimize", optimize), ("keras", keras),
        ("nms", nms), ("end2end", end2end), ("opset", opset),
        ("workspace", workspace), ("data", data), ("fraction", fraction),
        ("batch", batch),
    ]:
        if val is not None:
            kwargs[name] = val

    logger.info(f"Exporting: {kwargs}")
    exported = model.export(**kwargs)
    export_path = Path(str(exported)) if exported else weights.with_suffix(f".{format}")
    logger.success(f"Export complete: {export_path}")
    return export_path


@app.command()
def main(
    weights: Path = typer.Option(..., help="Path to model weights (.pt)"),
    format: str = typer.Option("engine", help="Export format (engine, onnx, torchscript, …)"),
    device: str = typer.Option("0", help="Device (e.g. '0', 'cpu')"),
    imgsz: Optional[int] = typer.Option(None, help="Image size (auto-read from args.yaml if omitted)"),
    half: Optional[bool] = typer.Option(None, help="FP16 half precision"),
    int8: Optional[bool] = typer.Option(None, help="INT8 quantization"),
    dynamic: Optional[bool] = typer.Option(None, help="Dynamic axes (ONNX)"),
    simplify: Optional[bool] = typer.Option(None, help="Simplify ONNX graph"),
    optimize: Optional[bool] = typer.Option(None, help="Optimize for mobile (TorchScript)"),
    keras: Optional[bool] = typer.Option(None, help="Use Keras for TF export"),
    nms: Optional[bool] = typer.Option(None, help="Add NMS to exported model"),
    end2end: Optional[bool] = typer.Option(None, help="End-to-end detection"),
    opset: Optional[int] = typer.Option(None, help="ONNX opset version"),
    workspace: Optional[float] = typer.Option(None, help="TensorRT workspace size (GB)"),
    data: Optional[str] = typer.Option(None, help="Dataset YAML for INT8 calibration"),
    fraction: Optional[float] = typer.Option(None, help="Fraction of dataset for calibration"),
    batch: Optional[int] = typer.Option(None, help="Batch size for export"),
) -> None:
    """Export a YOLO model to the specified format."""
    run(
        weights=weights,
        format=format,
        device=device,
        imgsz=imgsz,
        half=half,
        int8=int8,
        dynamic=dynamic,
        simplify=simplify,
        optimize=optimize,
        keras=keras,
        nms=nms,
        end2end=end2end,
        opset=opset,
        workspace=workspace,
        data=data,
        fraction=fraction,
        batch=batch,
    )


if __name__ == "__main__":
    app()
