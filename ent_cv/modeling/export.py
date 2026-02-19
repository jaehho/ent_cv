"""Export a YOLO model to ONNX."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from ultralytics import YOLO

app = typer.Typer(add_completion=False)


@dataclass
class ExportConfig:
    # required
    weights: Path
    half: bool
    dynamic: bool
    simplify: bool
    device: str
    batch: int
    # optional (None = not set)
    imgsz: Optional[int] = None
    opset: Optional[int] = None

    def __post_init__(self):
        self.weights = Path(self.weights)


def _load_config(config_file: Path) -> ExportConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in ExportConfig.__dataclass_fields__}
    if unknown := set(d) - set(ExportConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return ExportConfig(**known)


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


def run(cfg: ExportConfig) -> Path:
    """Export a YOLO model to ONNX. Returns the path to the exported file."""
    if not cfg.weights.exists():
        raise FileNotFoundError(f"Weights not found: {cfg.weights}")

    imgsz = cfg.imgsz or _read_imgsz(cfg.weights) or 640
    logger.info(f"Loading weights: {cfg.weights}")
    model = YOLO(str(cfg.weights))

    kwargs: dict = dict(
        format="onnx", imgsz=imgsz, half=cfg.half, dynamic=cfg.dynamic,
        simplify=cfg.simplify, device=cfg.device, batch=cfg.batch,
    )
    if cfg.opset is not None:
        kwargs["opset"] = cfg.opset

    logger.info(f"Exporting: {kwargs}")
    exported = model.export(**kwargs)
    export_path = Path(str(exported)) if exported else cfg.weights.with_suffix(".onnx")
    logger.success(f"Export complete: {export_path}")
    return export_path


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/export.yaml")


@app.command()
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Export a YOLO model to ONNX."""
    run(_load_config(config_file))


if __name__ == "__main__":
    app()
