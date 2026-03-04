"""Benchmark a YOLO model across export formats."""
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
class BenchmarkConfig:
    # required
    weights: Path
    data: Path
    # optional
    device: Optional[str] = None
    half: Optional[bool] = None
    int8: Optional[bool] = None
    verbose: Optional[bool] = None
    imgsz: Optional[int] = None
    format: Optional[str] = None

    def __post_init__(self):
        self.weights = Path(self.weights)
        if self.data is not None:
            self.data = Path(self.data)


def _load_config(config_file: Path) -> BenchmarkConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in BenchmarkConfig.__dataclass_fields__}
    if unknown := set(d) - set(BenchmarkConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return BenchmarkConfig(**known)


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


def run(cfg: BenchmarkConfig) -> Any:
    """Benchmark a YOLO model across export formats. Returns a DataFrame of results."""
    if not cfg.weights.exists():
        raise FileNotFoundError(f"Weights not found: {cfg.weights}")

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
        f"Benchmarking — imgsz={imgsz}  half={cfg.half}  int8={cfg.int8}  "
        f"device={cfg.device}  format={cfg.format}  verbose={cfg.verbose}"
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


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/benchmark.yaml")


@app.command()
@notify("Benchmark", results_fn=_results_fn)
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Benchmark a YOLO model across export formats."""
    return run(_load_config(config_file))


if __name__ == "__main__":
    app()
