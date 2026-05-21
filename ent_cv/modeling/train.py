"""Train a YOLO model."""
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from loguru import logger
import typer
from ultralytics import YOLO
import yaml

from ent_cv.config import MODELS_DIR, PRETRAINED_DIR, TRAIN_METRIC_KEYS
from ent_cv.gpu import DEFAULT_TRAIN_MIN_FREE_GIB, preflight_gpu, restore_services
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


def _validate_data_yaml(data_yaml: Path) -> None:
    """Trip early if the dataset YAML uses relative paths.

    CVAT's "Ultralytics YOLO Detection 1.0" export writes `path: .` plus
    a relative `train.txt` ("images/train/foo.png"). Ultralytics resolves
    those lines against cwd, so training only works when cwd is the
    dataset dir — and produces a cryptic "No valid images found" otherwise.
    `ent-cv prepare-dataset` rewrites everything to absolute; surface that
    requirement here instead of failing deep inside YOLO.
    """
    cfg = yaml.safe_load(data_yaml.read_text()) or {}
    hint = f"Run: ent-cv prepare-dataset {data_yaml.parent}"

    base = cfg.get("path")
    if not base or not Path(base).is_absolute():
        raise ValueError(f"Dataset YAML '{data_yaml}' has non-absolute path={base!r}. {hint}")

    for key in ("train", "val"):
        rel = cfg.get(key)
        if not rel:
            continue
        list_file = Path(rel) if Path(rel).is_absolute() else Path(base) / rel
        if not list_file.exists() or list_file.suffix != ".txt":
            continue
        for line in list_file.read_text().splitlines():
            entry = line.strip()
            if not entry:
                continue
            if not Path(entry).is_absolute():
                raise ValueError(
                    f"Split file '{list_file}' has relative entries (first: {entry!r}). {hint}"
                )
            break


def run(
    data: Path,
    model: str,
    epochs: int = 200,
    batch: int = -1,
    imgsz: int = 1024,
    rect: bool = True,
    scale: float = 0.5,
    device: str = "0",
    project: Path | None = None,
    *,
    # Sweep-extensible hparams (defaults match Ultralytics defaults).
    close_mosaic: int = 10,
    cos_lr: bool = False,
    cls: float = 0.5,
    mixup: float = 0.0,
    multi_scale: float = 0.0,
    weight_decay: float = 0.0005,
    freeze: int | None = None,
    lr0: float = 0.01,
    patience: int = 100,
    degrees: float = 0.0,
    perspective: float = 0.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    optimizer: str = "auto",
    name_suffix: str = "",
    wait_gpu: bool = False,
) -> Any:
    """Train a YOLO model. Returns the YOLO results object."""
    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    _validate_data_yaml(data)
    paused_services = preflight_gpu(
        device, min_free_gib=DEFAULT_TRAIN_MIN_FREE_GIB, wait=wait_gpu,
    )

    out_dir = Path(project) if project else MODELS_DIR

    model_p = Path(model)
    if model_p.suffix == ".pt":
        yolo_model_path = str(model_p)
        base_name = model_p.parents[1].name if len(model_p.parts) >= 3 else model_p.stem
    else:
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        yolo_model_path = str(PRETRAINED_DIR / f"{model}.pt")
        base_name = model

    suffix = f"_{name_suffix}" if name_suffix else ""
    model_name = f"{base_name}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Loading {yolo_model_path}…")
    model_obj = cast(Any, YOLO(yolo_model_path))
    logger.info(f"Dataset:  {data}")
    logger.info(f"Output:   {out_dir / model_name}")

    train_kwargs: dict[str, Any] = dict(
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
        close_mosaic=close_mosaic,
        cos_lr=cos_lr,
        cls=cls,
        mixup=mixup,
        multi_scale=multi_scale,
        weight_decay=weight_decay,
        lr0=lr0,
        patience=patience,
        degrees=degrees,
        perspective=perspective,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        optimizer=optimizer,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    try:
        results = model_obj.train(**train_kwargs)
    finally:
        restore_services(paused_services)

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
    scale: float = typer.Option(0.5, help="Scale augmentation"),
    device: str = typer.Option("0", help="Device (e.g. '0', '0,1', 'cpu')"),
    wait_gpu: bool = typer.Option(
        False, "--wait-gpu/--no-wait-gpu",
        help="If GPU memory is too low, wait for it to free up (up to 1h) instead of failing immediately.",
    ),
):
    """Train a YOLO model."""
    return run(
        data=data, model=model, epochs=epochs, batch=batch,
        imgsz=imgsz, rect=rect, scale=scale, device=device,
        wait_gpu=wait_gpu,
    )


if __name__ == "__main__":
    app()
