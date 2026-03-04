"""Run YOLO inference on a single source."""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from ultralytics import YOLO

from ent_cv.config import PREDICTIONS_DIR
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


@dataclass
class PredictConfig:
    # required
    source: Path
    weights: Path
    conf: float
    iou: float
    imgsz: int
    device: str
    onnx: bool
    overwrite: bool
    save: bool
    save_conf: bool
    save_txt: bool
    save_json: bool
    save_frames: bool
    verbose: bool = True
    # optional (None = not set)
    output_dir: Optional[Path] = None
    suffix: Optional[str] = None

    def __post_init__(self):
        self.source = Path(self.source)
        self.weights = Path(self.weights)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


def _load_config(config_file: Path) -> PredictConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in PredictConfig.__dataclass_fields__}
    if unknown := set(d) - set(PredictConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return PredictConfig(**known)


def run(cfg: PredictConfig) -> Optional[tuple[Path, int]]:
    """Run YOLO inference. Returns (output_dir, frame_count) or None if skipped."""
    weights = cfg.weights.parent / "best.onnx" if cfg.onnx else cfg.weights
    if cfg.onnx:
        logger.info(f"ONNX mode: using {weights}")

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not cfg.source.exists():
        raise FileNotFoundError(f"Source not found: {cfg.source}")

    derived_name = cfg.source.stem + (f"_{cfg.suffix}" if cfg.suffix else "")
    output_dir = cfg.output_dir or (PREDICTIONS_DIR / derived_name)

    if output_dir.exists() and any(output_dir.iterdir()):
        if cfg.overwrite:
            shutil.rmtree(output_dir)
            logger.info(f"Removed existing output: {output_dir}")
        else:
            print(f"\nOutput already exists: {output_dir}")
            print("  [d] Delete and overwrite")
            print("  [s] Skip")
            print("  [a] Abort")
            while True:
                choice = input("Choice [d/s/a]: ").strip().lower()
                if choice == "d":
                    shutil.rmtree(output_dir)
                    logger.info(f"Removed existing output: {output_dir}")
                    break
                elif choice == "s":
                    logger.info(f"Skipping — output already exists: {output_dir}")
                    return None
                elif choice == "a":
                    raise SystemExit("Aborted by user.")
                else:
                    print("  Please enter d, s, or a.")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading weights: {weights}")
    model = YOLO(str(weights), task="detect")

    logger.info(f"Source: {cfg.source}  conf={cfg.conf}  iou={cfg.iou}")
    results_gen = model.predict(
        source=str(cfg.source),
        conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, device=cfg.device,
        save=cfg.save, save_conf=cfg.save_conf, save_txt=cfg.save_txt,
        save_frames=cfg.save_frames,
        project=str(PREDICTIONS_DIR), name=derived_name,
        exist_ok=True, stream=True, verbose=cfg.verbose,
    )

    all_frames = []
    n = 0

    for result in results_gen:
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                })

        all_frames.append({
            "frame": n,
            "source": str(result.path),
            "detections": detections,
        })
        n += 1

    if cfg.save_json and all_frames:
        with open(output_dir / "detections.json", "w") as fh:
            json.dump(all_frames, fh, indent=2)

    logger.success(f"Done — {n} frame(s) processed, output: {output_dir}")

    return output_dir, n


def _results_fn(result):
    if result is None:
        return "", []
    output_dir, n = result
    return f"  Frames: {n}\n  Output: {output_dir}", []


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/predict.yaml")


@app.command()
@notify("Prediction", results_fn=_results_fn)
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Run YOLO inference on a single source."""
    cfg = _load_config(config_file)
    out = cfg.output_dir or (PREDICTIONS_DIR / (cfg.source.stem + (f"_{cfg.suffix}" if cfg.suffix else "")))
    if out.exists() and any(out.iterdir()):
        if not typer.confirm(f"Output exists: {out}\nDelete and continue?", default=False):
            logger.info("Aborted.")
            return None
        cfg.overwrite = True
    return run(cfg)


if __name__ == "__main__":
    app()
