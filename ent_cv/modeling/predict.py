"""Run YOLO inference on a single source."""
import json
from pathlib import Path
import shutil
import subprocess
from typing import Optional

from loguru import logger
import polars as pl
import typer
from ultralytics import YOLO

from ent_cv.config import PREDICTIONS_DIR
from ent_cv.gpu import gpu_yield
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".ts", ".flv"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _get_fps(source: Path) -> Optional[str]:
    """Return FPS as a native fraction string (e.g. '30000/1001') via ffprobe.
    Returns None for images or if the source contains no video stream."""
    def _fps_from_file(path: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True, text=True, check=True,
            )
            value = result.stdout.strip()
            return value if value else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    if source.is_file():
        if source.suffix.lower() in _VIDEO_SUFFIXES:
            return _fps_from_file(source)
        return None  # single image / frame
    if source.is_dir():
        for candidate in sorted(source.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in _VIDEO_SUFFIXES:
                return _fps_from_file(candidate)
        return None  # directory of images
    return None


def run(
    source: Path,
    weights: Path,
    conf: float = 0.647,
    iou: float = 0.7,
    imgsz: int = 1024,
    device: str = "0",
    overwrite: bool = False,
    save: bool = False,
    save_conf: bool = True,
    save_txt: bool = False,
    save_json: bool = True,
    save_frames: bool = False,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    suffix: Optional[str] = None,
) -> Optional[tuple[Path, int]]:
    """Run YOLO inference. Returns (output_dir, frame_count) or None if skipped."""
    source, weights = Path(source), Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    derived_name = source.stem + (f"_{suffix}" if suffix else "")
    out_dir = output_dir or (PREDICTIONS_DIR / derived_name)

    if out_dir.exists() and any(out_dir.iterdir()):
        if overwrite:
            shutil.rmtree(out_dir)
            logger.info(f"Removed existing output: {out_dir}")
        else:
            print(f"\nOutput already exists: {out_dir}")
            print("  [d] Delete and overwrite")
            print("  [s] Skip")
            print("  [a] Abort")
            while True:
                choice = input("Choice [d/s/a]: ").strip().lower()
                if choice == "d":
                    shutil.rmtree(out_dir)
                    logger.info(f"Removed existing output: {out_dir}")
                    break
                elif choice == "s":
                    logger.info(f"Skipping — output already exists: {out_dir}")
                    return None
                elif choice == "a":
                    raise SystemExit("Aborted by user.")
                else:
                    print("  Please enter d, s, or a.")

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading weights: {weights}")
    model = YOLO(str(weights), task="detect")

    logger.info(f"Source: {source}  conf={conf}  iou={iou}")
    frames_list: list[pl.DataFrame] = []
    n = 0
    orig_shape: Optional[tuple[int, int]] = None  # (height, width) from first result
    with gpu_yield(device):
        results_gen = model.predict(
            source=str(source),
            project=str(PREDICTIONS_DIR),
            name=derived_name,
            exist_ok=True,
            stream=True,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=save,
            save_conf=save_conf,
            save_txt=save_txt,
            save_frames=save_frames,
            verbose=verbose,
        )
        for result in results_gen:
            if orig_shape is None and getattr(result, "orig_shape", None) is not None:
                orig_shape = tuple(int(v) for v in result.orig_shape[:2])  # (h, w)
            df = result.to_df()
            if not df.is_empty():
                df = df.with_columns(pl.lit(n).alias("frame"))
                frames_list.append(df)
            n += 1

    if save_json and frames_list:
        all_df = pl.concat(frames_list)
        all_df.write_json(out_dir / "detections.json")

    # Write sidecar metadata
    fps = _get_fps(source)
    metadata: dict = {
        "total_frames": n,
        "source": str(source),
    }
    if fps is not None:
        metadata["fps"] = fps
    if orig_shape is not None:
        metadata["height"], metadata["width"] = orig_shape
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.success(f"Done — {n} frame(s) processed, output: {out_dir}")
    return out_dir, n


def _results_fn(result):
    if result is None:
        return "", []
    output_dir, n = result
    return f"  Frames: {n}\n  Output: {output_dir}", []


@app.command()
@notify("Prediction", results_fn=_results_fn)
def main(
    source: Path = typer.Option(..., help="Path to video file or image directory"),
    weights: Path = typer.Option(..., help="Path to model weights"),
    conf: float = typer.Option(0.647, help="Confidence threshold"),
    iou: float = typer.Option(0.7, help="IoU threshold for NMS"),
    imgsz: int = typer.Option(1024, help="Inference image size"),
    device: str = typer.Option("0", help="Device (e.g. '0', 'cpu')"),
    overwrite: bool = typer.Option(False, help="Overwrite existing output without prompting"),
    save: bool = typer.Option(False, help="Save prediction images/video"),
    save_conf: bool = typer.Option(True, help="Save confidences in txt labels"),
    save_txt: bool = typer.Option(False, help="Save results as .txt"),
    save_json: bool = typer.Option(True, help="Save detections.json (Polars)"),
    save_frames: bool = typer.Option(False, help="Save individual frames"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    output_dir: Optional[Path] = typer.Option(None, help="Custom output directory"),
    suffix: Optional[str] = typer.Option(None, help="Suffix appended to output dir name"),
):
    """Run YOLO inference on a single source."""
    return run(
        source=source, weights=weights, conf=conf, iou=iou,
        imgsz=imgsz, device=device, overwrite=overwrite, save=save,
        save_conf=save_conf, save_txt=save_txt, save_json=save_json,
        save_frames=save_frames, verbose=verbose, output_dir=output_dir,
        suffix=suffix,
    )


if __name__ == "__main__":
    app()
