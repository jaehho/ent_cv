"""Pseudo-label every frame in datasets/current/ with the patient_v0 detector.

The bootstrap detector was trained on 4 cases and validated on 1; this step
takes its best.pt and runs inference over every labeled frame in the corpus
so we can:
  1. Spot-check via FiftyOne (does the detector generalize across cases?).
  2. Merge class-12 (Patient) lines into the existing per-frame .txt files
     so the next multi-class retrain has Patient supervision everywhere.

Output is YOLO .txt with the canonical class index for Patient (12) — that
keeps a single index space (config.LABELS) end-to-end, so the merge step is
just `cat existing.txt pseudo.txt > merged.txt`.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image
from tqdm import tqdm
import typer
from ultralytics import YOLO

from ent_cv.config import DATA_DIR, DATASETS_DIR, LABELS
from ent_cv.gpu import gpu_yield

app = typer.Typer(add_completion=False)

DEFAULT_WEIGHTS = (
    DATA_DIR
    / "models"
    / "patient_v0_sweep_20260515_145757"
    / "yolo11s_patient_v0_img640_lr0.001_20260515_145757"
    / "weights"
    / "best.pt"
)
DEFAULT_IMAGES = DATASETS_DIR / "current" / "images" / "train"
DEFAULT_OUT = DATA_DIR / "predictions" / "patient_v0_pseudo"
PATIENT_CLASS = LABELS.index("Patient")


@app.command()
def main(
    weights: Path = typer.Option(DEFAULT_WEIGHTS, help="Path to patient_v0 best.pt."),
    images_dir: Path = typer.Option(DEFAULT_IMAGES, help="Source image directory."),
    out_dir: Path = typer.Option(DEFAULT_OUT, help="Output directory."),
    conf: float = typer.Option(0.5, help="Confidence threshold."),
    iou: float = typer.Option(0.7, help="NMS IoU threshold."),
    imgsz: int = typer.Option(640, help="Inference image size (match training)."),
    device: str = typer.Option("0", help="Device."),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
):
    """Run inference and save canonical class-12 (Patient) YOLO .txt files."""
    if not weights.exists():
        raise typer.BadParameter(f"weights not found: {weights}")
    if not images_dir.is_dir():
        raise typer.BadParameter(f"images dir not found: {images_dir}")

    images = sorted(images_dir.glob("*.png"))
    if not images:
        raise typer.BadParameter(f"no .png files under {images_dir}")
    logger.info(f"Found {len(images)} images")

    labels_out = out_dir / "labels"
    if labels_out.exists() and any(labels_out.iterdir()):
        if not overwrite:
            raise typer.BadParameter(
                f"{labels_out} is not empty; pass --overwrite to clear it."
            )
        for p in labels_out.glob("*.txt"):
            p.unlink()
    labels_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading weights: {weights}")
    model = YOLO(str(weights), task="detect")

    n_frames_with_boxes = 0
    n_frames_empty = 0
    n_boxes_total = 0
    box_count_hist: dict[int, int] = {}

    logger.info(f"Inference: conf={conf}  iou={iou}  imgsz={imgsz}")
    with gpu_yield(device):
        results = model.predict(
            source=str(images_dir),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            batch=1,
            stream=True,
            verbose=False,
            save=False,
        )

        for result in tqdm(results, total=len(images), desc="pseudo-label"):
            img_path = Path(result.path)
            boxes = result.boxes
            n = 0 if boxes is None else int(boxes.shape[0])
            box_count_hist[n] = box_count_hist.get(n, 0) + 1
            if n == 0:
                n_frames_empty += 1
                continue
            n_frames_with_boxes += 1
            n_boxes_total += n

            # Image dims for normalization
            with Image.open(img_path) as im:
                w, h = im.size

            # YOLO box.xywh is in pixels; convert to normalized cxcywh
            lines: list[str] = []
            xywh = boxes.xywh.cpu().numpy()
            for cx, cy, bw, bh in xywh:
                lines.append(
                    f"{PATIENT_CLASS} "
                    f"{cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
            (labels_out / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")

    summary_lines = [
        f"Frames processed: {len(images)}",
        f"  with ≥1 Patient box: {n_frames_with_boxes}",
        f"  no boxes (below conf): {n_frames_empty}",
        f"Total Patient boxes: {n_boxes_total}",
        "Boxes-per-frame histogram:",
    ]
    for k in sorted(box_count_hist):
        summary_lines.append(f"  {k}: {box_count_hist[k]}")
    summary = "\n".join(summary_lines)
    logger.info("\n" + summary)

    (out_dir / "summary.txt").write_text(summary + "\n")
    logger.success(f"Wrote pseudo-labels to {labels_out}/")


if __name__ == "__main__":
    app()
