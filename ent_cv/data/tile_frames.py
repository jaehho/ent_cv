"""Tile a YOLO dataset into fixed-size overlapping patches."""

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import math
import random

import cv2
import typer
from loguru import logger

from ent_cv.config import DATASETS_DIR
from ent_cv.utils import notify


app = typer.Typer()


@dataclass(frozen=True)
class YoloBox:
    cls: int
    xc: float
    yc: float
    w: float
    h: float


def read_yolo_labels(label_path: Path) -> List[YoloBox]:
    if not label_path.exists():
        return []

    boxes: List[YoloBox] = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        boxes.append(YoloBox(int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def yolo_to_xyxy(box: YoloBox, W: int, H: int) -> Tuple[int, int, int, int]:
    xc = box.xc * W
    yc = box.yc * H
    bw = box.w * W
    bh = box.h * H
    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))
    return x1, y1, x2, y2


def area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


@app.command()
@notify("Tiling Dataset")
def main(
    dataset_dir: Path = DATASETS_DIR / "dataset_batch2",
    output_dir: Path = DATASETS_DIR / "dataset_batch2_tiled",
    tile_size: int = 1024,
    overlap: float = 0.25,
    min_visible_fraction: float = 0.2,
    keep_empty_prob: float = 0.1,
):
    """
    Tile YOLO-format dataset into fixed-size overlapping patches.
    """

    stride = int(tile_size * (1 - overlap))
    logger.info(f"Tile size: {tile_size}")
    logger.info(f"Overlap: {overlap} (stride={stride})")

    for split in ["train", "val"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        logger.info(f"[{split}] Found {len(image_paths)} images")

        total_tiles = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read {img_path}")
                continue

            H, W = img.shape[:2]
            label_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = read_yolo_labels(label_path)

            abs_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = yolo_to_xyxy(b, W, H)
                abs_boxes.append((b.cls, x1, y1, x2, y2))

            nx = max(1, math.ceil((W - tile_size) / stride) + 1)
            ny = max(1, math.ceil((H - tile_size) / stride) + 1)

            for iy in range(ny):
                top = min(iy * stride, max(0, H - tile_size))
                bottom = top + tile_size

                for ix in range(nx):
                    left = min(ix * stride, max(0, W - tile_size))
                    right = left + tile_size

                    tile = img[top:bottom, left:right]
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        continue

                    tile_labels = []

                    for cls, x1, y1, x2, y2 in abs_boxes:
                        ix1 = max(x1, left)
                        iy1 = max(y1, top)
                        ix2 = min(x2, right)
                        iy2 = min(y2, bottom)

                        inter_area = area(ix1, iy1, ix2, iy2)
                        orig_area = area(x1, y1, x2, y2)

                        if orig_area == 0 or inter_area == 0:
                            continue

                        if inter_area / orig_area < min_visible_fraction:
                            continue

                        tx1 = ix1 - left
                        ty1 = iy1 - top
                        tx2 = ix2 - left
                        ty2 = iy2 - top

                        bw = tx2 - tx1
                        bh = ty2 - ty1
                        xc = tx1 + bw / 2
                        yc = ty1 + bh / 2

                        tile_labels.append(
                            (
                                cls,
                                xc / tile_size,
                                yc / tile_size,
                                bw / tile_size,
                                bh / tile_size,
                            )
                        )

                    if not tile_labels and random.random() > keep_empty_prob:
                        continue

                    tile_name = f"{img_path.stem}_x{left}_y{top}"
                    out_img_path = out_img_dir / f"{tile_name}{img_path.suffix}"
                    out_lbl_path = out_lbl_dir / f"{tile_name}.txt"

                    cv2.imwrite(str(out_img_path), tile)

                    if tile_labels:
                        out_lbl_path.write_text(
                            "\n".join(
                                f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                                for c, xc, yc, w, h in tile_labels
                            )
                            + "\n"
                        )
                    else:
                        out_lbl_path.write_text("")

                    total_tiles += 1

        logger.success(f"[{split}] Wrote {total_tiles} tiles")

    logger.success(f"Tiling complete → {output_dir}")


if __name__ == "__main__":
    app()