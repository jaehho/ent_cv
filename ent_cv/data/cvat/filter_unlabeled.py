"""Extract unlabeled frames from a YOLO dataset into a new dataset.

A frame is considered "unlabeled" if its corresponding label file is either
missing or empty (no annotation rows).

Usage:
    uv run ent_cv/data/cvat/filter_unlabeled.py /path/to/dataset

Override the output directory:
    uv run ent_cv/data/cvat/filter_unlabeled.py /path/to/dataset --output-dir /path/to/unlabeled

The output directory has the standard Ultralytics YOLO layout:
    <output>/
        data.yaml
        train.txt
        images/train/<stem>.<ext>
        labels/train/<stem>.txt   (empty placeholder)
"""
import shutil
from pathlib import Path
from typing import Annotated

from loguru import logger
from rich.console import Console
from rich.table import Table
import typer
import yaml

from ent_cv.config import DATASETS_DIR

_console = Console()

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unlabeled(label_file: Path | None) -> bool:
    """Return True when *label_file* is absent or contains no annotation rows."""
    if label_file is None or not label_file.exists():
        return True
    return label_file.read_text().strip() == ""


def _find_image_files(images_dir: Path) -> dict[str, Path]:
    """Return {stem: path} for every image file under *images_dir*."""
    stem_map: dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        for img in images_dir.rglob(ext):
            if img.stem in stem_map:
                logger.warning(f"Duplicate image stem '{img.stem}' – keeping first occurrence")
            else:
                stem_map[img.stem] = img
    return stem_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    dataset_dir: Annotated[
        Path,
        typer.Argument(help="Dataset directory in Ultralytics YOLO layout."),
    ] = DATASETS_DIR / "combined",
    output_dir: Annotated[
        Path,
        typer.Option(help="Where to write the unlabeled-frames dataset."),
    ] = DATASETS_DIR / "unlabeled",
    copy_images: Annotated[
        bool,
        typer.Option(help="Copy image files into the output dataset."),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(help="Print what would be done without writing any files."),
    ] = False,
):
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        raise typer.Exit(1)

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    if not images_dir.exists():
        logger.error(f"images/train not found under {dataset_dir}")
        raise typer.Exit(1)

    # --------------------------------------------------------- discover frames
    image_files = _find_image_files(images_dir)
    if not image_files:
        logger.error(f"No image files found in {images_dir}")
        raise typer.Exit(1)

    logger.info(f"Found {len(image_files)} total frames in {images_dir}")

    unlabeled: dict[str, Path] = {}
    labeled_count = 0

    for stem, img_path in sorted(image_files.items()):
        label_file = labels_dir / f"{stem}.txt" if labels_dir.exists() else None
        if _is_unlabeled(label_file):
            unlabeled[stem] = img_path
        else:
            labeled_count += 1

    # --------------------------------------------------------- summary table
    summary = Table(title=f"Dataset analysis – {dataset_dir.name}", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right", style="cyan")
    summary.add_column("Percentage", justify="right")
    total = len(image_files)
    summary.add_row("Total frames", str(total), "100 %")
    summary.add_row(
        "Labeled frames",
        str(labeled_count),
        f"{labeled_count / total * 100:.1f} %",
    )
    summary.add_row(
        "Unlabeled frames",
        str(len(unlabeled)),
        f"{len(unlabeled) / total * 100:.1f} %",
    )
    _console.print(summary)

    if not unlabeled:
        _console.print("[green]No unlabeled frames found – nothing to do.[/green]")
        return

    if dry_run:
        _console.print(f"[yellow]Dry run – would write {len(unlabeled)} frames to {output_dir}[/yellow]")
        for stem in list(unlabeled)[:10]:
            _console.print(f"  {stem}")
        if len(unlabeled) > 10:
            _console.print(f"  … and {len(unlabeled) - 10} more")
        return

    # --------------------------------------------------------- build output dataset
    out_images = output_dir / "images" / "train"
    out_labels = output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Copy (or link) images and write empty label files
    for stem, img_path in sorted(unlabeled.items()):
        dest_img = out_images / img_path.name
        dest_lbl = out_labels / f"{stem}.txt"

        if copy_images:
            shutil.copy2(img_path, dest_img)
        else:
            # Symlink to avoid duplicating disk space
            if dest_img.exists() or dest_img.is_symlink():
                dest_img.unlink()
            dest_img.symlink_to(img_path)

        dest_lbl.write_text("")  # empty = no annotations

    logger.info(f"Wrote {len(unlabeled)} image files to {out_images}")

    # --------------------------------------------------------- data.yaml
    src_yaml = dataset_dir / "data.yaml"
    if src_yaml.exists():
        with src_yaml.open() as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    cfg["path"] = str(output_dir)
    cfg["train"] = "train.txt"
    cfg.pop("val", None)
    cfg.pop("test", None)

    out_yaml = output_dir / "data.yaml"
    with out_yaml.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Wrote {out_yaml}")

    # --------------------------------------------------------- train.txt
    train_txt_lines = "\n".join(
        f"images/train/{unlabeled[stem].name}" for stem in sorted(unlabeled)
    ) + "\n"
    (output_dir / "train.txt").write_text(train_txt_lines)
    logger.info(f"Wrote train.txt with {len(unlabeled)} entries")

    logger.success(
        f"Done! Unlabeled dataset ({len(unlabeled)} frames) is ready at: {output_dir}"
    )


if __name__ == "__main__":
    app()
