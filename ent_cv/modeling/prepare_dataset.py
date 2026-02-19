"""Prepare a dataset by auto-splitting training data into train/val splits."""
import random
from pathlib import Path

import typer
import yaml
from loguru import logger

app = typer.Typer(add_completion=False)


def resolve_image_paths(raw_paths: list[str], dataset_dir: Path) -> list[str]:
    """
    Convert relative image paths from train.txt to absolute paths.
    Strips a leading 'data/' prefix if present, since the exported txt uses
    'data/images/...' but the actual files are at dataset_dir/images/...
    """
    resolved = []
    for p in raw_paths:
        if p.startswith("data/"):
            p = p[len("data/"):]
        abs_path = (dataset_dir / p).resolve()
        if abs_path.exists():
            resolved.append(str(abs_path))
        else:
            logger.warning(f"Image not found, skipping: {abs_path}")
    return resolved


def run(dataset_dir: Path, val_split: float = 0.2, seed: int = 42) -> None:
    """Split training data into train/val and write data_with_val.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    new_yaml_path = dataset_dir / "data_with_val.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if "val" in data:
        logger.info("data.yaml already has a 'val' split — nothing to do.")
        return

    logger.info(f"No 'val' split found — auto-splitting {int(val_split * 100)}% of training data...")

    train_txt = dataset_dir / data["train"]
    with open(train_txt) as f:
        raw_paths = [line.strip() for line in f if line.strip()]

    all_images = resolve_image_paths(raw_paths, dataset_dir)
    logger.info(f"Resolved {len(all_images)}/{len(raw_paths)} image paths")

    if not all_images:
        raise RuntimeError("No images resolved — check dataset directory and train.txt paths.")

    random.seed(seed)
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_split))
    val_images = all_images[:n_val]
    train_images = all_images[n_val:]
    logger.info(f"Split: {len(train_images)} train / {len(val_images)} val")

    new_train_txt = dataset_dir / "train_split.txt"
    new_val_txt = dataset_dir / "val_split.txt"
    new_train_txt.write_text("\n".join(train_images))
    new_val_txt.write_text("\n".join(val_images))

    new_data = {**data, "train": str(new_train_txt), "val": str(new_val_txt), "path": str(dataset_dir)}
    with open(new_yaml_path, "w") as f:
        yaml.dump(new_data, f)

    logger.success(f"Dataset prepared — YAML written to: {new_yaml_path}")


@app.command()
def main(
    dataset_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    val_split: float = typer.Option(0.2, help="Fraction of training data for validation"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Split training data into train/val and write data_with_val.yaml."""
    run(dataset_dir, val_split, seed)


if __name__ == "__main__":
    app()
