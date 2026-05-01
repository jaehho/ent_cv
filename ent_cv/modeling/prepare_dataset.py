"""Prepare a dataset by auto-splitting training data into train/val splits.

Splits are case-level: all frames from the same surgical case stay in the
same split to prevent data leakage from temporal/patient correlation.
"""
from pathlib import Path
import random
import re

from loguru import logger
import typer
import yaml

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


_CASE_RE = re.compile(r"^(.+?)_[Pp]art\d+")
_FRAME_RE = re.compile(r"^(.+)_f\d+$")


def _extract_case(image_path: str) -> str:
    """Extract the case identifier from an image path.

    Naming convention: {case}_Part{N}_f{frame_idx}.{ext}
    e.g. 20251113_02_Part1_f003000.png → 20251113_02

    Falls back to video stem (strip _f{digits}) if no _Part found.
    """
    stem = Path(image_path).stem
    m = _CASE_RE.match(stem)
    if m:
        return m.group(1)
    m = _FRAME_RE.match(stem)
    if m:
        return m.group(1)
    return stem


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

    # Group by case — all frames from a surgical case stay in the same split
    cases: dict[str, list[str]] = {}
    for img in all_images:
        cases.setdefault(_extract_case(img), []).append(img)

    case_keys = sorted(cases.keys())
    random.seed(seed)
    random.shuffle(case_keys)

    n_val_cases = max(1, round(len(case_keys) * val_split))
    val_case_set = set(case_keys[:n_val_cases])

    val_images = [img for key in case_keys if key in val_case_set for img in cases[key]]
    train_images = [img for key in case_keys if key not in val_case_set for img in cases[key]]

    logger.info(f"Cases: {len(case_keys)} total, {n_val_cases} val, {len(case_keys) - n_val_cases} train")
    for key in sorted(val_case_set):
        logger.info(f"  val:   {key} ({len(cases[key])} images)")
    for key in sorted(set(case_keys) - val_case_set):
        logger.info(f"  train: {key} ({len(cases[key])} images)")
    logger.info(f"Split: {len(train_images)} train / {len(val_images)} val images")

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
