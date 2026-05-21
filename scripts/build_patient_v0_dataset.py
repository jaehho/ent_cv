"""Flatten the patient-only CVAT export into a clean 1-class YOLO dataset.

CVAT preserves the SHARE-relative path inside the zip, so labels and images
land at ``<split>/datasets/current/images/train/<stem>.<ext>``. We also need
to remap class 12 (Patient, the canonical multi-class index) to class 0 for a
single-class detector.

Frames without annotations are dropped: the user only reviewed ~100 of 420
uploaded frames, so unlabeled ones are *unreviewed* rather than confirmed
background. Treating them as background would inject false negatives.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer
import yaml

import random
import re

from ent_cv.config import DATASETS_DIR, LABELS
from ent_cv.data.manifest import write_manifest

_CASE_RE = re.compile(r"^(.+?)_[Pp]art\d+")

app = typer.Typer(add_completion=False)

PATIENT_MULTICLASS_INDEX = LABELS.index("Patient")  # contract: stays at 12
TARGET_CLASS = 0
TARGET_NAME = "Patient"


def _flatten(src_root: Path) -> list[tuple[Path, Path]]:
    """Walk a nested ``images/train/.../<stem>.png`` (or labels) and return
    [(src, flat_target_stem)] pairs.
    """
    pairs: list[tuple[Path, Path]] = []
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        pairs.append((p, Path(p.name)))
    return pairs


def _remap_labels(label_pairs: list[tuple[Path, Path]], dst_labels: Path) -> set[str]:
    """Copy label files to flat dst_labels, remapping class index to 0.
    Returns the set of stems that have at least one box.
    """
    labeled: set[str] = set()
    for src, name in label_pairs:
        text = src.read_text().strip()
        if not text:
            continue
        out_lines: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if int(parts[0]) != PATIENT_MULTICLASS_INDEX:
                raise ValueError(
                    f"Unexpected class index {parts[0]} in {src.name}; "
                    f"expected only {PATIENT_MULTICLASS_INDEX} (Patient)."
                )
            out_lines.append(" ".join([str(TARGET_CLASS), *parts[1:]]))
        if not out_lines:
            continue
        (dst_labels / name).write_text("\n".join(out_lines) + "\n")
        labeled.add(name.stem)
    return labeled


@app.command()
def main(
    export_dir: Path = typer.Option(
        DATASETS_DIR / "exports" / "patient-only" / "2026-05-15",
        help="Source CVAT export directory.",
    ),
    out_dir: Path = typer.Option(
        DATASETS_DIR / "patient_v0",
        help="Destination dataset directory.",
    ),
    val_split: float = typer.Option(0.2, help="Validation fraction (case-level)."),
    seed: int = typer.Option(42),
):
    """Flatten + remap CVAT export into a 1-class Patient dataset, then split."""
    images_src = export_dir / "images"
    labels_src = export_dir / "labels"
    if not images_src.is_dir() or not labels_src.is_dir():
        raise typer.BadParameter(f"missing images/ or labels/ under {export_dir}")

    out_images = out_dir / "images" / "train"
    out_labels = out_dir / "labels" / "train"
    if out_dir.exists():
        # Wipe to keep the build idempotent.
        for p in out_dir.rglob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
        for p in sorted(out_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        out_dir.rmdir()
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    label_pairs = _flatten(labels_src)
    logger.info(f"Found {len(label_pairs)} label files in export")
    labeled_stems = _remap_labels(label_pairs, out_labels)
    logger.info(
        f"  {len(labeled_stems)} frames have ≥1 Patient box (remapped to class 0)"
    )

    image_pairs = _flatten(images_src)
    logger.info(f"Found {len(image_pairs)} image files in export")
    n_linked = 0
    train_lines: list[str] = []
    for src, name in image_pairs:
        if name.stem not in labeled_stems:
            continue
        dst = out_images / name
        if not dst.exists():
            dst.symlink_to(src.resolve())
        n_linked += 1
        # Use the symlink path (not resolve()) so YOLO's `images/->labels/`
        # substitution lands inside this dataset's labels/train/, not the
        # deeply-nested CVAT export tree.
        train_lines.append(str(dst))
    logger.info(f"  symlinked {n_linked} labeled images to {out_images}")

    train_txt = out_dir / "train.txt"
    train_txt.write_text("\n".join(sorted(train_lines)) + "\n")

    data_yaml = {
        "path": str(out_dir),
        "train": str(train_txt),
        "names": {TARGET_CLASS: TARGET_NAME},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    write_manifest(
        out_dir,
        convention="patient-only",
        notes=(
            "1-class Patient bootstrap dataset. Class 12 (Patient in multi-class "
            "LABELS) remapped to class 0. Only frames with ≥1 Patient box kept."
        ),
    )

    # Case-level split, inlined to avoid prepare_dataset's `.resolve()` call
    # (which follows symlinks and breaks YOLO's images→labels substitution).
    cases: dict[str, list[str]] = {}
    for img_path in train_lines:
        stem = Path(img_path).stem
        m = _CASE_RE.match(stem)
        case = m.group(1) if m else stem
        cases.setdefault(case, []).append(img_path)
    case_keys = sorted(cases)
    random.seed(seed)
    random.shuffle(case_keys)
    n_val = max(1, round(len(case_keys) * val_split))
    val_cases = set(case_keys[:n_val])
    val_imgs = [im for k in case_keys if k in val_cases for im in cases[k]]
    train_imgs = [im for k in case_keys if k not in val_cases for im in cases[k]]
    (out_dir / "train_split.txt").write_text("\n".join(sorted(train_imgs)) + "\n")
    (out_dir / "val_split.txt").write_text("\n".join(sorted(val_imgs)) + "\n")
    data_yaml_full = {
        "path": str(out_dir),
        "train": str(out_dir / "train_split.txt"),
        "val": str(out_dir / "val_split.txt"),
        "names": {TARGET_CLASS: TARGET_NAME},
    }
    (out_dir / "data_with_val.yaml").write_text(
        yaml.safe_dump(data_yaml_full, sort_keys=False)
    )
    logger.info(
        f"Case split — {len(case_keys)} cases, {n_val} val ({len(val_imgs)} img) "
        f"/ {len(case_keys) - n_val} train ({len(train_imgs)} img)"
    )
    logger.info(f"  val cases:   {sorted(val_cases)}")
    logger.info(f"  train cases: {sorted(set(case_keys) - val_cases)}")

    logger.success(
        f"Done — {out_dir}/data_with_val.yaml is ready for training."
    )


if __name__ == "__main__":
    app()
