"""Merge HN-verified labels for 3 new cases into datasets/current_full_v1/.

Starting from datasets/current_with_patient/ (the v0 corpus, 14 cases,
single-dir + split.txt layout), this script:

  1. Mirrors all images and labels from the existing corpus into the new dst.
  2. Adds each new case's HN-verified labels + extracted frames.
  3. Writes new train_split.txt / val_split.txt respecting the case
     assignment from --train-cases / --val-cases.
  4. Verifies case-level isolation (no case in both splits) before finishing.

The HN-verified labels already include Patient (class 12), so no separate
Patient-pseudo merge is needed for the new cases. Existing cases keep their
already-merged Patient labels from current_with_patient/.
"""
from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR, DATASETS_DIR, LABELS
from ent_cv.data.manifest import write_manifest

app = typer.Typer(add_completion=False)

SRC_DATASET = DATASETS_DIR / "current_with_patient"
DST_DATASET = DATASETS_DIR / "current_full_v2"
DEFAULT_HN_LABELS = DATA_DIR / "predictions" / "full_v0_hn_verified" / "labels"
DEFAULT_NEW_FRAMES = DATA_DIR / "processed" / "extracted_frames" / "batch3"
CASE_RE = re.compile(r"^(.+?)_[Pp]art\d+")
NOT_SURE_CLASS = 11  # config.LABELS index for "Not Sure"


def _case_of(stem: str) -> str:
    m = CASE_RE.match(stem)
    return m.group(1) if m else stem


def _wipe(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def _label_has_class(label_path: Path, cls: int) -> bool:
    target = str(cls)
    for ln in label_path.read_text().splitlines():
        parts = ln.strip().split()
        if parts and parts[0] == target:
            return True
    return False


def _rewrite_split_file(src_path: Path, dst_images_dir: Path, dst_split: Path) -> int:
    """Rewrite a split file so paths point at dst_images_dir."""
    n = 0
    out_lines: list[str] = []
    for raw in src_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        out_lines.append(str(dst_images_dir / Path(line).name))
        n += 1
    dst_split.write_text("\n".join(out_lines) + "\n")
    return n


def _gather_new_case_images(case: str, frames_root: Path) -> dict[str, Path]:
    """{stem: image_path} for all extracted frames of one case."""
    out: dict[str, Path] = {}
    for img in frames_root.glob(f"{case}_Part*/*.png"):
        out[img.stem] = img
    return out


@app.command()
def main(
    train_cases: str = typer.Option(
        "20260128_01,20260205_03", help="New cases to add to train (comma-separated)."
    ),
    val_cases: str = typer.Option(
        "20260205_01", help="New cases to add to val (comma-separated)."
    ),
    hn_labels: Path = typer.Option(
        DEFAULT_HN_LABELS, help="HN-verified labels directory."
    ),
    new_frames: Path = typer.Option(
        DEFAULT_NEW_FRAMES, help="Where new-case extracted frames live."
    ),
    src: Path = typer.Option(SRC_DATASET, help="Existing merged dataset to extend."),
    dst: Path = typer.Option(DST_DATASET, help="Destination merged dataset."),
    drop_not_sure_frames: bool = typer.Option(
        True,
        help=(
            "Skip frames whose HN-verified label file contains a class-11 "
            "(Not Sure) box. Canonical labels are not modified; only the "
            "built dataset excludes these frames."
        ),
    ),
):
    """Build current_full_v1/ = current_with_patient/ + HN-verified new cases."""
    train_list = [c.strip() for c in train_cases.split(",") if c.strip()]
    val_list = [c.strip() for c in val_cases.split(",") if c.strip()]
    overlap = set(train_list) & set(val_list)
    if overlap:
        raise typer.BadParameter(f"case in both train and val: {overlap}")

    src = src.resolve()
    if not src.is_dir():
        raise typer.BadParameter(f"--src not found: {src}")
    if not hn_labels.is_dir():
        raise typer.BadParameter(f"--hn-labels not found: {hn_labels}")
    if not new_frames.is_dir():
        raise typer.BadParameter(f"--new-frames not found: {new_frames}")

    src_images = src / "images" / "train"
    src_labels = src / "labels" / "train"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise typer.BadParameter(f"src missing images/train or labels/train: {src}")

    _wipe(dst)
    dst_images = dst / "images" / "train"
    dst_labels = dst / "labels" / "train"
    dst_images.mkdir(parents=True)
    dst_labels.mkdir(parents=True)

    # 1. Mirror existing corpus (symlink images, copy labels).
    n_old_images = 0
    for img in src_images.glob("*.png"):
        (dst_images / img.name).symlink_to(img.resolve())
        n_old_images += 1
    n_old_labels = 0
    for lbl in src_labels.glob("*.txt"):
        shutil.copy2(lbl, dst_labels / lbl.name)
        n_old_labels += 1
    logger.info(f"Mirrored existing corpus: {n_old_images} images, {n_old_labels} labels")

    # 2. Add new cases. Track which frames go to train vs val for split files.
    new_train_stems: list[str] = []
    new_val_stems: list[str] = []
    per_case_stats: dict[str, dict] = {}
    dropped_not_sure: list[str] = []

    for case_list, split_name, sink in (
        (train_list, "train", new_train_stems),
        (val_list, "val", new_val_stems),
    ):
        for case in case_list:
            case_images = _gather_new_case_images(case, new_frames)
            if not case_images:
                raise RuntimeError(
                    f"No extracted frames found for case {case!r} under {new_frames}"
                )
            stats = {"frames": 0, "boxes": 0, "classes": Counter(), "skipped_not_sure": 0}
            for stem, img in case_images.items():
                label_path = hn_labels / f"{stem}.txt"
                if not label_path.exists():
                    raise RuntimeError(
                        f"No HN-verified label for image {stem} (case {case})"
                    )
                if drop_not_sure_frames and _label_has_class(label_path, NOT_SURE_CLASS):
                    stats["skipped_not_sure"] += 1
                    dropped_not_sure.append(stem)
                    continue
                (dst_images / img.name).symlink_to(img.resolve())
                shutil.copy2(label_path, dst_labels / f"{stem}.txt")
                lines = [
                    ln for ln in label_path.read_text().splitlines() if ln.strip()
                ]
                stats["frames"] += 1
                stats["boxes"] += len(lines)
                for ln in lines:
                    stats["classes"][int(ln.split()[0])] += 1
                sink.append(stem)
            per_case_stats[f"{case} ({split_name})"] = stats

    if dropped_not_sure:
        (dst / "dropped_not_sure_frames.txt").write_text(
            "# Frames excluded because their HN-verified label contains a class-11 "
            "(Not Sure) box.\n"
            f"# Canonical labels preserved at {hn_labels}\n"
            + "\n".join(sorted(dropped_not_sure))
            + "\n"
        )

    # 3. Rewrite split files: existing entries (re-pointed at dst) + new stems.
    n_existing_train = _rewrite_split_file(
        src / "train_split.txt", dst_images, dst / "train_split.txt"
    )
    n_existing_val = _rewrite_split_file(
        src / "val_split.txt", dst_images, dst / "val_split.txt"
    )
    # Append new stems
    with (dst / "train_split.txt").open("a") as f:
        for s in new_train_stems:
            f.write(f"{dst_images / (s + '.png')}\n")
    with (dst / "val_split.txt").open("a") as f:
        for s in new_val_stems:
            f.write(f"{dst_images / (s + '.png')}\n")

    # 4. Case-level isolation check.
    train_cases_in_dst = {
        _case_of(Path(ln.strip()).stem)
        for ln in (dst / "train_split.txt").read_text().splitlines()
        if ln.strip()
    }
    val_cases_in_dst = {
        _case_of(Path(ln.strip()).stem)
        for ln in (dst / "val_split.txt").read_text().splitlines()
        if ln.strip()
    }
    overlap = train_cases_in_dst & val_cases_in_dst
    if overlap:
        raise RuntimeError(
            f"Case-level isolation broken — cases in both splits: {sorted(overlap)}"
        )

    # 5. data.yaml
    data_yaml = {
        "path": str(dst),
        "train": "train_split.txt",
        "val": "val_split.txt",
        "names": dict(enumerate(LABELS)),
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # 6. Manifest
    write_manifest(
        dst,
        convention="grasp-coupled-with-patient",
        notes=(
            "current_with_patient corpus extended with HN-verified labels on 3 "
            f"new cases ({sorted(train_list + val_list)}). "
            f"Train: {sorted(train_list)}. Val: {sorted(val_list)}. "
            "HN-verified labels are hand-corrected; new cases include "
            "first-ever Empty Hand / Not Sure instances."
        ),
    )

    # 7. Report
    logger.info("Per-new-case stats:")
    for k, s in per_case_stats.items():
        cls_str = ", ".join(
            f"{LABELS[c]}:{n}" for c, n in sorted(s["classes"].items())
        )
        logger.info(
            f"  {k:<30} frames={s['frames']:>3}  boxes={s['boxes']:>4}\n"
            f"    {cls_str}"
        )
    logger.info(
        f"\nFinal split counts:\n"
        f"  train cases: {len(train_cases_in_dst)}   "
        f"frames: {n_existing_train + len(new_train_stems)}\n"
        f"  val   cases: {len(val_cases_in_dst)}   "
        f"frames: {n_existing_val + len(new_val_stems)}"
    )

    logger.success(
        f"Merged dataset ready at {dst}\n"
        f"  data.yaml:    {dst / 'data.yaml'}\n"
        f"  manifest:     {dst / 'manifest.yaml'}\n"
        f"Train with: scripts/train_full_v2.py (point it at this data.yaml)."
    )


if __name__ == "__main__":
    app()
