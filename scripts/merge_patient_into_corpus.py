"""Non-destructive merge of v2 Patient pseudo-labels into the instrument corpus.

Combines hand-labeled instrument GT (classes 0-11) from datasets/current/ with
verified v2 Patient pseudo-labels (class 12) from predictions/patient_v2_pseudo/
into a new dataset at datasets/current_with_patient/. The original
datasets/current/ is left untouched as a fallback.

Per-frame merge logic:
  - Frame has instrument labels AND patient pseudo → concatenate
  - Frame has only instrument labels             → keep instrument-only
                                                    (patient is true negative)
  - Frame has only patient pseudo                → keep patient-only
                                                    (no instruments in frame)
  - Frame has neither                            → no .txt (background)

Output structure mirrors current/ (single images/train + labels/train, split
tracked via train_split.txt / val_split.txt). This keeps existing training
tooling working without changes. The case-level split is preserved verbatim
from current/.

Sanity checks:
  - Instrument label files must contain only classes 0-11; any line at class
    12 is dropped (Patient should come from the pseudo dir, not from the
    instrument GT).
  - Patient pseudo files must contain only class 12; anything else is dropped.
  - The merged file count and frame-stem set must equal the source.
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

SRC_DATASET = DATASETS_DIR / "current"
DST_DATASET = DATASETS_DIR / "current_with_patient"
DEFAULT_PSEUDO_DIR = DATA_DIR / "predictions" / "patient_v2_pseudo" / "labels"

PATIENT_CLASS = LABELS.index("Patient")  # 12
INSTRUMENT_CLASSES = set(range(PATIENT_CLASS))  # {0..11}
CASE_RE = re.compile(r"^(.+?)_[Pp]art\d+")


def _case_of(stem: str) -> str:
    m = CASE_RE.match(stem)
    return m.group(1) if m else stem


def _keep_lines_by_class(path: Path, allowed: set[int]) -> tuple[list[str], Counter[int]]:
    """Filter lines to those whose class is in ``allowed``. Returns (kept_lines,
    full_class_counter). The counter includes dropped classes so the caller can
    surface them."""
    kept: list[str] = []
    counts: Counter[int] = Counter()
    if not path.exists():
        return kept, counts
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        counts[cls] += 1
        if cls in allowed:
            kept.append(line)
    return kept, counts


def _wipe(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def _rewrite_split_file(src_path: Path, src_dir: Path, dst_dir: Path, dst_split: Path) -> int:
    """Rewrite a YOLO split.txt so its image paths point to ``dst_dir`` instead
    of ``src_dir``. Preserves order and case-level split assignment.
    """
    n = 0
    out_lines: list[str] = []
    for raw in src_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        src_image = Path(line)
        # The split file might point at src_dir or its resolved real-path; we
        # only need the basename to map into dst.
        new_path = dst_dir / src_image.name
        out_lines.append(str(new_path))
        n += 1
    dst_split.write_text("\n".join(out_lines) + "\n")
    return n


@app.command()
def main(
    pseudo_dir: Path = typer.Option(
        DEFAULT_PSEUDO_DIR, help="Patient pseudo-label directory to merge in."
    ),
    src: Path = typer.Option(
        SRC_DATASET, help="Source instrument-only dataset directory."
    ),
    dst: Path = typer.Option(
        DST_DATASET, help="Destination merged dataset directory."
    ),
):
    """Merge instrument GT and patient pseudo-labels into a new dataset."""
    src = src.resolve()  # current/ is a symlink — work with the real target
    if not src.is_dir():
        raise typer.BadParameter(f"--src not found: {src}")
    if not pseudo_dir.is_dir():
        raise typer.BadParameter(f"--pseudo-dir not found: {pseudo_dir}")

    src_images = src / "images" / "train"
    src_labels = src / "labels" / "train"
    if not src_images.is_dir():
        raise typer.BadParameter(f"missing {src_images}")
    if not src_labels.is_dir():
        raise typer.BadParameter(f"missing {src_labels}")

    _wipe(dst)
    dst_images = dst / "images" / "train"
    dst_labels = dst / "labels" / "train"
    dst_images.mkdir(parents=True)
    dst_labels.mkdir(parents=True)

    images = sorted(src_images.glob("*.png"))
    logger.info(
        f"Merging {len(images)} frames\n"
        f"  src instrument labels: {src_labels}\n"
        f"  src patient pseudo:    {pseudo_dir}\n"
        f"  dst:                   {dst}"
    )

    stats = {
        "frames_total": len(images),
        "frames_with_instr_and_patient": 0,
        "frames_with_instr_only": 0,
        "frames_with_patient_only": 0,
        "frames_no_labels": 0,
        "instrument_lines_kept": 0,
        "patient_lines_kept": 0,
        "instrument_lines_dropped_wrong_class": 0,
        "patient_lines_dropped_wrong_class": 0,
    }

    for img in images:
        stem = img.stem
        instr_lines, instr_counts = _keep_lines_by_class(
            src_labels / f"{stem}.txt", INSTRUMENT_CLASSES
        )
        pat_lines, pat_counts = _keep_lines_by_class(
            pseudo_dir / f"{stem}.txt", {PATIENT_CLASS}
        )

        # Surface unexpected classes (Patient in instrument file, or
        # instrument in patient file) so we know if upstream got muddled.
        for cls in instr_counts:
            if cls not in INSTRUMENT_CLASSES:
                stats["instrument_lines_dropped_wrong_class"] += instr_counts[cls]
        for cls in pat_counts:
            if cls != PATIENT_CLASS:
                stats["patient_lines_dropped_wrong_class"] += pat_counts[cls]

        stats["instrument_lines_kept"] += len(instr_lines)
        stats["patient_lines_kept"] += len(pat_lines)

        if instr_lines and pat_lines:
            stats["frames_with_instr_and_patient"] += 1
        elif instr_lines:
            stats["frames_with_instr_only"] += 1
        elif pat_lines:
            stats["frames_with_patient_only"] += 1
        else:
            stats["frames_no_labels"] += 1

        merged = instr_lines + pat_lines
        if merged:
            (dst_labels / f"{stem}.txt").write_text("\n".join(merged) + "\n")

        # Symlink image to the underlying file (not to current/, which is itself
        # a symlink) so dst doesn't break if current/ is later repointed.
        dst_img = dst_images / img.name
        dst_img.symlink_to(img.resolve())

    # data.yaml + split files
    data_yaml = {
        "path": str(dst),
        "train": "train_split.txt",
        "val": "val_split.txt",
        "names": dict(enumerate(LABELS)),
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    n_train = _rewrite_split_file(
        src / "train_split.txt", src_images, dst_images, dst / "train_split.txt"
    )
    n_val = _rewrite_split_file(
        src / "val_split.txt", src_images, dst_images, dst / "val_split.txt"
    )
    logger.info(f"Split mirrored: train={n_train}, val={n_val}")

    # Case-level split sanity — same invariant as the patient_v1/v2 builders.
    train_cases = {_case_of(Path(p).stem) for p in (dst / "train_split.txt").read_text().splitlines() if p.strip()}
    val_cases = {_case_of(Path(p).stem) for p in (dst / "val_split.txt").read_text().splitlines() if p.strip()}
    overlap = train_cases & val_cases
    if overlap:
        raise RuntimeError(
            f"Case-level split broken: {len(overlap)} case(s) in both train and val "
            f"({sorted(overlap)[:5]})"
        )

    write_manifest(
        dst,
        convention="grasp-coupled-with-patient",
        notes=(
            "Instrument GT (classes 0-11, hand-labeled in CVAT) merged with "
            f"verified v2 Patient pseudo-labels (class 12) from {pseudo_dir}. "
            "Case-level split mirrored from datasets/current/."
        ),
    )

    logger.info("Merge stats:")
    for k, v in stats.items():
        logger.info(f"  {k:<40} {v}")
    if stats["instrument_lines_dropped_wrong_class"]:
        logger.warning(
            f"Dropped {stats['instrument_lines_dropped_wrong_class']} unexpected-class "
            f"lines from instrument labels (likely stale Patient lines from a "
            f"previous merge — investigate if non-zero)."
        )
    if stats["patient_lines_dropped_wrong_class"]:
        logger.warning(
            f"Dropped {stats['patient_lines_dropped_wrong_class']} unexpected-class "
            f"lines from patient pseudo labels — investigate."
        )

    logger.success(
        f"Merged dataset ready at {dst}\n"
        f"  train: {n_train} frames\n"
        f"  val:   {n_val} frames\n"
        f"Train with the existing pipeline pointed at {dst}/data.yaml."
    )


if __name__ == "__main__":
    app()
