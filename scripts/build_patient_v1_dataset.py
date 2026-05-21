"""Build patient_v1 dataset = v0 corpus + 90 hand-corrected frames from CVAT task 106.

Adds 3 new cases (20251210_01/02/03) to the bootstrap detector's training set.
Validation case stays the same as v0 (20251208_01) so mAP numbers are directly
comparable. Class 12 (Patient) is remapped to class 0 to match the single-class
detector's label space.

Structure: standard YOLO layout — images/{train,val}/ and labels/{train,val}/
each hold their own split. v0 dumped everything under images/train/ and tracked
the split with sidecar .txt files, which caused YOLO to derive the same
labels/train.cache path for both train and val scans; the val scan would
overwrite the train cache. Splitting the directories sidesteps that entirely.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from cvat_sdk import make_client
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer
import yaml

from ent_cv.config import DATASETS_DIR, LABELS
from ent_cv.data.manifest import write_manifest

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

V0_DATASET = DATASETS_DIR / "patient_v0"
V1_DATASET = DATASETS_DIR / "patient_v1"
CURRENT_IMAGES = DATASETS_DIR / "current" / "images" / "train"

PATIENT_MULTICLASS_INDEX = LABELS.index("Patient")  # 12
TARGET_CLASS = 0
TARGET_NAME = "Patient"

# Keep the same val case as v0 so the val number is comparable across versions.
VAL_CASES: set[str] = {"20251208_01"}

CASE_RE = re.compile(r"^(.+?)_[Pp]art\d+")


def _case_of(stem: str) -> str | None:
    m = CASE_RE.match(stem)
    return m.group(1) if m else None


def _remap_label_text(text: str, gt_class: int) -> str:
    """Keep only ``gt_class`` lines, remap to ``TARGET_CLASS``=0."""
    out_lines: list[str] = []
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        if int(parts[0]) != gt_class:
            continue
        out_lines.append(f"{TARGET_CLASS} " + " ".join(parts[1:]))
    return ("\n".join(out_lines) + "\n") if out_lines else ""


def _gt_patient_class(export_dir: Path) -> int:
    """Pull Patient's class index out of CVAT's exported data.yaml.

    CVAT indexes YOLO classes by project label order — don't assume it matches
    ent_cv.config.LABELS.
    """
    data_yaml = export_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"data.yaml missing in CVAT export: {data_yaml}")
    spec = yaml.safe_load(data_yaml.read_text())
    names = spec.get("names")
    if isinstance(names, dict):
        for idx, name in names.items():
            if name == "Patient":
                return int(idx)
    elif isinstance(names, list):
        for idx, name in enumerate(names):
            if name == "Patient":
                return idx
    raise RuntimeError(f"'Patient' not in data.yaml names: {names}")


def _wipe(p: Path) -> None:
    if not p.exists():
        return
    shutil.rmtree(p)


def _copy_v0(out_root: Path) -> tuple[list[str], list[str]]:
    """Pull all v0 frames into v1, routing by v0's existing split.

    Returns (train_paths, val_paths) — absolute paths to v1 symlinks.
    Images become fresh symlinks to the underlying file (skipping the v0
    symlink so v1 doesn't break if v0 is later wiped). Labels are copied as
    text since v0 already remapped class 12 → 0.
    """
    train: list[str] = []
    val: list[str] = []
    out_by_split = {"train": train, "val": val}

    for split, dst_list in out_by_split.items():
        split_file = V0_DATASET / f"{split}_split.txt"
        if not split_file.exists():
            raise RuntimeError(f"missing {split_file}; rebuild v0 first")
        for raw in split_file.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            v0_img = Path(raw)
            real = v0_img.resolve()
            if not real.exists():
                logger.warning(f"  v0 image missing on disk: {v0_img} → {real}")
                continue
            dst_img = out_root / "images" / split / v0_img.name
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(real)

            lbl_src = V0_DATASET / "labels" / "train" / f"{v0_img.stem}.txt"
            if lbl_src.exists():
                (out_root / "labels" / split / f"{v0_img.stem}.txt").write_text(
                    lbl_src.read_text()
                )
            dst_list.append(str(dst_img))

    return train, val


def _add_task_frames(
    out_root: Path,
    labels_by_stem: dict[str, str],
    gt_class: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    train: list[str] = []
    val: list[str] = []
    counts = {
        "added_train": 0, "added_val": 0,
        "skipped_no_image": 0, "skipped_no_boxes": 0, "skipped_unknown_case": 0,
    }

    for stem, text in labels_by_stem.items():
        remapped = _remap_label_text(text, gt_class)
        if not remapped.strip():
            counts["skipped_no_boxes"] += 1
            continue

        case = _case_of(stem)
        if case is None:
            counts["skipped_unknown_case"] += 1
            continue
        split = "val" if case in VAL_CASES else "train"

        src_img = CURRENT_IMAGES / f"{stem}.png"
        if not src_img.exists():
            counts["skipped_no_image"] += 1
            continue

        dst_img = out_root / "images" / split / f"{stem}.png"
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(src_img.resolve())

        (out_root / "labels" / split / f"{stem}.txt").write_text(remapped)
        if split == "val":
            val.append(str(dst_img))
            counts["added_val"] += 1
        else:
            train.append(str(dst_img))
            counts["added_train"] += 1
    return train, val, counts


@app.command()
def main(
    task_id: int = typer.Option(106, help="CVAT task ID with hand-corrected GT frames."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
):
    """Build patient_v1 dataset (v0 corpus + task 106 corrections)."""
    if not V0_DATASET.exists():
        raise typer.BadParameter(
            f"patient_v0 dataset not found at {V0_DATASET}. Run "
            f"scripts/build_patient_v0_dataset.py first."
        )

    _wipe(V1_DATASET)
    for split in ("train", "val"):
        (V1_DATASET / "images" / split).mkdir(parents=True)
        (V1_DATASET / "labels" / split).mkdir(parents=True)

    # ------------------------------------------------------------ v0 corpus
    v0_train, v0_val = _copy_v0(V1_DATASET)
    logger.info(f"v0 corpus carried over: train={len(v0_train)}, val={len(v0_val)}")

    # ------------------------------------------------------------ task 106
    with tempfile.TemporaryDirectory(prefix=f"v1_{task_id}_") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "task.zip"
        with make_client(
            host=host, port=port, credentials=(username, password)
        ) as client:
            task = client.tasks.retrieve(task_id)
            logger.info(f"Exporting CVAT task {task_id}: '{task.name}'")
            task.export_dataset(
                format_name="Ultralytics YOLO Detection 1.0",
                filename=str(zip_path),
                include_images=False,
            )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        gt_class = _gt_patient_class(tmp)
        logger.info(f"  CVAT exports Patient as class {gt_class}")
        labels_by_stem: dict[str, str] = {}
        for p in (tmp / "labels").rglob("*.txt"):
            labels_by_stem[p.stem] = p.read_text()
        logger.info(f"  pulled {len(labels_by_stem)} GT label files")
        new_train, new_val, counts = _add_task_frames(
            V1_DATASET, labels_by_stem, gt_class
        )
    logger.info(
        f"task {task_id}: +{counts['added_train']} train, +{counts['added_val']} val, "
        f"skipped(no_image={counts['skipped_no_image']}, "
        f"no_boxes={counts['skipped_no_boxes']}, "
        f"unknown_case={counts['skipped_unknown_case']})"
    )

    # ------------------------------------------------------------ data.yaml
    train_paths = sorted(v0_train + new_train)
    val_paths = sorted(v0_val + new_val)
    data_yaml = {
        "path": str(V1_DATASET),
        "train": "images/train",
        "val": "images/val",
        "names": {TARGET_CLASS: TARGET_NAME},
    }
    (V1_DATASET / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # Sanity: enforce no frame-stem collisions between train and val. With
    # case-level splitting this should be impossible, but check anyway so a
    # future bug doesn't silently leak val into train.
    train_stems = {Path(p).stem for p in train_paths}
    val_stems = {Path(p).stem for p in val_paths}
    overlap = train_stems & val_stems
    if overlap:
        raise RuntimeError(
            f"train/val stem overlap ({len(overlap)} frames) — case split is broken. "
            f"Examples: {sorted(overlap)[:5]}"
        )

    write_manifest(
        V1_DATASET,
        convention="patient-only",
        notes=(
            f"1-class Patient bootstrap dataset v1. v0 corpus + 90 hand-corrected "
            f"frames from CVAT task {task_id} (cases 20251210_01/02/03). "
            f"Val case: {sorted(VAL_CASES)} (same as v0)."
        ),
    )

    # Per-case summary so we can sanity-check the split.
    case_counts: dict[str, dict[str, int]] = {}
    for split_name, paths in (("train", train_paths), ("val", val_paths)):
        for path in paths:
            case = _case_of(Path(path).stem) or "unknown"
            case_counts.setdefault(case, {"train": 0, "val": 0})[split_name] += 1
    logger.info("Per-case frame counts:")
    for case in sorted(case_counts):
        c = case_counts[case]
        logger.info(f"  {case:<20}  train={c['train']:>4}  val={c['val']:>4}")

    logger.success(
        f"patient_v1 dataset ready at {V1_DATASET}\n"
        f"  train: {len(train_paths)} frames\n"
        f"  val:   {len(val_paths)} frames\n"
        f"Train with:\n"
        f"  uv run python scripts/train_patient_v1.py"
    )


if __name__ == "__main__":
    app()
