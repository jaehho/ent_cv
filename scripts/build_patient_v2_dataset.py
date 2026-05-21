"""Build patient_v2 dataset = v1 corpus + tasks 107 + 108 (hand-corrections).

v1 used 8 cases (Nov-Dec 2025). Task 107 added 3 cases of corrections after
v1's first round of pseudo-labels (20251217_01, 20251217_03, 20260108_01).
Task 108 covers the remaining 3 unseen cases (20251217_02, 20251218_01,
20251218_02). v2 trains on all 14 cases.

Val stays as 20251208_01 (same as v0/v1) so the val mAP number is comparable
across versions. New cases all go to train — we want v2 to see the broadest
domain variety since we've exhausted the unseen-case test reservoir.

Layout: dual-dir (images/{train,val}/ + labels/{train,val}/), the cleaner
structure introduced after v1 to avoid YOLO's cache-name collision.
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

V1_DATASET = DATASETS_DIR / "patient_v1"
V2_DATASET = DATASETS_DIR / "patient_v2"
CURRENT_IMAGES = DATASETS_DIR / "current" / "images" / "train"

PATIENT_MULTICLASS_INDEX = LABELS.index("Patient")  # 12
TARGET_CLASS = 0
TARGET_NAME = "Patient"

# Same val case as v0/v1 — comparable cross-version metric.
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
    """Pull Patient's class index out of CVAT's exported data.yaml."""
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


def _copy_v1(out_root: Path) -> tuple[list[str], list[str]]:
    """Carry v1 frames into v2.

    Reads v1's split files (v1 is in legacy single-dir layout) and routes
    images and labels into v2's dual-dir layout. Images become fresh symlinks
    to the underlying file so v2 survives a v1 wipe. Labels are copied as
    text — v1 already remapped class 12 → 0.
    """
    train: list[str] = []
    val: list[str] = []
    out_by_split = {"train": train, "val": val}

    for split, dst_list in out_by_split.items():
        split_file = V1_DATASET / f"{split}_split.txt"
        if not split_file.exists():
            raise RuntimeError(f"missing {split_file}; rebuild v1 first")
        for raw in split_file.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            v1_img = Path(raw)
            real = v1_img.resolve()
            if not real.exists():
                logger.warning(f"  v1 image missing on disk: {v1_img} → {real}")
                continue
            dst_img = out_root / "images" / split / v1_img.name
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(real)

            lbl_src = V1_DATASET / "labels" / "train" / f"{v1_img.stem}.txt"
            if lbl_src.exists():
                (out_root / "labels" / split / f"{v1_img.stem}.txt").write_text(
                    lbl_src.read_text()
                )
            dst_list.append(str(dst_img))

    return train, val


def _add_task_frames(
    out_root: Path,
    task_id: int,
    labels_by_stem: dict[str, str],
    gt_class: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Add frames from one CVAT task. Splits by case (val if in VAL_CASES, else train)."""
    train: list[str] = []
    val: list[str] = []
    counts = {
        "added_train": 0, "added_val": 0,
        "skipped_no_image": 0, "skipped_no_boxes": 0, "skipped_unknown_case": 0,
        "skipped_collision": 0,
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
            # Same stem already pulled (e.g., from v1 or earlier task). Skip
            # rather than overwrite — v1's label is the same hand-correction
            # round-tripped through patient_v0_pseudo. Future tasks may differ;
            # log so we can investigate.
            counts["skipped_collision"] += 1
            logger.debug(f"  task {task_id}: stem already present: {stem}")
            continue
        dst_img.symlink_to(src_img.resolve())

        (out_root / "labels" / split / f"{stem}.txt").write_text(remapped)
        if split == "val":
            val.append(str(dst_img))
            counts["added_val"] += 1
        else:
            train.append(str(dst_img))
            counts["added_train"] += 1
    return train, val, counts


def _pull_task(
    client, task_id: int, tmp: Path
) -> tuple[dict[str, str], int]:
    """Export a CVAT task to ``tmp`` and return ({stem: label_text}, gt_class)."""
    zip_path = tmp / f"task_{task_id}.zip"
    task = client.tasks.retrieve(task_id)
    logger.info(f"Exporting task {task_id}: '{task.name}'")
    task.export_dataset(
        format_name="Ultralytics YOLO Detection 1.0",
        filename=str(zip_path),
        include_images=False,
    )
    extract_dir = tmp / f"task_{task_id}_unzipped"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    gt_class = _gt_patient_class(extract_dir)
    labels_by_stem: dict[str, str] = {}
    for p in (extract_dir / "labels").rglob("*.txt"):
        labels_by_stem[p.stem] = p.read_text()
    return labels_by_stem, gt_class


def _wipe(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


@app.command()
def main(
    task_ids: str = typer.Option(
        "107,108",
        help="Comma-separated CVAT task IDs of new corrections to fold into v2.",
    ),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
):
    """Build patient_v2 dataset (v1 corpus + task corrections)."""
    if not V1_DATASET.exists():
        raise typer.BadParameter(f"patient_v1 dataset not found at {V1_DATASET}")
    task_id_list = [int(t.strip()) for t in task_ids.split(",") if t.strip()]
    if not task_id_list:
        raise typer.BadParameter("--task-ids cannot be empty")

    _wipe(V2_DATASET)
    for split in ("train", "val"):
        (V2_DATASET / "images" / split).mkdir(parents=True)
        (V2_DATASET / "labels" / split).mkdir(parents=True)

    # ------------------------------------------------------------ v1 corpus
    v1_train, v1_val = _copy_v1(V2_DATASET)
    logger.info(f"v1 corpus carried over: train={len(v1_train)}, val={len(v1_val)}")

    # ------------------------------------------------------------ tasks
    new_train: list[str] = []
    new_val: list[str] = []
    with tempfile.TemporaryDirectory(prefix="v2_build_") as tmpdir:
        tmp = Path(tmpdir)
        with make_client(
            host=host, port=port, credentials=(username, password)
        ) as client:
            for task_id in task_id_list:
                labels_by_stem, gt_class = _pull_task(client, task_id, tmp)
                logger.info(
                    f"  task {task_id}: {len(labels_by_stem)} GT files "
                    f"(Patient = class {gt_class})"
                )
                t, v, counts = _add_task_frames(
                    V2_DATASET, task_id, labels_by_stem, gt_class
                )
                new_train.extend(t)
                new_val.extend(v)
                logger.info(
                    f"  task {task_id}: +{counts['added_train']} train, "
                    f"+{counts['added_val']} val, "
                    f"skipped(no_image={counts['skipped_no_image']}, "
                    f"no_boxes={counts['skipped_no_boxes']}, "
                    f"unknown_case={counts['skipped_unknown_case']}, "
                    f"collision={counts['skipped_collision']})"
                )

    # ------------------------------------------------------------ data.yaml
    train_paths = sorted(v1_train + new_train)
    val_paths = sorted(v1_val + new_val)
    data_yaml = {
        "path": str(V2_DATASET),
        "train": "images/train",
        "val": "images/val",
        "names": {TARGET_CLASS: TARGET_NAME},
    }
    (V2_DATASET / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    train_stems = {Path(p).stem for p in train_paths}
    val_stems = {Path(p).stem for p in val_paths}
    overlap = train_stems & val_stems
    if overlap:
        raise RuntimeError(
            f"train/val stem overlap ({len(overlap)} frames) — case split is broken. "
            f"Examples: {sorted(overlap)[:5]}"
        )

    write_manifest(
        V2_DATASET,
        convention="patient-only",
        cvat_task_ids=task_id_list,
        notes=(
            f"1-class Patient bootstrap dataset v2. v1 corpus + hand-corrections "
            f"from CVAT tasks {task_id_list}. Covers all 14 cases in the corpus. "
            f"Val case: {sorted(VAL_CASES)} (same as v0/v1)."
        ),
    )

    # Per-case summary
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
        f"patient_v2 dataset ready at {V2_DATASET}\n"
        f"  train: {len(train_paths)} frames\n"
        f"  val:   {len(val_paths)} frames\n"
        f"Train with:\n"
        f"  uv run python scripts/train_patient_v2.py"
    )


if __name__ == "__main__":
    app()
