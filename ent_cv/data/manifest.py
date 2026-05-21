"""Generate manifest.yaml for a dataset export.

Convention is provenance metadata — it must be passed in by whoever ran the
export. It is never inferred from box geometry or schema.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

VALID_CONVENTIONS = ["tool-only", "grasp-coupled", "patient-only", "grasp-coupled-with-patient"]
DEFAULT_CONVENTION = "grasp-coupled"


def _load_class_names(export_dir: Path) -> dict[int, str]:
    with (export_dir / "data.yaml").open() as f:
        cfg = yaml.safe_load(f)
    return {int(k): v for k, v in cfg["names"].items()}


def _labels_dir(export_dir: Path) -> Path:
    """Return the labels root that contains .txt files (anywhere underneath).

    Handles two layouts:
      - single-dir: labels/train/*.txt  (legacy v0, CVAT exports)
      - split-dir: labels/train/*.txt + labels/val/*.txt  (v1+)
    Callers should pass the returned path to _compute_stats, which walks
    recursively.
    """
    for c in (export_dir / "labels", export_dir / "labels" / "train"):
        if c.is_dir() and any(c.rglob("*.txt")):
            return c
    raise FileNotFoundError(f"no labels dir with .txt files under {export_dir}")


def _compute_stats(labels: Path) -> dict:
    """Walk ``labels`` recursively so split-dir layouts are counted together."""
    box_counts: Counter[int] = Counter()
    frames_with_ann = 0
    background = 0
    total_files = 0
    for p in labels.rglob("*.txt"):
        total_files += 1
        n = 0
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            box_counts[int(line.split()[0])] += 1
            n += 1
        if n > 0:
            frames_with_ann += 1
        else:
            background += 1
    return {
        "label_files_total": total_files,
        "frames_with_annotations": frames_with_ann,
        "frames_background": background,
        "boxes_total": int(sum(box_counts.values())),
        "boxes_per_class": {int(k): int(v) for k, v in sorted(box_counts.items())},
    }


def _case_id(stem: str) -> str:
    """Strip ``_f<frame>`` suffix to give the case+part identifier."""
    return stem.rsplit("_f", 1)[0]


def _case_set_from_file(split_file: Path) -> list[str]:
    cases: set[str] = set()
    if not split_file.exists():
        return []
    for line in split_file.read_text().splitlines():
        stem = Path(line.strip()).stem
        if stem:
            cases.add(_case_id(stem))
    return sorted(cases)


def _case_set_from_dir(images_dir: Path) -> list[str]:
    cases: set[str] = set()
    for p in images_dir.iterdir():
        if p.is_file() or p.is_symlink():
            cases.add(_case_id(p.stem))
    return sorted(cases)


def _split_info(export_dir: Path) -> dict | None:
    """Discover split membership from either sidecar txt files or dir layout."""
    train_file = export_dir / "train_split.txt"
    val_file = export_dir / "val_split.txt"
    if train_file.exists() and val_file.exists():
        return {
            "policy": "case-level",
            "train_cases": _case_set_from_file(train_file),
            "val_cases": _case_set_from_file(val_file),
        }
    train_dir = export_dir / "images" / "train"
    val_dir = export_dir / "images" / "val"
    if train_dir.is_dir() and val_dir.is_dir() and any(val_dir.iterdir()):
        return {
            "policy": "case-level",
            "train_cases": _case_set_from_dir(train_dir),
            "val_cases": _case_set_from_dir(val_dir),
        }
    return None


def write_manifest(
    export_dir: Path,
    convention: str,
    cvat_project_id: int | None = None,
    cvat_task_ids: list[int] | None = None,
    notes: str = "",
) -> Path:
    """Compute stats from the export's contents and write manifest.yaml.

    Returns the path to the written manifest file.
    """
    if convention not in VALID_CONVENTIONS:
        raise ValueError(f"convention must be one of {VALID_CONVENTIONS}, got {convention!r}")
    export_dir = Path(export_dir).resolve()
    if not export_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {export_dir}")

    classes = _load_class_names(export_dir)
    stats = _compute_stats(_labels_dir(export_dir))

    manifest = {
        "convention": convention,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "cvat_project_id": cvat_project_id,
            "cvat_task_ids": list(cvat_task_ids or []),
        },
        "classes_declared": classes,
        "classes_with_annotations": sorted(stats["boxes_per_class"].keys()),
        "stats": stats,
        "split": _split_info(export_dir),
        "notes": notes or None,
    }

    out = export_dir / "manifest.yaml"
    with out.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    return out
