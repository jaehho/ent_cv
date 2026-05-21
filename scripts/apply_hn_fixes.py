"""Pull verified annotations from HN-mining CVAT tasks back to a labels dir.

Companion to ``mine_hard_negatives.py`` + ``create_hn_verify_task.py``. After
labeling all four bucket tasks (multi_patient, uncertain, silent, confident),
run this with the task IDs to fold every frame's final annotations into one
YOLO labels directory.

Per-frame behavior (across all tasks):
  - Frame has 1+ annotations in CVAT export   → write .txt with canonical
                                                class indices
  - Frame in task but NOT in export           → write empty .txt (explicit
                                                true negative — user opened
                                                the frame and saved with no
                                                boxes)
  - Frame in multiple tasks                   → fail loud (buckets should be
                                                disjoint by construction)

Output:
  predictions/full_v0_hn_verified/
    labels/<stem>.txt        YOLO normalized cxcywh, canonical class indices
    summary.txt              counts + class distribution
    manifest.yaml            task IDs, model, run params

Non-Patient classes are translated from the CVAT project's label IDs to the
canonical config.LABELS indices by NAME, so the project's internal ordering
doesn't have to match.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import zipfile

from cvat_sdk import make_client
from cvat_sdk.api_client.api.tasks_api import TasksApi
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

DEFAULT_OUT_DIR = DATA_DIR / "predictions" / "full_v0_hn_verified"
LABEL_NAME_TO_CANONICAL = {name: i for i, name in enumerate(LABELS)}


def _gt_class_map(export_dir: Path) -> dict[int, int]:
    """Build CVAT-class-index → canonical-class-index map from export data.yaml."""
    spec = yaml.safe_load((export_dir / "data.yaml").read_text())
    names = spec.get("names")
    if isinstance(names, dict):
        cvat_classes = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        cvat_classes = dict(enumerate(names))
    else:
        raise RuntimeError(f"unexpected names schema in {export_dir / 'data.yaml'}")

    out: dict[int, int] = {}
    unknown: list[str] = []
    for cvat_idx, name in cvat_classes.items():
        if name in LABEL_NAME_TO_CANONICAL:
            out[cvat_idx] = LABEL_NAME_TO_CANONICAL[name]
        else:
            unknown.append(name)
    if unknown:
        raise RuntimeError(
            f"CVAT export has labels not in config.LABELS: {unknown}. "
            f"Add to config.LABELS or stop labeling them."
        )
    return out


def _flatten_label_files(export_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in (export_dir / "labels").rglob("*.txt"):
        out[p.stem] = p
    return out


def _remap_lines(text: str, class_map: dict[int, int]) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cvat_cls = int(parts[0])
        canonical = class_map[cvat_cls]
        out.append(f"{canonical} " + " ".join(parts[1:]))
    return out


def _pull_task(
    client, task_id: int
) -> tuple[set[str], dict[str, list[str]]]:
    """Return (all_task_stems, stem -> canonical-class YOLO lines).

    stems in all_task_stems but not in the dict are explicit-clears.
    """
    tasks_api = TasksApi(client.api_client)
    task = client.tasks.retrieve(task_id)
    logger.info(f"Task {task_id}: '{task.name}' (project {task.project_id})")

    with tempfile.TemporaryDirectory(prefix=f"cvat_hn_{task_id}_") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "export.zip"
        task.export_dataset(
            format_name="Ultralytics YOLO Detection 1.0",
            filename=str(zip_path),
            include_images=False,
        )
        (meta, _) = tasks_api.retrieve_data_meta(id=task_id)
        all_task_stems = {Path(f.name).stem for f in meta.frames}

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        class_map = _gt_class_map(tmp)
        labels = _flatten_label_files(tmp)
        annotated: dict[str, list[str]] = {}
        for stem, path in labels.items():
            annotated[stem] = _remap_lines(path.read_text(), class_map)

    cleared = all_task_stems - annotated.keys()
    logger.info(
        f"  Task {task_id}: {len(all_task_stems)} frames | "
        f"annotated {len(annotated)} | explicit-clear {len(cleared)}"
    )
    return all_task_stems, annotated


def _write_summary(out_dir: Path, labels: dict[str, list[str]]) -> None:
    class_count: Counter[int] = Counter()
    frames_with_boxes = 0
    frames_empty = 0
    for lines in labels.values():
        if not lines:
            frames_empty += 1
            continue
        frames_with_boxes += 1
        for ln in lines:
            class_count[int(ln.split()[0])] += 1

    lines_out = [
        f"# Verified at {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"Total frames:        {len(labels)}",
        f"  with boxes:        {frames_with_boxes}",
        f"  true negatives:    {frames_empty}",
        f"Total boxes:         {sum(class_count.values())}",
        "Boxes per class:",
    ]
    for c in sorted(class_count):
        lines_out.append(f"  [{c:>2}] {LABELS[c]:<35} {class_count[c]}")
    (out_dir / "summary.txt").write_text("\n".join(lines_out) + "\n")


@app.command()
def main(
    task_ids: str = typer.Option(
        ..., help="Comma-separated CVAT task IDs (one per bucket)."
    ),
    out_dir: Path = typer.Option(
        DEFAULT_OUT_DIR, help="Output directory for verified labels."
    ),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    dry_run: bool = typer.Option(False, help="Print plan; don't write files."),
):
    """Combine verified labels from N CVAT tasks into one YOLO labels dir."""
    ids = [int(x) for x in task_ids.split(",") if x.strip()]
    if not ids:
        raise typer.BadParameter("--task-ids must include at least one ID")

    all_stems: dict[str, list[int]] = {}      # stem -> list of task_ids that contain it
    combined_labels: dict[str, list[str]] = {}

    with make_client(host=host, port=port, credentials=(username, password)) as client:
        for tid in ids:
            task_stems, annotated = _pull_task(client, tid)
            for stem in task_stems:
                all_stems.setdefault(stem, []).append(tid)
                if stem in annotated:
                    # Explicit-clear in another task would overwrite; we don't
                    # expect overlap so this is captured by the cross-task
                    # check below.
                    combined_labels[stem] = annotated[stem]
                else:
                    combined_labels.setdefault(stem, [])  # explicit-clear

    overlaps = {s: ts for s, ts in all_stems.items() if len(ts) > 1}
    if overlaps:
        raise typer.Exit(
            f"Buckets should be disjoint, but {len(overlaps)} frame(s) appear "
            f"in multiple tasks: {dict(list(overlaps.items())[:5])}"
        )

    logger.info(
        f"Combined plan: {len(combined_labels)} frames "
        f"({sum(1 for v in combined_labels.values() if v)} with boxes, "
        f"{sum(1 for v in combined_labels.values() if not v)} empty)"
    )

    if dry_run:
        logger.info("--dry-run set, not writing.")
        return

    labels_dir = out_dir / "labels"
    if labels_dir.exists():
        for p in labels_dir.glob("*.txt"):
            p.unlink()
    labels_dir.mkdir(parents=True, exist_ok=True)
    for stem, lines in combined_labels.items():
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    _write_summary(out_dir, combined_labels)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_task_ids": ids,
        "n_frames": len(combined_labels),
        "n_with_boxes": sum(1 for v in combined_labels.values() if v),
        "n_true_negative": sum(1 for v in combined_labels.values() if not v),
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    logger.success(
        f"Verified labels written: {out_dir}\n"
        f"  Frames:           {len(combined_labels)}\n"
        f"  With boxes:       {manifest['n_with_boxes']}\n"
        f"  True negatives:   {manifest['n_true_negative']}\n"
        f"  See summary.txt for class distribution."
    )


if __name__ == "__main__":
    app()
