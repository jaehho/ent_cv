"""Create a combined hard-negative + silent-frame verify task in CVAT.

Reads ``predictions/<run_name>/selected/{images,preload}/`` produced by
``mine_hard_negatives.py`` and uploads to CVAT. Bucket A frames come with
model predictions pre-loaded so the annotator can quickly delete/correct FPs.
Bucket B frames (silent — no model detections) start blank for fresh
labeling of model-silent moments (Empty Hand, Drill, etc).

The CVAT project must have all 13 labels from config.LABELS. Preloaded class
indices are canonical (config.LABELS index) and translated by NAME to the
project's label IDs at upload time. If any label is missing, fail loud with
the offending name.

After human review, pull corrections back into a YOLO label dir with
scripts/apply_hn_fixes.py (separate script).
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.api_client.api.labels_api import LabelsApi
from cvat_sdk.api_client.api.tasks_api import TasksApi
from cvat_sdk.api_client.model.labeled_shape_request import LabeledShapeRequest
from cvat_sdk.api_client.model.patched_labeled_data_request import (
    PatchedLabeledDataRequest,
)
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import DATA_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

SHARE_ROOT = DATA_DIR
DEFAULT_RUN_DIR = DATA_DIR / "predictions" / "full_v0_hn_candidates"
BUCKETS = ("multi_patient", "uncertain", "confident", "silent")


def _build_label_map(labels_api: LabelsApi, project_id: int) -> dict[str, int]:
    out: dict[str, int] = {}
    page = 1
    while True:
        (paginated, _) = labels_api.list(project_id=project_id, page=page)
        for lbl in paginated.results:
            out[lbl.name] = lbl.id
        if getattr(paginated, "next", None) is None:
            return out
        page += 1


def _frame_info_by_stem(tasks_api: TasksApi, task_id: int) -> dict[str, tuple[int, int, int]]:
    (meta, _) = tasks_api.retrieve_data_meta(id=task_id)
    out: dict[str, tuple[int, int, int]] = {}
    for i, f in enumerate(meta.frames):
        out[Path(f.name).stem] = (i, int(f.width), int(f.height))
    return out


def _build_shapes(
    preload_dir: Path,
    label_map: dict[str, int],
    frame_info: dict[str, tuple[int, int, int]],
) -> tuple[list[LabeledShapeRequest], Counter[str], list[str]]:
    """Parse YOLO preload .txt and emit LabeledShapeRequest per box.

    Returns (shapes, per_class_count, missing_classes).
    """
    shapes: list[LabeledShapeRequest] = []
    per_class: Counter[str] = Counter()
    missing: set[str] = set()

    for txt in sorted(preload_dir.glob("*.txt")):
        stem = txt.stem
        if stem not in frame_info:
            logger.warning(f"  preload for unknown frame: {stem} (skipping)")
            continue
        idx, w, h = frame_info[stem]
        for raw in txt.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) != 5:
                continue
            cls_canonical = int(parts[0])
            name = LABELS[cls_canonical]
            if name not in label_map:
                missing.add(name)
                continue
            cx, cy, bw, bh = (float(x) for x in parts[1:])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            shapes.append(
                LabeledShapeRequest(
                    type="rectangle", occluded=False, outside=False,
                    z_order=0, rotation=0.0, points=[x1, y1, x2, y2],
                    frame=idx, label_id=label_map[name], group=0,
                    source="auto", attributes=[],
                )
            )
            per_class[name] += 1
    return shapes, per_class, sorted(missing)


@app.command()
def main(
    bucket: str = typer.Option(
        ..., help=f"Bucket to upload. One of: {', '.join(BUCKETS)}."
    ),
    run_dir: Path = typer.Option(
        DEFAULT_RUN_DIR, help="Mining output root (parent of selected/)."
    ),
    task_name: str = typer.Option(
        "", help="CVAT task name. Defaults to full_v0_<bucket>_verify_<date>."
    ),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    project_id: int = typer.Option(default=..., envvar="CVAT_PROJECT_ID"),
    dry_run: bool = typer.Option(False, help="Skip CVAT upload, print plan."),
):
    """Upload one mining bucket to CVAT as a focused verify task."""
    if bucket not in BUCKETS:
        raise typer.BadParameter(f"--bucket must be one of {BUCKETS}, got {bucket!r}")

    selected_dir = run_dir / "selected" / bucket
    images_dir = selected_dir / "images"
    preload_dir = selected_dir / "preload"
    if not images_dir.is_dir():
        raise typer.BadParameter(f"images dir missing: {images_dir}")
    if not preload_dir.is_dir():
        raise typer.BadParameter(f"preload dir missing: {preload_dir}")

    if not task_name:
        task_name = f"full_v0_{bucket}_verify_{date.today().isoformat()}"

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise typer.BadParameter(f"no images in {images_dir}")

    preload_stems = {p.stem for p in preload_dir.glob("*.txt")}
    with_preload = [p for p in image_paths if p.stem in preload_stems]
    blank = [p for p in image_paths if p.stem not in preload_stems]

    logger.info(
        f"Bucket '{bucket}' upload plan:\n"
        f"  total:       {len(image_paths)}\n"
        f"  preloaded:   {len(with_preload)}\n"
        f"  blank:       {len(blank)}"
    )

    # Resolve symlinks to the canonical share-relative path.
    share_paths: list[str] = []
    for p in image_paths:
        resolved = p.resolve()
        try:
            share_paths.append(str(resolved.relative_to(SHARE_ROOT)))
        except ValueError as exc:
            raise typer.BadParameter(
                f"resolved frame is outside SHARE_ROOT: {resolved}"
            ) from exc
    share_paths.sort()

    if dry_run:
        logger.info("--dry-run set, skipping CVAT calls.")
        logger.info(f"Would create task '{task_name}' in project {project_id}.")
        for p in share_paths[:5]:
            logger.info(f"  upload: {p}")
        if len(share_paths) > 5:
            logger.info(f"  ... + {len(share_paths) - 5} more")
        logger.info(
            f"Would push preload shapes for {len(with_preload)} frames "
            f"(from {len(list(preload_dir.glob('*.txt')))} .txt files)."
        )
        return

    with make_client(host=host, port=port, credentials=(username, password)) as client:
        labels_api = LabelsApi(client.api_client)
        tasks_api = TasksApi(client.api_client)

        label_map = _build_label_map(labels_api, project_id)
        missing_in_project = [n for n in LABELS if n not in label_map]
        if missing_in_project:
            raise typer.Exit(
                f"CVAT project {project_id} is missing labels needed for this task: "
                f"{missing_in_project}.\nAvailable: {sorted(label_map)}"
            )
        logger.info(f"CVAT project {project_id} has all 13 LABELS.")

        task = client.tasks.create(spec={"name": task_name, "project_id": project_id})
        logger.info(f"Created task '{task_name}' (ID: {task.id})")
        logger.info(f"Uploading {len(share_paths)} frames via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)

        frame_info = _frame_info_by_stem(tasks_api, task.id)
        if len(frame_info) != len(share_paths):
            logger.warning(
                f"Frame count mismatch: uploaded {len(share_paths)}, "
                f"CVAT reports {len(frame_info)}"
            )

        shapes, per_class, missing = _build_shapes(preload_dir, label_map, frame_info)
        if missing:
            raise typer.Exit(
                f"Preload references labels not in CVAT project: {missing}"
            )
        if shapes:
            tasks_api.partial_update_annotations(
                action="create",
                id=task.id,
                patched_labeled_data_request=PatchedLabeledDataRequest(shapes=shapes),
            )

    logger.success(
        f"Task '{task_name}' (ID: {task.id}) ready.\n"
        f"  Bucket:           {bucket}\n"
        f"  Frames uploaded:  {len(share_paths)}\n"
        f"  Preloaded:        {len(with_preload)} ({len(shapes)} shapes)\n"
        f"  Blank:            {len(blank)}\n"
        f"  Per-class preload: {dict(per_class)}\n"
        f"After review, pull corrections with apply_hn_fixes.py --task-id {task.id}"
    )


if __name__ == "__main__":
    app()
