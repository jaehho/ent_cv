"""Create a CVAT task for hand-labeling a held-out Patient test set.

Goal: get a real precision/recall number for the current patient detector's
pseudo-labels. The training val set never generalizes as a quality bar — we
need cases the detector has never seen.

Default selection: 3 unseen cases × 30 evenly-spaced frames = 90 frames.
Pseudo-labels from --pseudo-dir are pre-loaded as initial annotations so
labeling is mostly "adjust + accept" instead of "draw from scratch". To stay
honest, hide predictions on each frame at least once before touching them —
pre-loaded boxes anchor reviewers toward what's already drawn.

You hand-label/correct Patient in CVAT, then run scripts/eval_pseudo_labels.py
with the task ID to compute pseudo-label P/R against this test set.

Pre-loading goes via the cvat-sdk API directly (POST shapes per frame), not
through CVAT's YOLO importer. The YOLO importer needs the image bytes present
under CVAT's data dir; SHARE-mounted images only exist by path reference and
trip the importer ("Can't find image info for ..."). Pushing shapes by frame
index sidesteps that.

Version bookkeeping: when you retrain (v1 → v2 ...), bump DEFAULT_PSEUDO_LABELS_DIR
and DEFAULT_SEEN_CASES so the next run targets the new model's blind spots.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

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

from ent_cv.config import DATA_DIR, DATASETS_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

SHARE_ROOT = DATA_DIR
CURRENT_DATASET = DATASETS_DIR / "current"
FRAME_RE = re.compile(r"^(?P<case>.+?)_Part(?P<part>\d+)_f(?P<frame>\d+)$")
PATIENT_CLASS = LABELS.index("Patient")

# Defaults track the latest patient detector. Override via --pseudo-dir /
# --seen-cases when sampling against an older model.
DEFAULT_PSEUDO_LABELS_DIR = DATA_DIR / "predictions" / "patient_v1_pseudo" / "labels"
# patient_v1 trained on these 8 cases (v0's 5 + 20251210_0{1,2,3}). Anything
# outside this set is fair game for the held-out test set.
DEFAULT_SEEN_CASES = (
    "20251113_02,20251124_01,20251204_01,20251208_01,20251208_02,"
    "20251210_01,20251210_02,20251210_03"
)


from typing import TypeVar
_T = TypeVar("_T")


def _evenly_spaced(items: list[_T], k: int) -> list[_T]:
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _group_by_case(image_paths: list[Path]) -> dict[str, list[Path]]:
    bucket: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    for p in image_paths:
        m = FRAME_RE.match(p.stem)
        if not m:
            continue
        bucket[m["case"]].append((int(m["part"]), int(m["frame"]), p))
    return {case: [p for _, _, p in sorted(rows)] for case, rows in bucket.items()}


def _get_project_label_id(labels_api: LabelsApi, project_id: int, name: str) -> int:
    page = 1
    while True:
        (paginated, _) = labels_api.list(project_id=project_id, page=page)
        for lbl in paginated.results:
            if lbl.name == name:
                return lbl.id
        if getattr(paginated, "next", None) is None:
            raise RuntimeError(
                f"Label '{name}' not found in project {project_id}"
            )
        page += 1


def _frame_info_by_stem(tasks_api: TasksApi, task_id: int) -> dict[str, tuple[int, int, int]]:
    """Return {stem: (frame_index, width, height)} for every frame in the task."""
    (meta, _) = tasks_api.retrieve_data_meta(id=task_id)
    out: dict[str, tuple[int, int, int]] = {}
    for i, f in enumerate(meta.frames):
        stem = Path(f.name).stem
        out[stem] = (i, int(f.width), int(f.height))
    return out


def _push_pseudo_labels(
    client,
    task_id: int,
    project_id: int,
    labels_by_stem: dict[str, str],
) -> tuple[int, int]:
    """Push pseudo-labels as Patient rectangles. Returns (shapes_pushed, frames_with_shapes)."""
    tasks_api = TasksApi(client.api_client)
    labels_api = LabelsApi(client.api_client)
    patient_id = _get_project_label_id(labels_api, project_id, "Patient")

    frame_info = _frame_info_by_stem(tasks_api, task_id)
    logger.info(f"  task has {len(frame_info)} frames in metadata")

    shapes: list[LabeledShapeRequest] = []
    frames_touched: set[str] = set()
    skipped_unmatched = 0
    for stem, text in labels_by_stem.items():
        if stem not in frame_info:
            skipped_unmatched += 1
            continue
        idx, w, h = frame_info[stem]
        for raw in text.strip().splitlines():
            parts = raw.split()
            if len(parts) != 5 or int(parts[0]) != PATIENT_CLASS:
                continue
            cx, cy, bw, bh = (float(x) for x in parts[1:])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            shapes.append(
                LabeledShapeRequest(
                    type="rectangle",
                    occluded=False,
                    outside=False,
                    z_order=0,
                    rotation=0.0,
                    points=[x1, y1, x2, y2],
                    frame=idx,
                    label_id=patient_id,
                    group=0,
                    source="auto",
                    attributes=[],
                )
            )
            frames_touched.add(stem)

    if skipped_unmatched:
        logger.warning(
            f"  {skipped_unmatched} pseudo-label stems didn't match any task frame"
        )
    if not shapes:
        return (0, 0)

    tasks_api.partial_update_annotations(
        action="create",
        id=task_id,
        patched_labeled_data_request=PatchedLabeledDataRequest(shapes=shapes),
    )
    return (len(shapes), len(frames_touched))


def _pseudo_labels_for_stems(
    stems: list[str], pseudo_dir: Path
) -> dict[str, str]:
    out: dict[str, str] = {}
    for stem in stems:
        p = pseudo_dir / f"{stem}.txt"
        if p.exists():
            out[stem] = p.read_text()
    return out


def _parse_seen_cases(spec: str) -> set[str]:
    return {c.strip() for c in spec.split(",") if c.strip()}


@app.command()
def main(
    task_name: str = typer.Option("patient_v1_test_set", help="CVAT task name (new task only)."),
    task_id: int | None = typer.Option(
        None, "--task-id",
        help="Existing task ID to (re-)push pseudo-labels to. Skips sampling + upload.",
    ),
    n_cases: int = typer.Option(3, help="Number of unseen cases to sample."),
    frames_per_case: int = typer.Option(30, help="Frames sampled per case."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    project_id: int = typer.Option(default=..., envvar="CVAT_PROJECT_ID"),
    pseudo_dir: Path = typer.Option(
        DEFAULT_PSEUDO_LABELS_DIR,
        help="Pseudo-label directory to pre-load from. Default is the latest patient model.",
    ),
    seen_cases: str = typer.Option(
        DEFAULT_SEEN_CASES,
        help="Comma-separated cases the patient detector has trained on. Held-out test set is sampled from outside this set.",
    ),
    preload_pseudo: bool = typer.Option(
        True, "--preload-pseudo/--no-preload-pseudo",
        help="Pre-load patient pseudo-labels as initial annotations.",
    ),
    dry_run: bool = typer.Option(False, help="Print plan without touching CVAT."),
):
    """Sample frames from unseen cases and ship them to CVAT for Patient labeling."""
    if not pseudo_dir.is_dir():
        raise typer.BadParameter(
            f"--pseudo-dir not found: {pseudo_dir}. Generate with "
            f"scripts/pseudo_label_patient.py first."
        )
    seen = _parse_seen_cases(seen_cases)
    logger.info(
        f"Pseudo-dir: {pseudo_dir}\n"
        f"Seen cases ({len(seen)}): {sorted(seen)}"
    )

    # ----------------------------------------------------------- retry mode
    if task_id is not None:
        if not preload_pseudo:
            raise typer.BadParameter("--task-id only makes sense with pre-loading.")
        with make_client(
            host=host, port=port, credentials=(username, password)
        ) as client:
            task = client.tasks.retrieve(task_id)
            logger.info(f"Retry mode: task {task_id} ('{task.name}')")
            tasks_api = TasksApi(client.api_client)
            stems = list(_frame_info_by_stem(tasks_api, task_id))
            labels_by_stem = _pseudo_labels_for_stems(stems, pseudo_dir)
            logger.info(
                f"  pseudo-labels available for {len(labels_by_stem)}/{len(stems)} frames"
            )
            if dry_run:
                logger.info("--dry-run set — skipping CVAT.")
                return
            n_shapes, n_frames = _push_pseudo_labels(
                client, task_id, project_id, labels_by_stem
            )
        logger.success(
            f"Pushed {n_shapes} pre-loaded Patient boxes across {n_frames} frames "
            f"to task {task_id}.\n"
            f"Next: label/correct in CVAT, then run:\n"
            f"    uv run python scripts/eval_pseudo_labels.py {task_id} "
            f"--pseudo-dir {pseudo_dir}"
        )
        return

    # ----------------------------------------------------------- new task
    images_dir = CURRENT_DATASET / "images" / "train"
    all_images = sorted(images_dir.glob("*.png"))
    by_case = _group_by_case(all_images)

    unseen = sorted(set(by_case) - seen)
    if not unseen:
        raise typer.BadParameter(
            "No unseen cases left — every case appears in --seen-cases. "
            "Add new cases to the corpus or relax the seen set."
        )
    logger.info(f"Unseen cases ({len(unseen)}): {unseen}")

    if n_cases > len(unseen):
        logger.warning(
            f"Asked for {n_cases} cases but only {len(unseen)} unseen — using all."
        )
        n_cases = len(unseen)

    # Spread picks across the sorted list rather than taking the first N —
    # unseen cases are date-sorted (YYYYMMDD_NN), and a contiguous prefix would
    # all come from the same recording day. We want temporal diversity.
    test_cases = _evenly_spaced(unseen, n_cases)
    sampled: list[Path] = []
    for case in test_cases:
        picks = _evenly_spaced(by_case[case], frames_per_case)
        sampled.extend(picks)
        logger.info(f"  {case}: sampled {len(picks)}/{len(by_case[case])}")

    logger.info(
        f"Total: {len(sampled)} frames across {n_cases} unseen cases "
        f"({', '.join(test_cases)})"
    )

    labels_by_stem: dict[str, str] = {}
    if preload_pseudo:
        labels_by_stem = _pseudo_labels_for_stems([p.stem for p in sampled], pseudo_dir)
        logger.info(
            f"Pre-loading {len(labels_by_stem)} pseudo-label files "
            f"({len(sampled) - len(labels_by_stem)} frames will start blank)"
        )
    else:
        logger.info("Pre-loading disabled — every frame starts blank.")

    if dry_run:
        logger.info("--dry-run set — skipping CVAT.")
        return

    share_paths = sorted(str(p.relative_to(SHARE_ROOT)) for p in sampled)
    with make_client(
        host=host, port=port, credentials=(username, password)
    ) as client:
        task = client.tasks.create(
            spec={"name": task_name, "project_id": project_id}
        )
        logger.info(f"Created task '{task_name}' (ID: {task.id})")
        logger.info(f"Uploading {len(share_paths)} frames via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)

        n_shapes = n_frames = 0
        if preload_pseudo and labels_by_stem:
            n_shapes, n_frames = _push_pseudo_labels(
                client, task.id, project_id, labels_by_stem
            )

    logger.success(
        f"Task '{task_name}' (ID: {task.id}) ready.\n"
        f"  Cases: {test_cases}\n"
        f"  Frames: {len(share_paths)}\n"
        f"  Pre-loaded: {n_shapes} boxes across {n_frames} frames\n"
        f"Important: review each frame at least once with predictions hidden\n"
        f"so the pre-loaded boxes don't anchor your judgment.\n"
        f"Then run:\n"
        f"    uv run python scripts/eval_pseudo_labels.py {task.id} "
        f"--pseudo-dir {pseudo_dir}"
    )


if __name__ == "__main__":
    app()
