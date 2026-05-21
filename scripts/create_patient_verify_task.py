"""Create a CVAT task with v_N pseudo-label frames that need human verification.

Domain constraint: every frame has exactly one Patient — or zero, if the
patient isn't visible. Anything else is wrong:
  - 0 boxes → either a true negative (patient out of frame) OR a detector miss
  - 2+ boxes → always wrong; only one patient exists per frame

This script finds those suspect frames and uploads them to CVAT with the
current pseudo-labels pre-loaded (so multi-box frames show the extras for
deletion). After verifying, run scripts/apply_patient_verify_fixes.py to pull
corrections back into the pseudo directory.

Why two scripts (verify vs. test_set): create_patient_test_set.py samples
N cases × M frames for a held-out generalization measurement. This script
samples specific suspect frames from across the whole corpus for a quality
clean-up — different selection strategy.
"""
from __future__ import annotations

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

from ent_cv.config import DATA_DIR, DATASETS_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

SHARE_ROOT = DATA_DIR
DEFAULT_IMAGES_DIR = DATASETS_DIR / "current" / "images" / "train"
DEFAULT_PSEUDO_DIR = DATA_DIR / "predictions" / "patient_v2_pseudo" / "labels"
PATIENT_CLASS = LABELS.index("Patient")


def _find_suspect_frames(
    images_dir: Path, pseudo_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Return (empty_frames, multi_box_frames) — images without a pseudo .txt
    and images whose pseudo .txt has ≥2 lines respectively."""
    empty: list[Path] = []
    multi: list[Path] = []
    for img in sorted(images_dir.glob("*.png")):
        txt = pseudo_dir / f"{img.stem}.txt"
        if not txt.exists():
            empty.append(img)
            continue
        lines = [ln for ln in txt.read_text().splitlines() if ln.strip()]
        if len(lines) >= 2:
            multi.append(img)
    return empty, multi


def _get_project_label_id(labels_api: LabelsApi, project_id: int, name: str) -> int:
    page = 1
    while True:
        (paginated, _) = labels_api.list(project_id=project_id, page=page)
        for lbl in paginated.results:
            if lbl.name == name:
                return lbl.id
        if getattr(paginated, "next", None) is None:
            raise RuntimeError(f"Label '{name}' not found in project {project_id}")
        page += 1


def _frame_info_by_stem(
    tasks_api: TasksApi, task_id: int
) -> dict[str, tuple[int, int, int]]:
    (meta, _) = tasks_api.retrieve_data_meta(id=task_id)
    out: dict[str, tuple[int, int, int]] = {}
    for i, f in enumerate(meta.frames):
        out[Path(f.name).stem] = (i, int(f.width), int(f.height))
    return out


def _push_pseudo_labels(
    client,
    task_id: int,
    project_id: int,
    labels_by_stem: dict[str, str],
) -> tuple[int, int]:
    """Push existing pseudo-labels as Patient rectangles. Returns (shapes, frames)."""
    tasks_api = TasksApi(client.api_client)
    labels_api = LabelsApi(client.api_client)
    patient_id = _get_project_label_id(labels_api, project_id, "Patient")
    frame_info = _frame_info_by_stem(tasks_api, task_id)

    shapes: list[LabeledShapeRequest] = []
    frames_touched: set[str] = set()
    for stem, text in labels_by_stem.items():
        if stem not in frame_info:
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
                    type="rectangle", occluded=False, outside=False,
                    z_order=0, rotation=0.0, points=[x1, y1, x2, y2],
                    frame=idx, label_id=patient_id, group=0,
                    source="auto", attributes=[],
                )
            )
            frames_touched.add(stem)
    if not shapes:
        return (0, 0)
    tasks_api.partial_update_annotations(
        action="create",
        id=task_id,
        patched_labeled_data_request=PatchedLabeledDataRequest(shapes=shapes),
    )
    return (len(shapes), len(frames_touched))


@app.command()
def main(
    pseudo_dir: Path = typer.Option(
        DEFAULT_PSEUDO_DIR,
        help="Pseudo-label directory whose suspect frames to verify.",
    ),
    images_dir: Path = typer.Option(
        DEFAULT_IMAGES_DIR, help="Source image directory."
    ),
    task_name: str = typer.Option(
        "patient_v2_verify", help="CVAT task name."
    ),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    project_id: int = typer.Option(default=..., envvar="CVAT_PROJECT_ID"),
    dry_run: bool = typer.Option(False, help="Print plan without touching CVAT."),
):
    """Find !=1-box pseudo-label frames and ship them to CVAT for verification."""
    if not pseudo_dir.is_dir():
        raise typer.BadParameter(f"--pseudo-dir not found: {pseudo_dir}")
    if not images_dir.is_dir():
        raise typer.BadParameter(f"--images-dir not found: {images_dir}")

    empty, multi = _find_suspect_frames(images_dir, pseudo_dir)
    suspect = sorted(empty + multi, key=lambda p: p.name)
    logger.info(
        f"Suspect frames in {pseudo_dir}:\n"
        f"  empty (no pseudo box):     {len(empty)}\n"
        f"  multi-box (≥2 pseudos):    {len(multi)}\n"
        f"  total to verify:           {len(suspect)}"
    )
    if not suspect:
        logger.success("No suspect frames — pseudo dir is clean.")
        return

    # Pre-load only for multi-box frames so the user can see and delete extras.
    # Empty frames start blank — the user adds a box only if patient is visible.
    multi_set = set(multi)
    labels_by_stem: dict[str, str] = {
        img.stem: (pseudo_dir / f"{img.stem}.txt").read_text() for img in multi
    }

    if dry_run:
        logger.info("--dry-run: skipping CVAT.")
        for img in suspect[:15]:
            kind = "multi" if img in multi_set else "empty"
            logger.info(f"  [{kind:>5}] {img.name}")
        if len(suspect) > 15:
            logger.info(f"  ... and {len(suspect) - 15} more")
        return

    share_paths = sorted(str(p.relative_to(SHARE_ROOT)) for p in suspect)
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        task = client.tasks.create(spec={"name": task_name, "project_id": project_id})
        logger.info(f"Created task '{task_name}' (ID: {task.id})")
        logger.info(f"Uploading {len(share_paths)} frames via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)
        n_shapes, n_frames = _push_pseudo_labels(
            client, task.id, project_id, labels_by_stem
        )

    logger.success(
        f"Task '{task_name}' (ID: {task.id}) ready.\n"
        f"  Frames:                    {len(share_paths)}\n"
        f"  Empty (decide add or skip): {len(empty)}\n"
        f"  Multi-box (delete extras): {len(multi)}\n"
        f"  Pre-loaded:                {n_shapes} boxes across {n_frames} frames\n"
        f"Rule: every frame ends with 0 boxes (true negative) or 1 box (patient).\n"
        f"After verifying, pull corrections back with:\n"
        f"  uv run python scripts/apply_patient_verify_fixes.py "
        f"--task-id {task.id} --pseudo-dir {pseudo_dir}"
    )


if __name__ == "__main__":
    app()
