"""Subset the already-labeled dataset and upload to CVAT for Patient labeling.

Replaces the prior "extract new sparse frames" approach. We now sample from
the existing labeled corpus so eventual merged labels stay frame-consistent:
every labeled frame will have both instrument and patient coverage.

Patient class definition: any visible facial tissue.

Bootstrap workflow:
  1. (this script) Pick ~30 frames/case from datasets/current → upload to CVAT.
  2. In CVAT, annotate Patient only on those frames. Instrument labels are
     intentionally NOT pre-loaded — keep the task focused on the new class
     and avoid scope creep on existing annotations.
  3. Export, train a 1-class patient detector.
  4. Use it to pseudo-label every frame in the labeled corpus, spot-check via
     FiftyOne, then merge patient lines into existing .txt label files.
  5. Train the unified multi-class model with Patient added to LABELS.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import DATA_DIR, DATASETS_DIR

load_dotenv(find_dotenv())

SHARE_ROOT = DATA_DIR  # /mnt/data/ent_cv -> /home/django/share in CVAT
CURRENT_DATASET = DATASETS_DIR / "current"
FRAMES_PER_CASE = 30

# Matches stems like "20251113_02_Part1_f018000"
FRAME_RE = re.compile(r"^(?P<case>.+?)_Part(?P<part>\d+)_f(?P<frame>\d+)$")

app = typer.Typer(add_completion=False)


def _evenly_spaced(items: list[Path], k: int) -> list[Path]:
    """Pick k items evenly spaced across items by index, endpoints included."""
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _group_by_case(image_paths: list[Path]) -> dict[str, list[Path]]:
    """Group images by case ID, sorted within each case by (part, frame)."""
    bucket: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    for p in image_paths:
        m = FRAME_RE.match(p.stem)
        if not m:
            logger.warning(f"  unparseable stem, skipping: {p.name}")
            continue
        bucket[m["case"]].append((int(m["part"]), int(m["frame"]), p))
    return {case: [p for _, _, p in sorted(rows)] for case, rows in bucket.items()}


@app.command()
def main(
    task_name: str = typer.Option("patient_sparse", help="CVAT task name."),
    frames_per_case: int = typer.Option(FRAMES_PER_CASE, help="Max frames per case."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    project_id: int = typer.Option(default=..., envvar="CVAT_PROJECT_ID"),
    dry_run: bool = typer.Option(False, help="Print sample plan without creating the task."),
):
    """Subset existing labeled frames and upload to CVAT for Patient annotation."""
    images_dir = CURRENT_DATASET / "images" / "train"
    if not images_dir.is_dir():
        raise typer.BadParameter(f"Missing images dir: {images_dir}")

    all_images = sorted(images_dir.glob("*.png"))
    logger.info(f"Found {len(all_images)} labeled images in {images_dir}")

    by_case = _group_by_case(all_images)
    logger.info(f"Grouped into {len(by_case)} cases")

    sampled: list[Path] = []
    for case in sorted(by_case):
        case_frames = by_case[case]
        picks = _evenly_spaced(case_frames, frames_per_case)
        sampled.extend(picks)
        logger.info(f"  {case}: sampled {len(picks)}/{len(case_frames)}")

    logger.info(f"Total sampled: {len(sampled)} frames across {len(by_case)} cases")

    if dry_run:
        logger.info("--dry-run set, skipping CVAT upload")
        return

    share_paths = sorted(str(p.relative_to(SHARE_ROOT)) for p in sampled)
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        task = client.tasks.create(spec={
            "name": task_name,
            "project_id": project_id,
        })
        logger.info(f"Created task '{task_name}' (ID: {task.id})")

        logger.info(f"Uploading {len(share_paths)} images via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)
        logger.info(f"Upload complete — {len(share_paths)} frames")

    logger.success(
        f"Task '{task_name}' (ID: {task.id}) ready with {len(share_paths)} frames.\n"
        f"  {len(by_case)} cases × up to {frames_per_case} frames/case\n"
        f"  Source: subset of {CURRENT_DATASET.name}/images/train"
    )


if __name__ == "__main__":
    app()
