"""Create a CVAT task with all unlabeled frames for Empty Hand / Not Sure annotation.

One-off script — finds frames with empty label files in the combined dataset
and uploads them to CVAT via the share volume.
"""
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import DATA_DIR, DATASETS_DIR

load_dotenv(find_dotenv())

SHARE_ROOT = DATA_DIR  # /mnt/data/ent_cv -> /home/django/share
COMBINED_DIR = DATASETS_DIR / "dataset"

app = typer.Typer(add_completion=False)


@app.command()
def main(
    task_name: str = "unlabeled_empty_hand",
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    project_id: int = typer.Option(default=..., envvar="CVAT_PROJECT_ID"),
):
    labels_dir = COMBINED_DIR / "labels" / "train"
    images_dir = COMBINED_DIR / "images" / "train"

    # Find frames with empty label files
    empty_labels = sorted(
        p for p in labels_dir.glob("*.txt") if p.stat().st_size == 0
    )
    logger.info(f"Found {len(empty_labels)} unlabeled frames")

    # Verify corresponding images exist
    image_paths: list[Path] = []
    for lbl in empty_labels:
        img = images_dir / f"{lbl.stem}.png"
        if not img.exists():
            logger.warning(f"Missing image for {lbl.stem}, skipping")
            continue
        image_paths.append(img)

    logger.info(f"Matched {len(image_paths)} images to upload")

    # Show per-case breakdown
    cases: dict[str, int] = {}
    for p in image_paths:
        # Extract case from stem like 20251208_01_Part1_f018000
        parts = p.stem.split("_Part")
        case = parts[0] if parts else p.stem
        cases[case] = cases.get(case, 0) + 1
    for case, count in sorted(cases.items()):
        logger.info(f"  {case}: {count} frames")

    # Create CVAT task and upload via SHARE
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        task = client.tasks.create(spec={
            "name": task_name,
            "project_id": project_id,
        })
        logger.info(f"Created task '{task_name}' (ID: {task.id})")

        share_paths = [
            str(p.relative_to(SHARE_ROOT)) for p in image_paths
        ]
        logger.info(f"Uploading {len(share_paths)} images via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)
        task.fetch()
        logger.info(f"Upload complete — {task.size} frames")

    logger.success(
        f"Task '{task_name}' (ID: {task.id}) ready with {task.size} frames.\n"
        f"  Labels inherited from project {project_id}."
    )


if __name__ == "__main__":
    app()
