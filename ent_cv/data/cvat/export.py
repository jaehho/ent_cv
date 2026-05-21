"""Export a CVAT task as an Ultralytics YOLO dataset and write a manifest."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional
import zipfile

from cvat_sdk import make_client
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import DATASETS_DIR
from ent_cv.data.manifest import DEFAULT_CONVENTION, VALID_CONVENTIONS, write_manifest

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

app = typer.Typer()


@app.command()
def main(
    task_id: int = typer.Argument(..., help="CVAT task ID to export."),
    convention: str = typer.Option(
        DEFAULT_CONVENTION, "--convention",
        help=f"Annotation convention (provenance). One of: {', '.join(VALID_CONVENTIONS)}.",
    ),
    host: str = "https://cvat.jaehho.com",
    port: int = 443,
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True),
    export_format: str = "Ultralytics YOLO Detection 1.0",
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir",
        help="Where to extract. Default: datasets/exports/<convention>/<today>/",
    ),
    notes: str = typer.Option("", help="Free-text notes for the manifest."),
    include_images: bool = True,
    delete_zip: bool = True,
):
    if convention not in VALID_CONVENTIONS:
        raise typer.BadParameter(f"convention must be one of {VALID_CONVENTIONS}")

    if output_dir is None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        output_dir = DATASETS_DIR / "exports" / convention / today

    zip_path = DATASETS_DIR / "dataset.zip"

    logger.info(f"Connecting to CVAT at {host}:{port}...")
    cvat_project_id: int | None = None
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        task = client.tasks.retrieve(task_id)
        cvat_project_id = task.project_id
        logger.info(f"Task #{task_id}: '{task.name}' (project {cvat_project_id})")

        if output_dir.exists() and any(output_dir.iterdir()):
            typer.confirm(f"'{output_dir}' is not empty. Overwrite?", abort=True)

        logger.info(f"Exporting as '{export_format}'...")
        task.export_dataset(
            format_name=export_format,
            filename=str(zip_path),
            include_images=include_images,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    logger.info(f"Extracted to {output_dir}")

    if delete_zip:
        zip_path.unlink()

    manifest_path = write_manifest(
        output_dir,
        convention=convention,
        cvat_project_id=cvat_project_id,
        cvat_task_ids=[task_id],
        notes=notes,
    )
    logger.success(f"Done. Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    app()
