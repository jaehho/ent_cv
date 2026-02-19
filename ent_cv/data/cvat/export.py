import zipfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from loguru import logger
from cvat_sdk import make_client
import typer

from ent_cv.config import DATASETS_DIR

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

app = typer.Typer()


@app.command()
def main(
    task_id: int = 12,
    host: str = "https://cvat.jaehho.com",
    port: int = 443,
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True),
    export_format: str = "Ultralytics YOLO Detection 1.0",
    output_dir: Optional[Path] = None,
    include_images: bool = True,
    delete_zip: bool = True,
):
    zip_path = DATASETS_DIR / "dataset.zip"

    logger.info(f"Connecting to CVAT at {host}:{port}...")
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        logger.info(f"Fetching task {task_id}...")
        task = client.tasks.retrieve(task_id)
        logger.info(f"Found task: '{task.name}'")

        if output_dir is None:
            output_dir = DATASETS_DIR / task.name

        logger.info(f"Exporting as '{export_format}' to: {zip_path}")
        if output_dir.exists() and any(output_dir.iterdir()):
            typer.confirm(f"'{output_dir}' is not empty. Overwrite?", abort=True)
        
        logger.info("Waiting for CVAT to prepare the export (this may take a second)...")
        task.export_dataset(
            format_name=export_format,
            filename=str(zip_path),
            include_images=include_images,
        )
        logger.info("Download complete!")

    logger.info(f"Extracting to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    logger.info("Extraction complete!")

    if delete_zip:
        zip_path.unlink()
        logger.info(f"Deleted zip: {zip_path}")

    logger.success(f"Done! Dataset is ready at: {output_dir}")


if __name__ == "__main__":
    app()