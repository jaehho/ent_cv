"""Upload extracted frames to a CVAT task via the share volume."""

from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import LABELS

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

app = typer.Typer()


@app.command()
def main(
    task_name: str = typer.Argument(..., help="CVAT task name to create."),
    images_dir: Path = typer.Argument(..., help="Directory of images to upload."),
    host: str = "https://cvat.jaehho.com",
    port: int = 443,
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True),
    share_root: Path = Path("/home/jaeho/ent_cv/data"),
    project_id: int | None = typer.Option(
        None, envvar="CVAT_PROJECT_ID",
        help="CVAT project ID. Labels are inherited from the project.",
    ),
):
    image_paths = sorted(images_dir.rglob("*.jpg")) + sorted(images_dir.rglob("*.png"))
    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        raise typer.Exit(1)
    logger.info(f"Found {len(image_paths)} images in {images_dir}")

    spec: dict = {"name": task_name}
    if project_id is not None:
        spec["project_id"] = project_id
    else:
        spec["labels"] = [{"name": name} for name in LABELS]

    logger.info(f"Connecting to CVAT at {host}:{port}...")
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        logger.info(f"Creating task '{task_name}'...")
        task = client.tasks.create(spec=spec)
        logger.info(f"Task created with ID: {task.id}")

        logger.info("Registering server-side images (this may take a sec)...")
        server_file_paths = [str(p.relative_to(share_root)) for p in image_paths]
        task.upload_data(server_file_paths, resource_type=ResourceType.SHARE)
        task.fetch()
        frame_count = task.size
        logger.info(f"Upload complete! Task has {frame_count} frames.")

    logger.success(f"Done! Task '{task_name}' (ID: {task.id}) is ready in CVAT.")


if __name__ == "__main__":
    app()