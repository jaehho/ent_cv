"""Import a single YOLO dataset directory into a CVAT task.

Usage:
    uv run ent_cv/data/cvat/import.py /path/to/dataset

To combine multiple datasets first, run combine.py:
    uv run ent_cv/data/cvat/combine.py /path/to/ds1 /path/to/ds2 --output-dir /path/to/combined
    uv run ent_cv/data/cvat/import.py /path/to/combined
"""
import tempfile
import zipfile
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from loguru import logger
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
import typer
import yaml

from ent_cv.config import DATASETS_DIR, LABELS

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_annotations_zip(dataset_dir: Path, zip_path: Path) -> int:
    """Build a flat zip compatible with CVAT's 'Ultralytics YOLO Detection 1.0' importer.

    Expects the dataset to already be in the standard Ultralytics layout with
    flat labels/train/<stem>.txt files and a data.yaml + train.txt at the root.

    Zip layout:
        data.yaml
        train.txt
        images/train/.keep       (placeholder; datumaro requires the dir)
        labels/train/<stem>.txt

    Returns the number of label files written.
    """
    labels_root = dataset_dir / "labels"
    label_files = sorted(labels_root.rglob("*.txt"))
    if not label_files:
        logger.warning(f"No label .txt files found under {labels_root}")

    stem_map: dict[str, Path] = {}
    for lf in label_files:
        if lf.stem in stem_map:
            logger.warning(f"Duplicate stem '{lf.stem}' – keeping first occurrence")
        else:
            stem_map[lf.stem] = lf

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    # Rebuild train.txt from the actual label stems so it matches what's in the zip
    train_txt_lines = "\n".join(f"images/train/{stem}.png" for stem in stem_map) + "\n"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(data_yaml, "data.yaml")
        zf.writestr("train.txt", train_txt_lines)
        zf.writestr("images/train/.keep", "")
        for stem, lf in stem_map.items():
            zf.write(lf, f"labels/train/{stem}.txt")

    logger.info(f"Annotations zip: {len(stem_map)} label files -> {zip_path}")
    return len(stem_map)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    dataset_dir: Path = typer.Argument(
        DATASETS_DIR / "combined_new",
        help="Dataset directory in Ultralytics YOLO layout (images/train/, labels/train/, data.yaml).",
    ),
    host: str = "https://cvat.jaehho.com",
    port: int = 443,
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    task_name: str = typer.Option("", help="Task name; defaults to the dataset directory name."),
    import_format: str = "Ultralytics YOLO Detection 1.0",
    delete_tmp_zip: bool = True,
):
    if not task_name:
        task_name = dataset_dir.name

    # --------------------------------------------------------- collect images
    images_root = dataset_dir / "images"
    image_paths = sorted(images_root.rglob("*.jpg")) + sorted(images_root.rglob("*.png"))
    if not image_paths:
        logger.error(f"No images found under {images_root}")
        raise typer.Exit(1)
    logger.info(f"Found {len(image_paths)} images under {images_root}")

    # --------------------------------------------------------- build CVAT labels from data.yaml
    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        logger.error(f"data.yaml not found in {dataset_dir}")
        raise typer.Exit(1)
    with data_yaml_path.open() as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    label_list = names if isinstance(names, list) else [names[k] for k in sorted(names)]

    if set(label_list) != set(LABELS):
        logger.warning(
            f"data.yaml classes differ from canonical LABELS.\n"
            f"  data.yaml : {label_list}\n"
            f"  config    : {list(LABELS)}\n"
            "Consider running combine.py with --label-aliases to normalise first."
        )

    cvat_labels = [{"name": name} for name in label_list]

    # --------------------------------------------------------- CVAT
    logger.info(f"Connecting to CVAT at {host}:{port}...")
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        logger.info(f"Creating task '{task_name}'...")
        task = client.tasks.create(spec={"name": task_name, "labels": cvat_labels})
        logger.info(f"Task created with ID: {task.id}")

        logger.info(f"Uploading {len(image_paths)} images (local)...")
        task.upload_data(
            resources=[str(p) for p in image_paths],
            resource_type=ResourceType.LOCAL,
        )
        task.fetch()
        logger.info(f"Upload complete! Task has {task.size} frames.")

        with tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False, prefix="cvat_annotations_"
        ) as tmp:
            annotations_zip = Path(tmp.name)

        logger.info(f"Building annotations zip at {annotations_zip}...")
        n_annotations = _build_annotations_zip(dataset_dir, annotations_zip)

        logger.info(f"Importing annotations (format: '{import_format}')...")
        task.import_annotations(format_name=import_format, filename=str(annotations_zip))
        logger.info("Annotation import complete!")

        if delete_tmp_zip:
            annotations_zip.unlink()
            logger.debug(f"Deleted temp zip: {annotations_zip}")

    logger.success(
        f"Done! Task '{task_name}' (ID: {task.id}) is ready in CVAT "
        f"with {task.size} frames and {n_annotations} annotated frames."
    )


if __name__ == "__main__":
    app()
