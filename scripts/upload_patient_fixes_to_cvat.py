"""Ship tagged FiftyOne samples to CVAT as a fixup task.

Why this exists: FiftyOne's in-App detection editor had save-related bugs
through 1.13.x and 1.14.x. CVAT is the source of truth for geometry edits in
this project, so we triage in FO (tagging is reliable) and round-trip the
actual fixes through CVAT.

Default selection: every sample tagged ``needs_fix`` in the
``ent_cv_patient_v0`` FiftyOne dataset (auto-applied to the 67 frames with no
pseudo-label, plus anything the user has tagged manually). The current on-disk
pseudo-labels are pre-loaded so existing boxes only need adjustment, not
redrawing.

The new CVAT task is created under the canonical 13-class project (env
``CVAT_PROJECT_ID``); use it normally, then run
``scripts/apply_patient_fixes.py <task-id>`` to overwrite the per-frame .txt
files in ``predictions/patient_v0_pseudo/labels/``.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
import fiftyone as fo
from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

FO_DATASET = "ent_cv_patient_v0"
PSEUDO_LABELS_DIR = DATA_DIR / "predictions" / "patient_v0_pseudo" / "labels"
SHARE_ROOT = DATA_DIR  # /mnt/data/ent_cv -> /home/django/share on the CVAT host


def _select_samples(ds: fo.Dataset, tag: str) -> list[fo.Sample]:
    """Return persistent-storage samples carrying ``tag``."""
    view = ds.match_tags(tag)
    return list(view)


def _build_annotations_zip(
    image_paths: list[Path],
    labels_by_stem: dict[str, str],
    zip_path: Path,
) -> int:
    """Build a CVAT YOLO import zip.

    All 13 canonical class names are listed in ``data.yaml`` so existing
    class-12 (Patient) annotations resolve correctly when CVAT maps them
    back into the project's label space. Frames with no pseudo-label
    produce no ``.txt`` (CVAT treats them as unannotated).
    """
    data_yaml = yaml.safe_dump(
        {
            "path": ".",
            "train": "train.txt",
            "names": dict(enumerate(LABELS)),
        },
        default_flow_style=False,
    )
    train_txt = "\n".join(f"images/train/{p.stem}.png" for p in image_paths) + "\n"

    n_with_labels = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml)
        zf.writestr("train.txt", train_txt)
        zf.writestr("images/train/.keep", "")
        for p in image_paths:
            lines = labels_by_stem.get(p.stem, "")
            if lines.strip():
                zf.writestr(f"labels/train/{p.stem}.txt", lines)
                n_with_labels += 1
    return n_with_labels


@app.command()
def main(
    tag: str = typer.Option(
        "needs_fix", help="FiftyOne tag to ship to CVAT."
    ),
    task_name: str = typer.Option(
        "patient_v0_fixes", help="Name for the new CVAT task."
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
    """Upload tagged FO frames to CVAT for Patient-box fixes."""
    if not fo.dataset_exists(FO_DATASET):
        raise typer.BadParameter(
            f"FiftyOne dataset '{FO_DATASET}' not found. Build it first with "
            f"scripts/fiftyone_patient_review.py."
        )
    ds = fo.load_dataset(FO_DATASET)
    samples = _select_samples(ds, tag)
    if not samples:
        raise typer.BadParameter(
            f"No samples tagged '{tag}' in '{FO_DATASET}'."
        )

    image_paths = sorted(Path(s.filepath) for s in samples)
    logger.info(f"Selected {len(image_paths)} samples tagged '{tag}'")

    # Sanity: all images must live under the SHARE root so CVAT can mount them.
    share_paths: list[str] = []
    for p in image_paths:
        try:
            share_paths.append(str(p.relative_to(SHARE_ROOT)))
        except ValueError as exc:
            raise typer.BadParameter(
                f"Image is outside the CVAT SHARE root: {p}"
            ) from exc

    labels_by_stem: dict[str, str] = {}
    n_preloaded = 0
    for p in image_paths:
        label_path = PSEUDO_LABELS_DIR / f"{p.stem}.txt"
        if label_path.exists():
            labels_by_stem[p.stem] = label_path.read_text()
            n_preloaded += 1
    logger.info(
        f"  pre-loading {n_preloaded} frames with existing Patient boxes "
        f"({len(image_paths) - n_preloaded} blank — likely your misses)"
    )

    if dry_run:
        logger.info("--dry-run set — skipping CVAT.")
        for p in image_paths[:10]:
            tags = "pre-loaded" if p.stem in labels_by_stem else "blank"
            logger.info(f"  [{tags:>10}]  {p.name}")
        if len(image_paths) > 10:
            logger.info(f"  ... and {len(image_paths) - 10} more")
        return

    with make_client(
        host=host, port=port, credentials=(username, password)
    ) as client:
        task = client.tasks.create(
            spec={"name": task_name, "project_id": project_id}
        )
        logger.info(f"Created task '{task_name}' (ID: {task.id})")

        logger.info(f"Uploading {len(share_paths)} frames via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)

        with tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False, prefix=f"cvat_{task_name}_"
        ) as tmp:
            zip_path = Path(tmp.name)
        try:
            n = _build_annotations_zip(image_paths, labels_by_stem, zip_path)
            logger.info(f"Importing {n} pre-loaded annotations...")
            task.import_annotations(
                format_name="Ultralytics YOLO Detection 1.0",
                filename=str(zip_path),
            )
        finally:
            zip_path.unlink(missing_ok=True)

    logger.success(
        f"Done. Fix in CVAT, then run:\n"
        f"    uv run python scripts/apply_patient_fixes.py {task.id}"
    )


if __name__ == "__main__":
    app()
