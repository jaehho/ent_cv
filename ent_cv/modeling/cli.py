"""Unified CLI entry point: ``ent-cv <command> [OPTIONS]``."""

import typer

from ent_cv.data import extract_frames, tile_frames
from ent_cv.data.cvat import (
    combine,
    export_manual,
    import_dataset,
    project,
    upload,
    upload_predictions,
)
from ent_cv.data.cvat import (
    export as cvat_export,
)
from ent_cv.data.sharepoint import scrape_and_download
from ent_cv.modeling import (
    batch,
    benchmark,
    clean,
    compare_models,
    export,
    postprocess,
    predict,
    prepare_dataset,
    sweep,
    train,
    tune,
    val,
)

app = typer.Typer(
    name="ent-cv",
    add_completion=False,
    no_args_is_help=True,
    help="ENT computer-vision modeling toolkit.",
)

# ── Core modeling commands (flat) ─────────────────────────────────────────────
app.command(name="train", help="Train a YOLO model.")(train.main)
app.command(name="predict", help="Run YOLO inference on a source.")(predict.main)
app.command(name="val", help="Validate a YOLO model on a dataset split.")(val.main)
app.command(name="postprocess", help="Post-process YOLO temporal detections.")(postprocess.main)
app.command(name="batch", help="Run a batch of modeling operations from YAML.")(batch.main)
app.command(name="compare", help="Compare trained models and rank by metrics.")(
    compare_models.main
)
app.command(name="tune", help="Run evolutionary hyperparameter tuning.")(tune.main)
app.command(name="sweep", help="Automated 3-phase training sweep.")(sweep.main)
app.command(name="benchmark", help="Benchmark a YOLO model across export formats.")(
    benchmark.main
)
app.command(name="export", help="Export a YOLO model to a target format.")(export.main)
app.command(name="prepare-dataset", help="Auto-split train/val for a YOLO dataset.")(
    prepare_dataset.main
)
app.command(name="clean", help="Remove prediction artifacts (frames, videos, labels).")(
    clean.main
)

# ── Data preparation sub-group ────────────────────────────────────────────────
data_app = typer.Typer(
    name="data",
    help="Data preparation commands.",
    no_args_is_help=True,
)

data_app.command(name="extract-frames", help="Extract frames from surgical videos.")(
    extract_frames.extract_frames
)
data_app.command(name="tile", help="Tile YOLO dataset into overlapping patches.")(
    tile_frames.main
)
data_app.command(name="download", help="Download videos from SharePoint.")(
    scrape_and_download.main
)

app.add_typer(data_app)

# ── CVAT integration sub-group ────────────────────────────────────────────────
cvat_app = typer.Typer(
    name="cvat",
    help="CVAT annotation tool integration.",
    no_args_is_help=True,
)

cvat_app.command(name="export", help="Export annotations from CVAT as YOLO dataset.")(
    cvat_export.main
)
cvat_app.command(name="import", help="Import a YOLO dataset into CVAT.")(import_dataset.main)
cvat_app.command(name="upload", help="Upload images to a CVAT task.")(upload.main)
cvat_app.command(name="upload-predictions", help="Upload prediction annotations to CVAT.")(
    upload_predictions.main
)
cvat_app.command(name="export-manual", help="Export manual annotations from a CVAT job.")(
    export_manual.main
)
cvat_app.command(
    name="setup-project",
    help="Create a CVAT project with canonical labels and migrate orphan tasks.",
)(project.main)
cvat_app.command(name="combine", help="Combine multiple YOLO datasets.")(combine.main)

app.add_typer(cvat_app)

if __name__ == "__main__":
    app()
