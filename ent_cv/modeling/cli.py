"""Unified CLI entry point: ``ent-cv <command> [OPTIONS]``."""

import typer

from ent_cv.modeling import (
    batch,
    benchmark,
    compare_models,
    export,
    postprocess,
    predict,
    prepare_dataset,
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

# Register each module's main() directly as a flat command (no nesting).
app.command(name="train", help="Train a YOLO model.")(train.main)
app.command(name="predict", help="Run YOLO inference on a source.")(predict.main)
app.command(name="val", help="Validate a YOLO model on a dataset split.")(val.main)
app.command(name="tune", help="Run evolutionary hyperparameter tuning.")(tune.main)
app.command(name="benchmark", help="Benchmark a YOLO model across export formats.")(benchmark.main)
app.command(name="export", help="Export a YOLO model to a target format.")(export.main)
app.command(name="postprocess", help="Post-process YOLO temporal detections.")(postprocess.main)
app.command(name="compare", help="Compare trained models and rank by metrics.")(compare_models.main)
app.command(name="batch", help="Run a batch of modeling operations from YAML.")(batch.main)
app.command(name="prepare-dataset", help="Auto-split train/val for a YOLO dataset.")(prepare_dataset.main)

if __name__ == "__main__":
    app()
