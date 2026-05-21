"""Pull verified Patient annotations from a CVAT task back to the pseudo dir.

Companion to ``create_patient_verify_task.py``. After verifying the suspect
(==0 or ≥2 box) frames in CVAT, run this with the task ID to overwrite the
corresponding .txt files in the pseudo directory and refresh ``summary.txt``.

Domain rule enforced here: every frame in the CVAT task must have 0 or 1
Patient box. Multi-box frames in the export are a verification bug — we'd
rather fail loud than silently mis-train.

Per-frame behavior:
  - CVAT has 1 Patient box  → overwrite the .txt with the corrected box
                              (class index remapped from the project's Patient
                              index to canonical 12, matching pseudo format)
  - CVAT has 0 boxes        → delete any existing .txt so the frame is
                              treated as a confirmed true negative downstream
  - CVAT has ≥2 boxes       → fail loud — the verifier missed deduplicating
  - CVAT has non-Patient    → fail loud — only Patient should be in this task
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import zipfile

from cvat_sdk import make_client
from cvat_sdk.api_client.api.tasks_api import TasksApi
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

DEFAULT_PSEUDO_DIR = DATA_DIR / "predictions" / "patient_v2_pseudo" / "labels"
CANONICAL_PATIENT_CLASS = LABELS.index("Patient")  # 12 — the pseudo dir contract


def _flatten_label_files(export_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in (export_dir / "labels").rglob("*.txt"):
        out[p.stem] = p
    return out


def _gt_patient_class(export_dir: Path) -> int:
    """Find Patient's class index in CVAT's exported data.yaml."""
    data_yaml = export_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"data.yaml missing in CVAT export: {data_yaml}")
    spec = yaml.safe_load(data_yaml.read_text())
    names = spec.get("names")
    if isinstance(names, dict):
        for idx, name in names.items():
            if name == "Patient":
                return int(idx)
    elif isinstance(names, list):
        for idx, name in enumerate(names):
            if name == "Patient":
                return idx
    raise RuntimeError(f"'Patient' not in data.yaml names: {names}")


def _classify(text: str, patient_class_in_export: int) -> tuple[list[str], Counter[int]]:
    """Filter to Patient lines, remap class index to canonical 12, return (lines, counts).

    ``counts`` includes ALL classes encountered so the caller can flag non-Patient.
    """
    kept: list[str] = []
    counts: Counter[int] = Counter()
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        counts[cls] += 1
        if cls == patient_class_in_export:
            kept.append(
                f"{CANONICAL_PATIENT_CLASS} " + " ".join(parts[1:])
            )
    return kept, counts


@app.command()
def main(
    task_id: int = typer.Argument(..., help="CVAT verify-task ID."),
    pseudo_dir: Path = typer.Option(
        DEFAULT_PSEUDO_DIR,
        help="Pseudo-label directory to update.",
    ),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    dry_run: bool = typer.Option(False, help="Print plan without modifying disk."),
):
    """Apply verified corrections from a CVAT task back to the pseudo directory."""
    if not pseudo_dir.is_dir():
        raise typer.BadParameter(f"--pseudo-dir not found: {pseudo_dir}")

    with tempfile.TemporaryDirectory(prefix=f"cvat_verify_{task_id}_") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "export.zip"
        with make_client(
            host=host, port=port, credentials=(username, password)
        ) as client:
            task = client.tasks.retrieve(task_id)
            logger.info(f"Task {task_id}: '{task.name}' (project {task.project_id})")
            logger.info(f"Exporting to {zip_path}...")
            task.export_dataset(
                format_name="Ultralytics YOLO Detection 1.0",
                filename=str(zip_path),
                include_images=False,
            )
            # Fetch the full frame manifest. CVAT's YOLO export omits frames
            # with zero annotations, so we can't tell "user explicitly cleared"
            # from "frame doesn't exist" by looking at the export alone.
            tasks_api = TasksApi(client.api_client)
            (meta, _) = tasks_api.retrieve_data_meta(id=task_id)
            all_task_stems = {Path(f.name).stem for f in meta.frames}
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        patient_in_export = _gt_patient_class(tmp)
        logger.info(f"  CVAT exports Patient as class {patient_in_export}")

        # Collect plan before mutating anything so dry-run is meaningful and so a
        # cardinality violation aborts before any file change.
        to_write: dict[str, list[str]] = {}
        to_delete: list[str] = []
        non_patient: Counter[str] = Counter()
        multi_box: list[str] = []

        labels = _flatten_label_files(tmp)
        logger.info(
            f"Task has {len(all_task_stems)} frames; "
            f"{len(labels)} with annotations in export "
            f"(other {len(all_task_stems) - len(labels)} explicitly cleared)"
        )
        for stem, path in labels.items():
            kept, counts = _classify(path.read_text(), patient_in_export)
            for cls, n in counts.items():
                if cls != patient_in_export:
                    non_patient[LABELS[cls] if cls < len(LABELS) else f"cls_{cls}"] += n
            if len(kept) > 1:
                multi_box.append(stem)
                continue
            if kept:
                to_write[stem] = kept
            else:
                # No Patient boxes in the export entry → all boxes were
                # non-Patient (caught by non_patient counter above). The
                # explicit-clear path is handled below.
                if (pseudo_dir / f"{stem}.txt").exists():
                    to_delete.append(stem)

        # Explicit-clear: frames in the task with no annotations in the export.
        # CVAT only includes annotated frames in YOLO export, so these are the
        # frames the user opened, deleted all annotations on, and saved.
        explicitly_cleared = all_task_stems - set(labels.keys())
        for stem in explicitly_cleared:
            if (pseudo_dir / f"{stem}.txt").exists():
                to_delete.append(stem)

    failures: list[str] = []
    if multi_box:
        failures.append(
            f"  {len(multi_box)} frames still have ≥2 Patient boxes "
            f"(examples: {multi_box[:5]})"
        )
    if non_patient:
        failures.append(f"  non-Patient classes in task: {dict(non_patient)}")
    if failures:
        for f in failures:
            logger.error(f)
        raise typer.Exit(
            "Verification incomplete — fix the above in CVAT and re-run."
        )

    n_existed = sum(1 for s in to_write if (pseudo_dir / f"{s}.txt").exists())
    n_new = len(to_write) - n_existed
    logger.info(
        f"Plan: write {len(to_write)} files ({n_new} new, {n_existed} overwrite), "
        f"delete {len(to_delete)} stale files."
    )

    if dry_run:
        for stem in list(to_write)[:5]:
            logger.info(f"  WRITE  {stem}.txt")
        for stem in to_delete[:5]:
            logger.info(f"  DELETE {stem}.txt")
        logger.info("--dry-run set — no changes made.")
        return

    for stem, lines in to_write.items():
        (pseudo_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    for stem in to_delete:
        (pseudo_dir / f"{stem}.txt").unlink()

    # Post-apply sanity: scan for residual ≥2-box files.
    # CVAT only exports frames with annotations, so a verify-task frame that
    # was opened but never explicitly edited slips through — its stale ≥2-box
    # pseudo .txt isn't in the export and never gets overwritten.
    residual_multi: list[str] = []
    for f in pseudo_dir.glob("*.txt"):
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        if len(lines) >= 2:
            residual_multi.append(f.stem)

    # Refresh summary.txt — the original was emitted by pseudo_label_patient.py
    # before verification; downstream tools will read the new state.
    _refresh_summary(pseudo_dir)

    logger.success(
        f"Applied verifications from CVAT task {task_id}.\n"
        f"  Wrote:    {len(to_write)} files\n"
        f"  Deleted:  {len(to_delete)} files\n"
        f"Summary refreshed at {pseudo_dir.parent / 'summary.txt'}."
    )

    if residual_multi:
        logger.warning(
            f"{len(residual_multi)} pseudo file(s) still have ≥2 boxes — "
            f"likely verify-task frames you opened but didn't explicitly edit. "
            f"Re-open them in the verify task, edit (even if just clicking a "
            f"box to confirm), save, and re-run this script.\n  " +
            "\n  ".join(residual_multi[:10])
        )
        raise typer.Exit(1)


def _refresh_summary(pseudo_dir: Path) -> None:
    """Recompute and write summary.txt to reflect post-verification state."""
    files = list(pseudo_dir.glob("*.txt"))
    box_count_hist: Counter[int] = Counter()
    boxes_total = 0
    frames_with_boxes = 0
    for f in files:
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        n = len(lines)
        box_count_hist[n] += 1
        boxes_total += n
        if n > 0:
            frames_with_boxes += 1
    summary_lines = [
        f"# Post-verification refresh at {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"Files in pseudo dir: {len(files)} (frames with no .txt = confirmed true negatives)",
        f"  with ≥1 Patient box: {frames_with_boxes}",
        f"Total Patient boxes: {boxes_total}",
        "Boxes-per-frame histogram (over files only):",
    ]
    for k in sorted(box_count_hist):
        summary_lines.append(f"  {k}: {box_count_hist[k]}")
    (pseudo_dir.parent / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    app()
