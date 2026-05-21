"""Pull corrected Patient annotations from a CVAT task back to on-disk pseudo-labels.

Companion to ``upload_patient_fixes_to_cvat.py``. After fixing boxes in CVAT,
run this with the task ID to overwrite the corresponding ``.txt`` files under
``predictions/patient_v0_pseudo/labels/``. Only frames present in the CVAT task
are touched; everything else is left alone.

Behavior per frame:
  - CVAT has ≥1 Patient box   → overwrite the .txt with corrected boxes
  - CVAT has 0 boxes          → delete the stale .txt (if any), so the merge
                                step treats the frame as patient-free
  - CVAT has non-Patient boxes → fail loud — the project schema lets you draw
                                them but the bootstrap pipeline only knows
                                about Patient
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import zipfile

from cvat_sdk import make_client
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import DATA_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

PSEUDO_LABELS_DIR = DATA_DIR / "predictions" / "patient_v0_pseudo" / "labels"
SUMMARY_PATH = DATA_DIR / "predictions" / "patient_v0_pseudo" / "summary.txt"
PATIENT_CLASS = LABELS.index("Patient")


def _flatten_label_files(export_dir: Path) -> dict[str, Path]:
    """Walk the (possibly deeply nested) labels dir and return {stem: path}."""
    out: dict[str, Path] = {}
    for p in (export_dir / "labels").rglob("*.txt"):
        out[p.stem] = p
    return out


def _classify_lines(text: str) -> tuple[list[str], Counter[int]]:
    """Filter to Patient lines (class == PATIENT_CLASS) and count by class."""
    kept: list[str] = []
    counts: Counter[int] = Counter()
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        cls = int(parts[0])
        counts[cls] += 1
        if cls == PATIENT_CLASS:
            kept.append(raw)
    return kept, counts


@app.command()
def main(
    task_id: int = typer.Argument(..., help="CVAT task ID with the corrected boxes."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    dry_run: bool = typer.Option(False, help="Print plan without modifying disk."),
):
    """Pull a fixup task back to predictions/patient_v0_pseudo/labels/."""
    PSEUDO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"cvat_apply_{task_id}_") as tmpdir:
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
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        labels = _flatten_label_files(tmp)
        logger.info(f"Found {len(labels)} label files in export")

        # Collect plan first so dry-run can print without mutating.
        to_write: dict[str, list[str]] = {}
        to_delete: list[str] = []
        non_patient_seen: Counter[str] = Counter()
        for stem, path in labels.items():
            kept, counts = _classify_lines(path.read_text())
            for cls, n in counts.items():
                if cls != PATIENT_CLASS:
                    non_patient_seen[LABELS[cls]] += n
            if kept:
                to_write[stem] = kept
            else:
                # CVAT exported a .txt but no Patient lines remain. If a stale
                # .txt sits on disk, delete it so the merge step is honest.
                if (PSEUDO_LABELS_DIR / f"{stem}.txt").exists():
                    to_delete.append(stem)

    if non_patient_seen:
        logger.error(
            f"Non-Patient boxes found in the CVAT task: {dict(non_patient_seen)}.\n"
            f"This pipeline only round-trips Patient. Either delete those boxes "
            f"in CVAT and re-run, or split the task before applying."
        )
        raise typer.Exit(1)

    n_new = sum(1 for s in to_write if not (PSEUDO_LABELS_DIR / f"{s}.txt").exists())
    n_overwrite = len(to_write) - n_new
    logger.info(
        f"Plan: write {len(to_write)} files ({n_new} new, {n_overwrite} overwrite), "
        f"delete {len(to_delete)} stale files."
    )

    if dry_run:
        for stem in list(to_write)[:5]:
            logger.info(f"  WRITE  {stem}.txt ({len(to_write[stem])} boxes)")
        for stem in to_delete[:5]:
            logger.info(f"  DELETE {stem}.txt")
        logger.info("--dry-run set — no changes made.")
        return

    for stem, lines in to_write.items():
        (PSEUDO_LABELS_DIR / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    for stem in to_delete:
        (PSEUDO_LABELS_DIR / f"{stem}.txt").unlink()

    SUMMARY_PATH.write_text(
        f"Last fix-up applied {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"  CVAT task: {task_id}\n"
        f"  Files written: {len(to_write)} ({n_new} new, {n_overwrite} overwritten)\n"
        f"  Files deleted: {len(to_delete)}\n"
    )
    logger.success(f"Applied fixes from CVAT task {task_id}.")


if __name__ == "__main__":
    app()
