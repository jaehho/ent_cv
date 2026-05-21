"""Snapshot a patient pseudo-label directory for reproducibility.

Pseudo-label dirs are mutable on disk — the verification pipeline writes/deletes
.txt files in place across bootstrap rounds. To freeze the state that produced
a trained model, copy the pseudo dir into snapshots/ with a manifest recording
provenance (CVAT task IDs, trained weights path, git state) and the scripts
that produced it.

Output layout:
    predictions/snapshots/<version>_<date>/
    ├── SNAPSHOT.yaml      provenance + content stats
    ├── labels/            verbatim copy of the source labels/
    ├── summary.txt        verbatim copy of summary.txt (if present)
    └── code/              copy of patient-related scripts at write time

Snapshot files are chmod'd 0o444 to discourage accidental writes. To remove
or overwrite a snapshot, delete it explicitly (chmod -R u+w first).
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import shutil
import subprocess

from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR

app = typer.Typer(add_completion=False)

PREDICTIONS_DIR = DATA_DIR / "predictions"
SNAPSHOTS_DIR = PREDICTIONS_DIR / "snapshots"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATTERNS = ("*patient*.py", "train_full*.py")


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _count_pseudo_state(labels_dir: Path) -> dict:
    files = list(labels_dir.glob("*.txt"))
    boxes = 0
    frames_with_boxes = 0
    for f in files:
        n = sum(1 for ln in f.read_text().splitlines() if ln.strip())
        boxes += n
        if n > 0:
            frames_with_boxes += 1
    return {
        "label_files": len(files),
        "frames_with_boxes": frames_with_boxes,
        "boxes_total": boxes,
    }


def _copy_scripts(code_dst: Path) -> list[str]:
    seen: set[Path] = set()
    for pat in SCRIPT_PATTERNS:
        for src in (REPO_ROOT / "scripts").glob(pat):
            if src in seen:
                continue
            shutil.copy2(src, code_dst / src.name)
            seen.add(src)
    return sorted(p.name for p in seen)


def _make_readonly(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(0o444)


@app.command()
def main(
    version: str = typer.Argument(..., help="Snapshot version tag, e.g. 'patient_v2'."),
    source: Path = typer.Option(
        None, help="Source pseudo dir. Defaults to predictions/<version>_pseudo/."
    ),
    verify_task_id: int = typer.Option(
        ..., help="CVAT verify task ID — the final human-touch task producing this state."
    ),
    cvat_task_ids: str = typer.Option(
        "",
        help="Comma-separated CVAT task IDs in the chain that contributed labels "
        "(e.g. '105,106,107,108').",
    ),
    trained_model: Path = typer.Option(
        ..., help="Path to best.pt this snapshot produced (for traceability)."
    ),
    notes: str = typer.Option("", help="Free-form notes."),
):
    """Freeze a pseudo-label dir into a versioned read-only snapshot."""
    source = source or PREDICTIONS_DIR / f"{version}_pseudo"
    if not source.is_dir():
        raise typer.BadParameter(f"source not found: {source}")
    labels_dir = source / "labels"
    if not labels_dir.is_dir():
        raise typer.BadParameter(f"source has no labels/ dir: {source}")
    if not trained_model.exists():
        logger.warning(f"trained_model path doesn't exist: {trained_model}")

    today = date.today().isoformat()
    dst = SNAPSHOTS_DIR / f"{version}_{today}"
    if dst.exists():
        raise typer.BadParameter(
            f"snapshot already exists: {dst}\n"
            f"Delete it first if you really want to overwrite "
            f"(chmod -R u+w {dst} && rm -rf {dst})."
        )

    logger.info(f"Source:    {source}")
    logger.info(f"Dest:      {dst}")
    logger.info(f"Verify:    CVAT task {verify_task_id}")
    dst.mkdir(parents=True)

    shutil.copytree(labels_dir, dst / "labels")
    summary = source / "summary.txt"
    if summary.exists():
        shutil.copy2(summary, dst / "summary.txt")

    code_dst = dst / "code"
    code_dst.mkdir()
    copied_scripts = _copy_scripts(code_dst)

    git_sha = _git(["rev-parse", "HEAD"])
    git_dirty = bool(_git(["status", "--porcelain"]))
    untracked = _git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    relevant_untracked = sorted(
        p for p in untracked if "patient" in p.lower() or "merge" in p.lower()
    )

    task_ids = [int(x) for x in cvat_task_ids.split(",") if x.strip()]
    stats = _count_pseudo_state(labels_dir)

    manifest = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_pseudo_dir": str(source.relative_to(DATA_DIR)),
        "provenance": {
            "cvat_verify_task_id": verify_task_id,
            "cvat_label_task_ids": task_ids,
            "trained_model": str(trained_model),
            "git_head_sha": git_sha,
            "git_tree_dirty": git_dirty,
            "relevant_untracked_files": relevant_untracked,
        },
        "contents": stats,
        "scripts_snapshotted": copied_scripts,
        "notes": notes or None,
    }
    (dst / "SNAPSHOT.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    _make_readonly(dst)

    logger.success(
        f"Snapshot written: {dst}\n"
        f"  files:    {stats['label_files']}\n"
        f"  frames:   {stats['frames_with_boxes']} with boxes\n"
        f"  boxes:    {stats['boxes_total']}\n"
        f"  scripts:  {len(copied_scripts)} copied\n"
        f"  notes:    {notes or '(none)'}"
    )


if __name__ == "__main__":
    app()
