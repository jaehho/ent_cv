"""Remove prediction artifacts (frames, overlay videos, YOLO labels)."""

from pathlib import Path
import shutil
from typing import Optional

from loguru import logger
import typer

from ent_cv.config import PREDICTIONS_DIR

app = typer.Typer(add_completion=False)


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _collect_targets(
    case: Optional[str],
    frames: bool,
    videos: bool,
    labels: bool,
) -> list[tuple[Path, str]]:
    """Return list of (path, kind) tuples to delete."""
    targets: list[tuple[Path, str]] = []

    case_dirs = (
        [PREDICTIONS_DIR / case] if case else sorted(PREDICTIONS_DIR.iterdir())
    )

    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        for item in sorted(case_dir.iterdir()):
            if frames and item.is_dir() and item.name.endswith("_frames"):
                targets.append((item, "frames"))
            elif videos and item.is_file() and item.suffix.lower() == ".avi":
                targets.append((item, "video"))
            elif labels and item.is_dir() and item.name == "labels":
                targets.append((item, "labels"))

    return targets


@app.command()
def main(
    frames: bool = typer.Option(False, help="Delete extracted frame directories"),
    videos: bool = typer.Option(False, help="Delete .avi overlay videos"),
    labels: bool = typer.Option(False, help="Delete YOLO label directories"),
    all_: bool = typer.Option(False, "--all", help="Delete frames, videos, and labels"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted"),
    case: Optional[str] = typer.Option(None, help="Target a specific case"),
):
    """Remove prediction artifacts to reclaim disk space."""
    if all_:
        frames = videos = labels = True

    if not (frames or videos or labels):
        logger.error("Specify --frames, --videos, --labels, or --all")
        raise typer.Exit(1)

    targets = _collect_targets(case, frames, videos, labels)

    if not targets:
        logger.info("Nothing to clean.")
        raise typer.Exit(0)

    total_size = 0
    for path, kind in targets:
        size = _dir_size(path) if path.is_dir() else path.stat().st_size
        total_size += size
        prefix = "[dry-run] " if dry_run else ""
        logger.info(f"{prefix}{'rm' if not dry_run else 'would rm'} {path}  ({_human_size(size)}, {kind})")

    logger.info(f"{'Would free' if dry_run else 'Freeing'}: {_human_size(total_size)}")

    if dry_run:
        return

    for path, _ in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    logger.success(f"Freed {_human_size(total_size)}")
