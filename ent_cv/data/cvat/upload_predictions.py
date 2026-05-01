"""Register raw videos in CVAT with prediction annotations.

CVAT's share volume already mounts /mnt/data/ent_cv, so no data copying is
needed.  For each video part this script creates a CVAT task, points it at
the video on disk via SHARE, and imports all YOLO prediction annotations.

The ``use_cache=True`` flag defers chunk encoding to annotation time so that
task creation only needs to scan keyframes (not re-encode every frame).

Usage:
    uv run ent_cv/data/cvat/upload_predictions.py 20251113_02
    uv run ent_cv/data/cvat/upload_predictions.py --all
    uv run ent_cv/data/cvat/upload_predictions.py 20251113_02 --raw --min-conf 0.8
"""

import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from cvat_sdk import make_client
from cvat_sdk.core.client import Client
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer
import yaml

from ent_cv.config import DATA_DIR, LABELS, PREDICTIONS_DIR, RAW_DATA_DIR

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

# CVAT share volume maps /mnt/data/ent_cv -> /home/django/share
SHARE_ROOT = DATA_DIR


def _get_video_resolution(video_path: Path) -> tuple[int, int]:
    """Return (width, height) of a video file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = result.stdout.strip().split("x")
    w, h = parts[0], parts[1]
    return int(w), int(h)


def _load_detections(predictions_dir: Path, *, use_filtered: bool = True) -> list[dict]:
    """Load detections from a case predictions directory."""
    if use_filtered:
        path = predictions_dir / "filtered_detections.json"
        if not path.exists():
            logger.warning("No filtered_detections.json, falling back to detections.json")
            path = predictions_dir / "detections.json"
    else:
        path = predictions_dir / "detections.json"

    if not path.exists():
        raise FileNotFoundError(f"No detections found in {predictions_dir}")

    with open(path) as f:
        data = json.load(f)

    # filtered_detections.json wraps the list under {"_filter": ..., "detections": [...]}
    if isinstance(data, dict):
        return data["detections"]
    return data


def _count_video_frames(video_path: Path) -> int:
    """Return frame count of a video file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _build_part_frame_ranges(metadata: dict, raw_dir: Path) -> list[tuple[str, int, int]]:
    """Build (filename, global_start, global_end) for each video part.

    Returns parts in sorted filename order (matching predict.py's processing order).
    If metadata lacks ``part_frames``, infers frame counts from the video files.
    """
    part_frames = metadata.get("part_frames")

    if not part_frames:
        # Infer from video files
        videos = sorted(p.name for p in raw_dir.iterdir() if p.suffix.lower() == ".mp4")
        if not videos:
            raise FileNotFoundError(f"No .mp4 files in {raw_dir}")
        logger.info(
            f"metadata.json missing 'part_frames', counting frames from {len(videos)} videos..."
        )
        part_frames = {}
        for v in videos:
            part_frames[v] = _count_video_frames(raw_dir / v)
            logger.info(f"  {v}: {part_frames[v]} frames")

    ranges: list[tuple[str, int, int]] = []
    offset = 0
    for filename in sorted(part_frames):
        n_frames = part_frames[filename]
        ranges.append((filename, offset, offset + n_frames))
        offset += n_frames
    return ranges


def _build_yolo_annotations_zip(
    detections: list[dict],
    width: int,
    height: int,
    zip_path: Path,
) -> int:
    """Build YOLO annotation zip. Detections must have part-local frame numbers.

    Returns the number of annotated frames.
    """
    frames: dict[int, list[dict]] = {}
    for det in detections:
        frames.setdefault(det["frame"], []).append(det)

    data_yaml_content = yaml.dump(
        {
            "path": ".",
            "train": "train.txt",
            "names": {i: name for i, name in enumerate(LABELS)},
        },
        default_flow_style=False,
    )

    train_lines: list[str] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml_content)
        zf.writestr("images/train/.keep", "")

        for frame_num in sorted(frames):
            stem = f"frame_{frame_num:06d}"
            train_lines.append(f"images/train/{stem}.jpg")

            lines: list[str] = []
            for det in frames[frame_num]:
                x1, y1 = det["box"]["x1"], det["box"]["y1"]
                x2, y2 = det["box"]["x2"], det["box"]["y2"]

                x_center = max(0.0, min(1.0, (x1 + x2) / 2.0 / width))
                y_center = max(0.0, min(1.0, (y1 + y2) / 2.0 / height))
                bw = max(0.0, min(1.0, (x2 - x1) / width))
                bh = max(0.0, min(1.0, (y2 - y1) / height))

                lines.append(f"{det['class']} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            zf.writestr(f"labels/train/{stem}.txt", "\n".join(lines) + "\n")

        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    return len(frames)


def upload_case(
    case: str,
    *,
    client: Client,
    existing_tasks: dict[str, object],
    use_filtered: bool = True,
    min_conf: float = 0.0,
    project_id: int | None = None,
) -> tuple[list[int], list[int]]:
    """Create CVAT tasks for a case (one per video part).

    Returns (task_ids, failed_task_ids).
    """
    predictions_dir = PREDICTIONS_DIR / case
    raw_dir = RAW_DATA_DIR / case

    if not predictions_dir.exists():
        raise FileNotFoundError(f"No predictions for case: {case}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw video for case: {case}")

    with open(predictions_dir / "metadata.json") as f:
        metadata = json.load(f)
    part_ranges = _build_part_frame_ranges(metadata, raw_dir)

    all_detections = _load_detections(predictions_dir, use_filtered=use_filtered)
    logger.info(f"Loaded {len(all_detections)} detections across {len(part_ranges)} parts")

    width, height = _get_video_resolution(raw_dir / part_ranges[0][0])
    logger.info(f"Video resolution: {width}x{height}")

    task_ids: list[int] = []
    failed_parts: list[int] = []

    for filename, global_start, global_end in part_ranges:
        part_name = Path(filename).stem
        video_path = raw_dir / filename
        expected_frames = global_end - global_start

        # Check existing task health
        if part_name in existing_tasks:
            existing = existing_tasks[part_name]
            existing.fetch()
            task_size = getattr(existing, "size", None)
            has_frames = task_size is not None and task_size > 0

            if has_frames:
                ann = existing.get_annotations()
                n_shapes = len(ann.shapes)
                if n_shapes > 0:
                    logger.info(
                        f"Task '{part_name}' OK (ID: {existing.id}, "
                        f"{task_size} frames, {n_shapes} shapes), skipping"
                    )
                    task_ids.append(existing.id)
                    continue

            # Incomplete task — delete and recreate
            logger.warning(
                f"Task '{part_name}' (ID: {existing.id}) is incomplete "
                f"(frames={task_size}), deleting"
            )
            existing.remove()
            del existing_tasks[part_name]

        # Filter detections to this part and convert to local frame numbers
        part_dets = [
            {**det, "frame": det["frame"] - global_start}
            for det in all_detections
            if global_start <= det["frame"] < global_end and det["confidence"] >= min_conf
        ]

        logger.info(f"Part '{part_name}': {expected_frames} frames, {len(part_dets)} detections")

        # Register video via SHARE — no data copy
        share_path = str(video_path.relative_to(SHARE_ROOT))

        spec: dict = {"name": part_name}
        if project_id is not None:
            spec["project_id"] = project_id
        else:
            spec["labels"] = [{"name": name} for name in LABELS]
        task = client.tasks.create(spec=spec)
        logger.info(f"Task created: '{part_name}' (ID: {task.id})")

        task.upload_data(
            [share_path],
            resource_type=ResourceType.SHARE,
            params={"use_cache": True},
        )
        task.fetch()
        logger.info(f"Video registered — {task.size} frames")

        # Import annotations
        if part_dets:
            with tempfile.NamedTemporaryFile(
                suffix=".zip", delete=False, prefix=f"cvat_{part_name}_"
            ) as tmp:
                zip_path = Path(tmp.name)
            try:
                n_annotated = _build_yolo_annotations_zip(part_dets, width, height, zip_path)
                task.import_annotations(
                    format_name="Ultralytics YOLO Detection 1.0",
                    filename=str(zip_path),
                )

                # Verify import succeeded fully
                ann = task.get_annotations()
                actual = len(ann.shapes)
                if actual == len(part_dets):
                    logger.info(f"Imported {actual} annotations across {n_annotated} frames")
                else:
                    logger.error(
                        f"Annotation mismatch for '{part_name}' (task {task.id})! "
                        f"Expected {len(part_dets)} shapes, got {actual}. Will retry."
                    )
                    failed_parts.append(task.id)
            finally:
                zip_path.unlink(missing_ok=True)

        # Track new task for idempotency within the same run
        existing_tasks[part_name] = task
        task_ids.append(task.id)

    logger.success(f"Case '{case}' — {len(task_ids)} tasks: {task_ids}")
    return task_ids, failed_parts


@app.command()
def main(
    case: str = typer.Argument(
        None,
        help="Case name (e.g. '20251113_02'). Omit with --all for all cases.",
    ),
    all_cases: bool = typer.Option(False, "--all", help="Upload all cases with predictions."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    raw: bool = typer.Option(False, "--raw", help="Use raw detections instead of filtered."),
    min_conf: float = typer.Option(0.0, "--min-conf", help="Minimum confidence threshold."),
    project_id: int | None = typer.Option(
        None, envvar="CVAT_PROJECT_ID",
        help="CVAT project ID. Labels are inherited from the project.",
    ),
):
    """Register videos in CVAT with prediction annotations."""
    if not case and not all_cases:
        logger.error("Provide a case name or use --all")
        raise typer.Exit(1)

    if all_cases:
        cases = sorted(
            d.name
            for d in PREDICTIONS_DIR.iterdir()
            if d.is_dir() and (d / "detections.json").exists() and (RAW_DATA_DIR / d.name).exists()
        )
    else:
        cases = [case]

    logger.info(f"Processing {len(cases)} case(s): {', '.join(cases)}")

    with make_client(host=host, port=port, credentials=(username, password)) as client:
        # Single task list lookup for the entire run
        existing_tasks: dict[str, object] = {t.name: t for t in client.tasks.list()}

        all_failed: list[int] = []

        for c in cases:
            try:
                _, failed = upload_case(
                    c,
                    client=client,
                    existing_tasks=existing_tasks,
                    use_filtered=not raw,
                    min_conf=min_conf,
                    project_id=project_id,
                )
                all_failed.extend(failed)
            except Exception:
                logger.exception(f"Failed case '{c}' — check CVAT for partial tasks")
                if not all_cases:
                    raise

        # Retry tasks with annotation mismatches
        if all_failed:
            logger.warning(f"Retrying {len(all_failed)} tasks with annotation mismatches...")
            for tid in all_failed:
                try:
                    task = client.tasks.retrieve(tid)
                    name = task.name
                    logger.info(f"Deleting task {tid} ({name}) for retry")
                    task.remove()
                    existing_tasks.pop(name, None)
                except Exception:
                    logger.exception(f"Failed to delete task {tid}")

            # Second pass — recreate deleted tasks
            for c in cases:
                try:
                    upload_case(
                        c,
                        client=client,
                        existing_tasks=existing_tasks,
                        use_filtered=not raw,
                        min_conf=min_conf,
                    )
                except Exception:
                    logger.exception(f"Retry failed for case '{c}'")


if __name__ == "__main__":
    app()
