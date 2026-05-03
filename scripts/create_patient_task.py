"""Extract ~30 sparse frames per case and upload to CVAT for Patient labeling.

Uses a non-overlapping sampling offset (1500) to avoid collisions with
existing frames extracted at interval=3000 (i.e., frames 0, 3000, 6000, …).

One-off script for creating the patient annotation campaign.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import DATA_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

load_dotenv(find_dotenv())

SHARE_ROOT = DATA_DIR
FRAMES_PER_CASE = 30
OFFSET = 1500  # Avoid frames at multiples of 3000

app = typer.Typer(add_completion=False)


def _get_part_frame_counts(case: str) -> list[tuple[str, int]]:
    """Return [(filename, frame_count), ...] sorted by filename."""
    meta_path = PREDICTIONS_DIR / case / "metadata.json"
    raw_dir = RAW_DATA_DIR / case

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        part_frames = meta.get("part_frames", {})
        if part_frames:
            return [(fn, n) for fn, n in sorted(part_frames.items())]

    # Fallback: count frames via OpenCV (fast — reads metadata only)
    videos = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".mp4")
    result = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        result.append((v.name, n))
    return result


def _extract_frame(video_path: Path, frame_num: int, output_path: Path) -> bool:
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    cv2.imwrite(str(output_path), frame)
    return True


@app.command()
def main(
    task_name: str = "patient_sparse",
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "extracted_frames" / "patient_sparse",
        help="Where to save extracted frames.",
    ),
    frames_per_case: int = typer.Option(FRAMES_PER_CASE, help="Frames to sample per case."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    project_id: int = typer.Option(default=..., envvar="CVAT_PROJECT_ID"),
    extract_only: bool = typer.Option(False, help="Extract frames without uploading to CVAT."),
):
    # Discover cases with raw videos
    cases = sorted(
        d.name
        for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and d.name != "test" and any(d.glob("*.mp4"))
    )
    logger.info(f"Found {len(cases)} cases with raw videos")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_image_paths: list[Path] = []

    for case in cases:
        raw_dir = RAW_DATA_DIR / case
        parts = _get_part_frame_counts(case)
        total_frames = sum(n for _, n in parts)

        if total_frames < frames_per_case:
            logger.warning(f"  {case}: only {total_frames} frames, sampling all")
            interval = 1
        else:
            interval = total_frames // frames_per_case

        # Build global sample points offset from existing extraction grid
        sample_globals = []
        g = OFFSET
        while g < total_frames and len(sample_globals) < frames_per_case:
            sample_globals.append(g)
            g += interval

        # Map global indices to (part_filename, local_frame)
        samples: list[tuple[str, int, int]] = []  # (filename, local_frame, global_frame)
        for g in sample_globals:
            cumulative = 0
            for filename, n_frames in parts:
                if cumulative + n_frames > g:
                    local = g - cumulative
                    samples.append((filename, local, g))
                    break
                cumulative += n_frames

        case_dir = output_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)

        extracted = 0
        for filename, local_frame, global_frame in samples:
            part_stem = Path(filename).stem
            out_name = f"{part_stem}_f{global_frame:06d}.png"
            out_path = case_dir / out_name

            if out_path.exists():
                all_image_paths.append(out_path)
                extracted += 1
                continue

            video_path = raw_dir / filename
            if _extract_frame(video_path, local_frame, out_path):
                all_image_paths.append(out_path)
                extracted += 1
            else:
                logger.warning(f"  Failed to extract frame {local_frame} from {video_path}")

        logger.info(
            f"  {case}: {extracted}/{len(samples)} frames "
            f"(interval={interval}, total={total_frames})"
        )

    logger.info(f"Total: {len(all_image_paths)} frames extracted to {output_dir}")

    if extract_only:
        logger.info("--extract-only set, skipping CVAT upload")
        return

    # Upload to CVAT
    share_paths = sorted(str(p.relative_to(SHARE_ROOT)) for p in all_image_paths)

    with make_client(host=host, port=port, credentials=(username, password)) as client:
        task = client.tasks.create(spec={
            "name": task_name,
            "project_id": project_id,
        })
        logger.info(f"Created task '{task_name}' (ID: {task.id})")

        logger.info(f"Uploading {len(share_paths)} images via SHARE...")
        task.upload_data(share_paths, resource_type=ResourceType.SHARE)
        task.fetch()
        logger.info(f"Upload complete — {task.size} frames")

    logger.success(
        f"Task '{task_name}' (ID: {task.id}) ready with {task.size} frames.\n"
        f"  {len(cases)} cases × ~{frames_per_case} frames/case"
    )


if __name__ == "__main__":
    app()
