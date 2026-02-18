import cv2
import numpy as np
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from ent_cv.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# --- Helper Functions ---

def get_brightness(frame: np.ndarray) -> float:
    """Calculates average brightness of the frame using the V channel of HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))

def get_sharpness(frame: np.ndarray) -> float:
    """Calculates sharpness using Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# --- CLI Command ---

@app.command()
def extract_frames(
    input_video: Path = RAW_DATA_DIR / "20251113_02/20251113_02_Part1.mp4",
    output_dir: Path = PROCESSED_DATA_DIR / "extracted_frames",
    frame_interval: int = typer.Option(3000, help="Check every Nth frame"),
    blur_threshold: float = typer.Option(10.0, help="Minimum sharpness score"),
    min_brightness: float = typer.Option(40.0, help="Darkness cutoff"),
    max_brightness: float = typer.Option(210.0, help="Brightness cutoff"),
):
    if not input_video.exists():
        logger.error(f"File not found: {input_video}")
        raise typer.Exit(code=1)

    # Create a subfolder based on the video name to avoid filename collisions
    video_stem = input_video.stem
    save_path = output_dir / video_stem
    save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing {input_video.name} ({total_frames} total frames)")

    current_frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="Extracting") as pbar:
        while current_frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break

            # 1. Quality Filters
            sharpness = get_sharpness(frame)
            if sharpness < blur_threshold:
                logger.debug(f"Frame {current_frame_idx} rejected: Sharpness {sharpness:.2f}")
            else:
                brightness = get_brightness(frame)
                if not (min_brightness <= brightness <= max_brightness):
                    logger.debug(f"Frame {current_frame_idx} rejected: Brightness {brightness:.2f}")
                else:
                    # 2. SAVE LOGIC (Similarity check removed)
                    file_name = f"{video_stem}_f{current_frame_idx:06d}.jpg"
                    cv2.imwrite(str(save_path / file_name), frame)
                    saved_count += 1
            
            # Move index forward and update progress bar
            current_frame_idx += frame_interval
            pbar.update(frame_interval)

    cap.release()
    logger.success(f"Extraction complete! Saved {saved_count} frames to {save_path}")

if __name__ == "__main__":
    app()