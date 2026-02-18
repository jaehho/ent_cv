import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
import typer

from ent_cv.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# --- DATA STRUCTURES ---

@dataclass
class FrameMetrics:
    """Stores evaluation results for a single frame."""
    brightness: float
    sharpness: float
    is_bright: bool
    is_sharp: bool

    @property
    def passed(self) -> bool:
        return self.is_bright and self.is_sharp


# --- CORE LOGIC ---

def get_brightness(frame: np.ndarray) -> float:
    """Calculates brightness using the maximum of BGR channels."""
    return float(np.max(frame, axis=2).mean())

def get_sharpness(frame: np.ndarray) -> float:
    """Calculates sharpness using the variance of the Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def evaluate_frame(
    frame: np.ndarray,
    min_bright: float,
    max_bright: float,
    blur_thresh: float,
    force_full_eval: bool = False
) -> FrameMetrics:
    """Evaluates a frame, short-circuiting expensive checks if possible."""
    brightness = get_brightness(frame)
    is_bright = min_bright <= brightness <= max_bright
    
    sharpness = 0.0
    is_sharp = False

    # Only calculate expensive Laplacian if brightness passed, 
    # OR if we explicitly need the data for the UI/Logs
    if is_bright or force_full_eval:
        sharpness = get_sharpness(frame)
        is_sharp = sharpness >= blur_thresh

    return FrameMetrics(brightness, sharpness, is_bright, is_sharp)


# --- HELPERS ---

def advance_video(cap: cv2.VideoCapture, current_idx: int, interval: int) -> int:
    """Advances the video capture to the next desired frame efficiently."""
    next_idx = current_idx + interval
    
    if interval == 1:
        pass  # Next cap.read() fetches the natural next frame
    elif interval < 30:
        # cap.grab() skips decoding the image payload (fast for short jumps)
        for _ in range(interval - 1):
            cap.grab()
    else:
        # cap.set() forces keyframe decoding (required for long jumps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
        
    return next_idx

def update_preview(frame: np.ndarray, metrics: FrameMetrics, window_name: str) -> bool:
    """Renders the preview window. Returns False if user presses 'q' to quit."""
    display_frame = frame.copy()
    color = (0, 255, 0) if metrics.passed else (0, 0, 255) 
    status = "[SAVED]" if metrics.passed else "[REJECTED]"
    
    label = f"S:{metrics.sharpness:.1f} B:{metrics.brightness:.1f} {status}"
    cv2.putText(
        display_frame, label, (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )
    
    cv2.imshow(window_name, display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.warning("Preview closed by user.")
        cv2.destroyAllWindows()
        return False
    return True


# --- MAIN COMMAND ---

@app.command()
def extract_frames(
    input_video: Path = RAW_DATA_DIR / "20251113_02/20251113_02_Part1.mp4",
    output_dir: Path = PROCESSED_DATA_DIR / "extracted_frames",
    frame_interval: int = typer.Option(3000, help="Check every Nth frame"),
    blur_threshold: float = typer.Option(10.0, help="Minimum sharpness score"),
    min_brightness: float = typer.Option(40.0, help="Darkness cutoff"),
    max_brightness: float = typer.Option(210.0, help="Brightness cutoff"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Show real-time preview window"),
):
    if not input_video.exists():
        logger.error(f"File not found: {input_video}")
        raise typer.Exit(code=1)

    save_path = output_dir / input_video.stem
    save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    current_idx = 0
    saved_count = 0
    
    preview_window = "Frame Preview (Press 'q' to stop)"
    if preview:
        # Allows resizing while locking the aspect ratio
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(preview_window, 1280, 720) # Set to a comfortable viewing size

    with tqdm(total=total_frames, desc="Extracting") as pbar:
        while current_idx < total_frames:
            ret, frame = cap.read()
            if not ret: 
                break

            # 1. Evaluate
            force_eval = preview or verbose
            metrics = evaluate_frame(
                frame, min_brightness, max_brightness, blur_threshold, force_eval
            )

            # 2. Log & Save
            if verbose:
                logger.info(f"F:{current_idx} | S:{metrics.sharpness:.1f} | B:{metrics.brightness:.1f} | PASS:{metrics.passed}")

            if metrics.passed:
                file_name = f"{input_video.stem}_f{current_idx:06d}.jpg"
                cv2.imwrite(str(save_path / file_name), frame)
                saved_count += 1

            # 3. Handle UI
            if preview:
                preview = update_preview(frame, metrics, preview_window)

            # 4. Advance
            current_idx = advance_video(cap, current_idx, frame_interval)
            pbar.update(frame_interval)

    cap.release()
    cv2.destroyAllWindows()
    logger.success(f"Done! Saved {saved_count} frames.")

if __name__ == "__main__":
    app()