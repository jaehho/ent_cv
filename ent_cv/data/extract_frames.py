from typing import Literal, Optional, overload
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from collections import deque

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from ent_cv.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from ent_cv.utils import notify

app = typer.Typer()

_SEEK_THRESHOLD = 10
_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}
)

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

_PREVIEW_LABEL_HEIGHT = 56
_PREVIEW_PANEL_MAX = 1920


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    """Attach a dark banner with white text beneath an image."""
    h, w = img.shape[:2]
    banner = np.zeros((_PREVIEW_LABEL_HEIGHT, w, 3), dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (12, _PREVIEW_LABEL_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([img, banner])


_INSET_SRC_SIZE = 100   # pixels cropped from center of each source image
_INSET_ZOOM = 3         # magnification applied to the crop
_INSET_PAD = 8          # gap from panel edge (px)
_INSET_BORDER = 2       # border thickness (px)


def _add_inset(
    panel: np.ndarray,
    src_img: np.ndarray,
    corner: str,          # "br" | "bl" | "tr" | "tl"
    label_h: int,
) -> np.ndarray:
    """
    Overlay a magnified crop of the centre of *src_img* onto *panel*
    at the specified corner.  *label_h* is the height of the label
    banner already attached at the bottom of *panel* (so "br"/"bl"
    corners stay above the label).
    """
    half = _INSET_SRC_SIZE // 2
    sh, sw = src_img.shape[:2]
    cy, cx = sh // 2, sw // 2
    y1s = max(0, cy - half)
    y2s = min(sh, cy + half)
    x1s = max(0, cx - half)
    x2s = min(sw, cx + half)
    crop = src_img[y1s:y2s, x1s:x2s]

    inset_size = _INSET_SRC_SIZE * _INSET_ZOOM
    inset = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)

    # Add a coloured border
    b = _INSET_BORDER
    inset_bordered = cv2.copyMakeBorder(
        inset, b, b, b, b, cv2.BORDER_CONSTANT, value=(255, 220, 80)
    )
    ih, iw = inset_bordered.shape[:2]

    out = panel.copy()
    ph, pw = out.shape[:2]
    image_h = ph - label_h   # height of the image region (above label)

    pad = _INSET_PAD
    if corner == "br":
        oy = image_h - ih - pad
        ox = pw - iw - pad
    elif corner == "bl":
        oy = image_h - ih - pad
        ox = pad
    elif corner == "tr":
        oy = pad
        ox = pw - iw - pad
    else:  # "tl"
        oy = pad
        ox = pad

    # Clip to panel bounds just in case
    oy = max(0, oy)
    ox = max(0, ox)
    oy2 = min(ph, oy + ih)
    ox2 = min(pw, ox + iw)
    inset_bordered = inset_bordered[: oy2 - oy, : ox2 - ox]
    out[oy:oy2, ox:ox2] = inset_bordered
    return out


def _build_comparison(
    panels: list[tuple[np.ndarray, str]],
    frame_idx: int,
    video_name: str,
) -> np.ndarray:
    """Build a 2×2 grid from up to 4 (image, label) pairs."""
    assert 1 <= len(panels) <= 4, "need 1–4 panels"

    h, w = panels[0][0].shape[:2]
    scale = min(1.0, _PREVIEW_PANEL_MAX / max(h, w))
    panel_w = max(1, round(w * scale))
    panel_h = max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    def _make_panel(img: np.ndarray, label: str) -> np.ndarray:
        small = cv2.resize(img, (panel_w, panel_h), interpolation=interp)
        return _draw_label(small, label)

    labelled = [_make_panel(img, lbl) for img, lbl in panels]

    # Inner-corner positions for each panel slot (where panels meet the centre)
    _inset_corners = ["br", "bl", "tr", "tl"]
    for i, (img, _) in enumerate(panels):
        corner = _inset_corners[i]
        labelled[i] = _add_inset(labelled[i], img, corner, _PREVIEW_LABEL_HEIGHT)

    blank = np.zeros_like(labelled[0])
    while len(labelled) < 4:
        labelled.append(blank)

    top = np.hstack([labelled[0], labelled[1]])
    bot = np.hstack([labelled[2], labelled[3]])
    grid = np.vstack([top, bot])

    bar_h = 64
    bar = np.full((bar_h, grid.shape[1], 3), 30, dtype=np.uint8)
    status = (
        f"{video_name}  |  frame {frame_idx}  |  "
        f"[SPACE] next  [S] save  [Q] quit"
    )
    cv2.putText(
        bar, status, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 210, 255), 2, cv2.LINE_AA
    )

    return np.vstack([grid, bar])


@contextmanager
def _null_context():
    yield None


class PreviewWindow:
    """Persistent OpenCV window for before/after comparisons."""

    _NAME = "Frame Preview  —  SPACE: next  S: save  Q: quit"

    def __init__(self, save_dir: Path | None) -> None:
        self.save_dir = save_dir
        self._last_comparison: np.ndarray | None = None

    def __enter__(self) -> "PreviewWindow":
        cv2.namedWindow(self._NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self._NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._NAME)
        except Exception:
            pass

    def show(
        self,
        panels: list[tuple[np.ndarray, str]],
        frame_idx: int,
        video_name: str,
    ) -> str:
        self._last_comparison = _build_comparison(panels, frame_idx, video_name)
        cv2.imshow(self._NAME, self._last_comparison)
        cv2.setWindowProperty(self._NAME, cv2.WND_PROP_TOPMOST, 1)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                return "quit"
            if key in (ord("s"), ord("S")):
                self._save(video_name, frame_idx)
                return "next"
            if key == ord(" "):
                return "next"

    def _save(self, video_name: str, frame_idx: int) -> None:
        if self.save_dir is None:
            logger.warning("  --preview-save-dir not set; comparison not saved.")
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fname = self.save_dir / f"{Path(video_name).stem}_f{frame_idx:06d}_comparison.jpg"
        if self._last_comparison is None:
            return
        cv2.imwrite(str(fname), self._last_comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info(f"  Saved comparison → {fname}")


# ---------------------------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------------------------

# Pre-computed unsharp mask kernel (avoid recomputing per frame)
_SHARPEN_KERNEL = np.array(
    [[ 0, -0.3,  0],
     [-0.3,  2.2, -0.3],
     [ 0, -0.3,  0]],
    dtype=np.float32,
)


class FrameEnhancer:
    """
    Reusable frame enhancer that caches the CLAHE object across frames.

    Creating cv2.createCLAHE() is cheap but not free — caching it avoids
    redundant allocations when processing thousands of frames with the
    same parameters.
    """

    def __init__(
        self,
        clahe_clip_limit: float,
        clahe_tile_grid_size: int,
        denoise: bool,
        sharpen: bool,
    ) -> None:
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.denoise = denoise
        self.sharpen = sharpen

        self._clahe: cv2.CLAHE | None = None
        if clahe_clip_limit > 0:
            self._clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
            )

        # Determine if pipeline is a no-op
        self._is_noop = not denoise and self._clahe is None and not sharpen

    @overload
    def __call__(self, frame: np.ndarray, return_stages: Literal[False] = ...) -> np.ndarray: ...
    @overload
    def __call__(self, frame: np.ndarray, return_stages: Literal[True]) -> tuple[np.ndarray, ...]: ...
    def __call__(
        self,
        frame: np.ndarray,
        return_stages: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Apply quality enhancements to a BGR frame.

        Pipeline: 1. Denoise → 2. CLAHE → 3. Sharpen

        If return_stages=True, returns (original, after_denoise, after_clahe, final).
        """
        if self._is_noop and not return_stages:
            return frame

        stage0 = frame

        # Stage 1: denoise
        if self.denoise:
            frame = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                h=3,
                hColor=3,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        stage1 = frame

        # Stage 2: CLAHE on L channel only
        if self._clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # In-place apply on L channel avoids split/merge overhead
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        stage2 = frame

        # Stage 3: sharpen via single filter2D instead of blur + addWeighted
        if self.sharpen:
            frame = cv2.filter2D(frame, -1, _SHARPEN_KERNEL)
        stage3 = frame

        if return_stages:
            return stage0, stage1, stage2, stage3
        return stage3


# ---------------------------------------------------------------------------
# Video navigation
# ---------------------------------------------------------------------------

def advance_video(cap: cv2.VideoCapture, current_idx: int, interval: int) -> int:
    """Advance video capture by `interval` frames from `current_idx`."""
    next_idx = current_idx + interval
    if interval == 1:
        pass
    elif interval < _SEEK_THRESHOLD:
        for _ in range(interval - 1):
            cap.grab()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
    return next_idx


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_videos(input_path: Path) -> list[Path]:
    """Find all video files under a path. If given a file, return it directly."""
    if input_path.is_file():
        if input_path.suffix.lower() not in _VIDEO_EXTENSIONS:
            logger.warning(
                f"'{input_path}' does not have a recognised video extension — proceeding anyway."
            )
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
        )

    return []


# ---------------------------------------------------------------------------
# Async I/O writer
# ---------------------------------------------------------------------------

class AsyncFrameWriter:
    """
    Offload cv2.imwrite to a thread pool so the main loop can decode + enhance
    the next frame while the previous one is flushed to disk.

    Uses a bounded deque to avoid unbounded memory growth — if the writer
    falls behind, submit() blocks until a slot opens.
    """

    def __init__(self, max_workers: int = 2, max_pending: int = 32) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: deque[Future] = deque()
        self._max_pending = max_pending
        self._lock = Lock()
        self.errors: int = 0

    def submit(self, path: str, frame: np.ndarray, params: list[int]) -> None:
        # Back-pressure: if too many writes pending, wait for oldest to finish
        while len(self._pending) >= self._max_pending:
            oldest = self._pending.popleft()
            oldest.result()

        fut = self._pool.submit(self._write, path, frame, params)
        self._pending.append(fut)

    def _write(self, path: str, frame: np.ndarray, params: list[int]) -> None:
        ok = cv2.imwrite(path, frame, params)
        if not ok:
            with self._lock:
                self.errors += 1
            logger.warning(f"imwrite failed: {path}")

    def drain(self) -> None:
        """Wait for all pending writes to finish."""
        while self._pending:
            self._pending.popleft().result()

    def shutdown(self) -> None:
        self.drain()
        self._pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------

def _letterbox_frame(
    frame: np.ndarray,
    target: int,
    interp: int,
    pad_value: int = 114,
) -> np.ndarray:
    """Resize + pad to target×target, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left

    return cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_frames_from_video(
    input_video: Path,
    save_path: Path,
    frame_interval: int,
    output_width: int | None,
    fmt: str,
    jpeg_quality: int,
    letterbox: bool,
    letterbox_size: int,
    enhancer: FrameEnhancer,
    preview_window: PreviewWindow | None,
    preview_scale: float = 1.0,
    tqdm_position: int = 1,
) -> int:
    """Extract and preprocess frames from a single video. Returns frames saved."""
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        logger.error(f"OpenCV could not open video: {input_video}")
        return 0

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- Determine output dimensions ---
        if letterbox:
            out_w = out_h = letterbox_size
            resize_mode = f"letterbox {letterbox_size}x{letterbox_size}"
        elif output_width is not None:
            out_w = output_width
            out_h = round(output_width * source_height / source_width)
            resize_mode = f"scale {out_w}x{out_h}"
        else:
            out_w, out_h = source_width, source_height
            resize_mode = "native"

        needs_resize = letterbox or (out_w != source_width or out_h != source_height)

        logger.info(
            f"Video: {total_frames} frames @ {fps:.1f} fps | "
            f"Sampling every {frame_interval} frames ({frame_interval / fps:.1f}s)"
        )
        logger.info(
            f"Output: {save_path} ({fmt.upper()}) | Resize: {resize_mode} | "
            f"CLAHE clip={enhancer.clahe_clip_limit} | "
            f"Denoise={enhancer.denoise} | Sharpen={enhancer.sharpen}"
        )

        if preview_window is not None:
            logger.info("Preview mode active — no frames will be saved until you exit preview.")

        save_path.mkdir(parents=True, exist_ok=True)

        interp = (
            cv2.INTER_AREA
            if (out_w < source_width or out_h < source_height)
            else cv2.INTER_LINEAR
        )
        write_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if fmt == "jpg" else []
        expected_frames = max(1, total_frames // frame_interval)
        saved = 0
        current_idx = 0
        aborted = False

        # Pre-compute the output filename stem to avoid repeated .stem calls
        video_stem = input_video.stem

        writer = AsyncFrameWriter() if preview_window is None else None

        try:
            with tqdm(
                total=expected_frames,
                desc=f"  {input_video.name}",
                unit="img",
                disable=preview_window is not None,
                position=tqdm_position,
                leave=False,
            ) as pbar:
                while current_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"cap.read() failed at frame {current_idx}. Ending early.")
                        break

                    # --- Resize / letterbox ---
                    if letterbox:
                        resized = _letterbox_frame(frame, letterbox_size, interp)
                    elif needs_resize:
                        resized = cv2.resize(frame, (out_w, out_h), interpolation=interp)
                    else:
                        resized = frame

                    if preview_window is not None:
                        # --- Preview path ---
                        if preview_scale < 1.0:
                            ph = max(1, round(resized.shape[0] * preview_scale))
                            pw = max(1, round(resized.shape[1] * preview_scale))
                            preview_frame = cv2.resize(
                                resized, (pw, ph), interpolation=cv2.INTER_AREA
                            )
                        else:
                            preview_frame = resized

                        s0, s1, s2, s3 = enhancer(preview_frame, return_stages=True)
                        panels = [(s0, "Original")]
                        applied: list[str] = []
                        if enhancer.denoise:
                            label = ((" + ".join(applied) + " > ") if applied else "") + "+Denoise"
                            panels.append((s1, label))
                            applied.append("Denoise")
                        if enhancer.clahe_clip_limit > 0:
                            label = ((" + ".join(applied) + " > ") if applied else "") + "+CLAHE"
                            panels.append((s2, label))
                            applied.append("CLAHE")
                        if enhancer.sharpen:
                            label = ((" + ".join(applied) + " > ") if applied else "") + "+Sharpen"
                            panels.append((s3, label))
                        action = preview_window.show(
                            panels=panels,
                            frame_idx=current_idx,
                            video_name=input_video.name,
                        )
                        if action == "quit":
                            logger.info("Preview aborted by user.")
                            aborted = True
                            break
                        current_idx = advance_video(cap, current_idx, frame_interval)
                        continue

                    # --- Save path ---
                    enhanced = enhancer(resized)

                    file_name = f"{video_stem}_f{current_idx:06d}.{fmt}"
                    assert writer is not None
                    writer.submit(str(save_path / file_name), enhanced, write_params)
                    saved += 1
                    pbar.update(1)

                    current_idx = advance_video(cap, current_idx, frame_interval)
        finally:
            if writer is not None:
                writer.shutdown()

        if preview_window is not None and not aborted:
            logger.info("Preview complete. Re-run without --preview to save frames.")

        return -1 if aborted else saved

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _extract_results_fn(info: dict) -> tuple[str, list[Path]]:
    """Build a summary string from the frame extraction result dict."""
    lines: list[str] = []

    # Totals
    lines.append(f"  Frames saved : {info['total_saved']}")
    lines.append(f"  Videos       : {info['num_videos']}")

    # Per-video breakdown
    lines.append("")
    lines.append("── Videos ───────────────────────────────────")
    for name, count in info["video_counts"].items():
        lines.append(f"  {name}  →  {count} frames")

    # Settings
    lines.append("")
    lines.append("── Settings ─────────────────────────────────")
    opts = info["options"]
    lines.append(f"  Output dir      : {opts['output_dir']}")
    lines.append(f"  Frame interval  : {opts['frame_interval']}")
    lines.append(f"  Format          : {opts['fmt']}" + (
        f"  (quality {opts['jpeg_quality']})" if opts['fmt'] == 'jpg' else ""))
    if opts["letterbox"]:
        lines.append(f"  Letterbox       : {opts['letterbox_size']}×{opts['letterbox_size']}")
    elif opts["output_width"]:
        lines.append(f"  Output width    : {opts['output_width']}px")
    else:
        lines.append(  "  Resize          : none (source resolution)")
    lines.append(f"  CLAHE           : clip={opts['clahe_clip_limit']}  tile={opts['clahe_tile_grid_size']}×{opts['clahe_tile_grid_size']}")
    lines.append(f"  Denoise         : {'yes' if opts['denoise'] else 'no'}")
    lines.append(f"  Sharpen         : {'yes' if opts['sharpen'] else 'no'}")

    return "\n".join(lines), []


@app.command()
@notify("Frame Extraction", results_fn=_extract_results_fn)
def extract_frames(
    input_path: Path = typer.Argument(
        RAW_DATA_DIR / "",
        help="Path to a video file or directory of videos (searched recursively).",
    ),
    output_dir: Path = typer.Argument(
        PROCESSED_DATA_DIR / "extracted_frames" / "batch1",
        help="Root directory for saved frames.",
    ),
    frame_interval: int = typer.Option(
        3000,
        help=(
            "Sample every Nth frame."
        ),
    ),
    output_width: Optional[int] = typer.Option(
        None,
        show_default="source width",
        help="Resize width in pixels (height derived automatically). Ignored when --letterbox is set.",
    ),
    fmt: str = typer.Option(
        "png", "--format", "-f", help="Output format: 'png' (lossless) or 'jpg'."
    ),
    jpeg_quality: int = typer.Option(
        95, min=0, max=100, help="JPEG quality (0–100). Used only with --format=jpg."
    ),
    letterbox: bool = typer.Option(
        False,
        help="Resize + pad to a square matching --letterbox-size. Recommended for YOLO training.",
    ),
    letterbox_size: int = typer.Option(
        640,
        help="Square side length when --letterbox is enabled. Use 1280 for small objects.",
    ),
    clahe_clip_limit: float = typer.Option(
        2.0,
        help="CLAHE clip limit for adaptive contrast. 0 disables. 1–3 is mild; >4 looks artificial.",
    ),
    clahe_tile_grid_size: int = typer.Option(
        8,
        help="CLAHE tile grid size (N×N). Smaller = more localised correction.",
    ),
    denoise: bool = typer.Option(
        True,
        help="Apply Non-Local Means denoising. Slow but effective for noisy footage.",
    ),
    sharpen: bool = typer.Option(
        True,
        help="Apply a mild unsharp mask after CLAHE. Helps on soft footage.",
    ),
    preview: bool = typer.Option(
        False,
        help=(
            "Open a side-by-side before/after window for each sampled frame. "
            "No frames are saved in this mode — use it to tune enhancement settings "
        ),
    ),
    preview_save_dir: Optional[Path] = typer.Option(
        Path("/home/jaeho/ent_cv/data/interim"),
        help="Directory to save comparison images when pressing S in preview mode.",
    ),
    preview_scale: float = typer.Option(
        0.5,
        min=0.1,
        max=1.0,
        help=(
            "Scale factor applied to frames before enhancement in preview mode. "
            "0.5 = half resolution, making NLM denoising ~4× faster. "
            "Does not affect saved output quality. Use 1.0 for full 4K preview."
        ),
    ),
) -> dict:
    """Extract evenly-spaced frames from one or more videos for YOLO annotation."""
    if not input_path.exists():
        logger.error(f"Path not found: {input_path}")
        raise typer.Exit(code=1)

    if frame_interval < 1:
        logger.error("--frame-interval must be >= 1.")
        raise typer.Exit(code=1)

    if fmt not in ("png", "jpg"):
        logger.error(f"Unsupported format '{fmt}'. Use 'png' or 'jpg'.")
        raise typer.Exit(code=1)

    if letterbox and output_width is not None:
        logger.warning("Both --letterbox and --output-width set; --output-width will be ignored.")

    if preview:
        logger.info(
            "Running in PREVIEW mode — frames will NOT be saved.\n"
            "  SPACE  advance to next frame\n"
            "  S      save current comparison image (requires --preview-save-dir)\n"
            "  Q/ESC  quit preview\n"
        )

    videos = discover_videos(input_path)
    if not videos:
        logger.error(f"No video files found at: {input_path}")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(videos)} video(s):")
    for i, v in enumerate(videos, 1):
        try:
            rel = v.relative_to(input_path)
        except ValueError:
            rel = v
        logger.info(f"  [{i}] {rel}")

    enhancer = FrameEnhancer(
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
        denoise=denoise,
        sharpen=sharpen,
    )

    total_saved = 0
    video_counts: dict[str, int] = {}
    with (PreviewWindow(save_dir=preview_save_dir) if preview else _null_context()) as win:
        preview_window = win if preview else None
        with tqdm(
            videos,
            desc="Videos",
            unit="video",
            position=1,
            leave=True,
            disable=preview is not None and preview,
        ) as video_pbar:
            for idx, video in enumerate(videos, 1):
                video_pbar.set_description(f"Videos [{video.name}]")
                logger.info(f"\n[{idx}/{len(videos)}] Processing: {video.name}")
                saved = extract_frames_from_video(
                    input_video=video,
                    save_path=output_dir / video.stem,
                    frame_interval=frame_interval,
                    output_width=output_width,
                    fmt=fmt,
                    jpeg_quality=jpeg_quality,
                    letterbox=letterbox,
                    letterbox_size=letterbox_size,
                    enhancer=enhancer,
                    preview_window=preview_window,
                    preview_scale=preview_scale,
                    tqdm_position=0,
                )
                video_pbar.update(1)
                if saved == -1:
                    logger.info("Preview aborted — stopping all videos.")
                    break
                total_saved += saved
                video_counts[video.name] = saved
                if not preview:
                    logger.success(f"  Saved {saved} frames → {output_dir / video.stem}")

    if not preview:
        logger.success(f"\nDone. {total_saved} frames from {len(videos)} video(s).")

    return {
        "total_saved": total_saved,
        "num_videos": len(videos),
        "video_counts": video_counts,
        "options": {
            "output_dir": output_dir,
            "frame_interval": frame_interval,
            "fmt": fmt,
            "jpeg_quality": jpeg_quality,
            "letterbox": letterbox,
            "letterbox_size": letterbox_size,
            "output_width": output_width,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid_size": clahe_tile_grid_size,
            "denoise": denoise,
            "sharpen": sharpen,
        },
    }


if __name__ == "__main__":
    app( )