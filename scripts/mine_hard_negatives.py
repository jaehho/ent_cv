"""Mine candidate frames from full_v0 inference into reviewer-specific buckets.

Runs the 13-class full_v0 detector on unseen surgical cases (raw videos with
no train/val frames yet), then splits frames into four buckets — each fed to a
separate CVAT task with a focused review workflow.

  multi_patient (FPs):   >=2 Patient predictions (domain rule violation).
                         Preload all boxes; reviewer deletes extras until 1.
  uncertain:             at least one prediction with conf < 0.7 (and not
                         multi-Patient). The actual HN-mining bucket — model
                         is unsure, FPs are concentrated here.
  confident:             every prediction has conf >= 0.7 (and not multi-Patient).
                         Mostly TPs; reviewer confirms or fixes the few wrong
                         ones. Could be auto-accepted but verifying is cheap.
  silent:                zero predictions. Reviewer labels from scratch for
                         rare classes (Drill, Empty Hand, Not Sure) or missed
                         instruments.

Output structure:
  predictions/<run_name>/
    inference_summaries.json    per-frame results (lines + per-prediction confs)
    selected/<bucket>/
      images/<stem>.png         symlinks to source frames
      preload/<stem>.txt        YOLO preload (silent bucket has none)
    candidates.tsv              one row per selected frame, bucket-annotated
    manifest.json               run params + provenance

The next step is `scripts/create_hn_verify_task.py --bucket <name>` which reads
one bucket's selected/ subdir and uploads to CVAT with class translation.
"""
from __future__ import annotations

import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from ultralytics import YOLO

from ent_cv.config import DATA_DIR, LABELS
from ent_cv.gpu import gpu_yield

app = typer.Typer(add_completion=False)

PATIENT_CLASS = LABELS.index("Patient")
DEFAULT_MODEL = (
    DATA_DIR
    / "models"
    / "full_v0_20260515_205037"
    / "yolo11s_full_v0_img640_20260515_205043"
    / "weights"
    / "best.pt"
)
DEFAULT_CASES = "20260128_01,20260205_01,20260205_03"
RAW_DIR = DATA_DIR / "raw"
FRAMES_BATCH_DIR = DATA_DIR / "processed" / "extracted_frames" / "batch3"
PREDICTIONS_DIR = DATA_DIR / "predictions"


def _extract_frames(case: str, frame_interval: int) -> None:
    """Idempotent: call ent-cv data extract-frames once per case."""
    case_raw = RAW_DIR / case
    if not case_raw.is_dir():
        raise FileNotFoundError(f"raw case dir missing: {case_raw}")

    existing = list(FRAMES_BATCH_DIR.glob(f"{case}_Part*/*.png"))
    if existing:
        logger.info(f"  [{case}] {len(existing)} frames already extracted, skipping.")
        return

    logger.info(f"  [{case}] extracting frames (interval={frame_interval})...")
    FRAMES_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "ent-cv", "data", "extract-frames",
            str(case_raw), str(FRAMES_BATCH_DIR),
            "--frame-interval", str(frame_interval),
            "--format", "png",
        ],
        check=True,
    )


def _gather_frames(case_list: list[str]) -> list[Path]:
    out: list[Path] = []
    for case in case_list:
        case_frames = sorted(FRAMES_BATCH_DIR.glob(f"{case}_Part*/*.png"))
        logger.info(f"  [{case}] {len(case_frames)} frames")
        out.extend(case_frames)
    return out


def _yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _run_inference(
    images: list[Path],
    model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> dict[str, dict]:
    """stem -> {n_total, n_patient, max_non_patient_conf, lines, classes_present}"""
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path), task="detect")
    summaries: dict[str, dict] = {}

    with gpu_yield(device):
        results = model.predict(
            source=[str(p) for p in images],
            conf=conf, iou=iou, imgsz=imgsz, device=device,
            batch=1, stream=True, verbose=False, save=False,
        )
        # Ultralytics streams results in input order, but with source=[list,...]
        # the per-result `path` attribute is a generic placeholder ("image0",
        # "image1", ...). Zip with the input paths to recover real stems.
        for img_path, result in tqdm(
            zip(images, results, strict=True), total=len(images), desc="infer"
        ):
            stem = img_path.stem
            boxes = result.boxes
            n_total = 0 if boxes is None else int(boxes.shape[0])
            lines: list[str] = []
            confs: list[float] = []
            classes_present: Counter[int] = Counter()

            if n_total > 0:
                h, w = result.orig_shape
                xyxy = boxes.xyxy.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy()
                for i in range(n_total):
                    c = int(cls_arr[i])
                    classes_present[c] += 1
                    x1, y1, x2, y2 = xyxy[i]
                    lines.append(_yolo_line(c, x1, y1, x2, y2, w, h))
                    confs.append(float(conf_arr[i]))

            summaries[stem] = {
                "n_total": n_total,
                "n_patient": classes_present.get(PATIENT_CLASS, 0),
                "lines": lines,
                "confs": confs,
                "classes_present": dict(classes_present),
            }
    return summaries


BUCKETS = ("multi_patient", "uncertain", "confident", "silent")


def _classify(s: dict, conf_threshold: float) -> str:
    """Return one of BUCKETS. Order matters: multi_patient first (it overrides
    conf bucketing), then silent, then uncertain vs confident by min-conf."""
    if s["n_patient"] >= 2:
        return "multi_patient"
    if s["n_total"] == 0:
        return "silent"
    # min-conf: a frame with even one uncertain prediction goes to uncertain
    # because that uncertain box is most likely to be wrong (and is the one
    # the reviewer should look at).
    return "uncertain" if min(s["confs"]) < conf_threshold else "confident"


def _case_of(stem: str) -> str:
    return stem.rsplit("_Part", 1)[0]


def _select_for_bucket(
    pool: list[tuple[Path, dict]],
    cap: int,
    rng: random.Random,
) -> list[tuple[Path, dict]]:
    """Uniform per-case sample, then cap. Caps of 0 keep everything."""
    if cap <= 0 or len(pool) <= cap:
        return list(pool)
    by_case: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for t in pool:
        by_case[_case_of(t[0].stem)].append(t)
    per_case = max(1, cap // max(1, len(by_case)))
    out: list[tuple[Path, dict]] = []
    for items in by_case.values():
        rng.shuffle(items)
        out.extend(items[:per_case])
    rng.shuffle(out)
    return out[:cap]


def _wipe_dir(d: Path) -> None:
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()


@app.command()
def main(
    cases: str = typer.Option(DEFAULT_CASES, help="Comma-separated cases to mine."),
    model: Path = typer.Option(DEFAULT_MODEL, help="Detector to mine with."),
    frame_interval: int = typer.Option(3000, help="Extract every Nth frame from raw videos."),
    run_name: str = typer.Option(
        "full_v0_hn_candidates", help="Output dir name under predictions/."
    ),
    cap_multi_patient: int = typer.Option(0, help="Max multi-Patient frames (0 = keep all)."),
    cap_uncertain: int = typer.Option(150, help="Max uncertainty-zone frames."),
    cap_confident: int = typer.Option(0, help="Max confident-only frames (0 = keep all)."),
    cap_silent: int = typer.Option(0, help="Max silent frames (0 = keep all)."),
    uncertain_threshold: float = typer.Option(
        0.7,
        help="Min-conf < this puts a frame in 'uncertain'; otherwise 'confident'.",
    ),
    conf: float = typer.Option(0.25, help="Inference confidence threshold (low to catch borderlines)."),
    iou: float = typer.Option(0.7, help="NMS IoU threshold."),
    imgsz: int = typer.Option(640, help="Inference image size."),
    device: str = typer.Option("0", help="CUDA device or 'cpu'."),
    seed: int = typer.Option(0, help="Bucket-B sampling seed."),
    skip_inference: bool = typer.Option(
        False, help="Reuse inference_summaries.json; just re-bucket and re-select."
    ),
):
    """Mine HN + silent candidates for a combined CVAT labeling pass."""
    case_list = [c.strip() for c in cases.split(",") if c.strip()]
    if not case_list:
        raise typer.BadParameter("no cases provided")
    if not model.exists():
        raise typer.BadParameter(f"model not found: {model}")

    out_root = PREDICTIONS_DIR / run_name
    selected_root = out_root / "selected"
    out_root.mkdir(parents=True, exist_ok=True)
    selected_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting frames for {len(case_list)} cases:")
    for case in case_list:
        _extract_frames(case, frame_interval)

    all_frames = _gather_frames(case_list)
    if not all_frames:
        raise typer.BadParameter("no frames extracted — check raw video paths.")
    logger.info(f"Total frames: {len(all_frames)}")

    summaries_path = out_root / "inference_summaries.json"
    if skip_inference:
        if not summaries_path.exists():
            raise typer.BadParameter(
                f"--skip-inference set but no {summaries_path} — run once without it first."
            )
        summaries = json.loads(summaries_path.read_text())
        if any("confs" not in v for v in summaries.values()):
            raise typer.BadParameter(
                "cached summaries missing 'confs' field — re-run without --skip-inference."
            )
        logger.info(f"Loaded {len(summaries)} cached inference summaries.")
    else:
        summaries = _run_inference(all_frames, model, conf, iou, imgsz, device)
        summaries_path.write_text(json.dumps(summaries, indent=2))
        logger.info(f"Wrote inference summaries: {summaries_path}")

    by_bucket: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for img in all_frames:
        s = summaries.get(img.stem)
        if s is None:
            raise RuntimeError(f"missing inference summary for {img.stem}")
        by_bucket[_classify(s, uncertain_threshold)].append((img, s))

    logger.info(
        "Bucketing pool sizes:\n  "
        + "\n  ".join(f"{b:<14} {len(by_bucket.get(b, [])):>4}" for b in BUCKETS)
    )

    rng = random.Random(seed)
    caps = {
        "multi_patient": cap_multi_patient,
        "uncertain": cap_uncertain,
        "confident": cap_confident,
        "silent": cap_silent,
    }
    selected: dict[str, list[tuple[Path, dict]]] = {}
    for b in BUCKETS:
        selected[b] = _select_for_bucket(by_bucket.get(b, []), caps[b], rng)
    logger.info(
        "Selected:\n  "
        + "\n  ".join(f"{b:<14} {len(selected[b]):>4}" for b in BUCKETS)
    )

    tsv_lines = [
        "stem\tbucket\tn_total\tn_patient\tmin_conf\tmax_conf\tclasses_present"
    ]
    for b in BUCKETS:
        images_dir = selected_root / b / "images"
        preload_dir = selected_root / b / "preload"
        images_dir.mkdir(parents=True, exist_ok=True)
        preload_dir.mkdir(parents=True, exist_ok=True)
        _wipe_dir(images_dir)
        _wipe_dir(preload_dir)
        write_preload = b != "silent"
        for img, s in selected[b]:
            (images_dir / img.name).symlink_to(img.resolve())
            if write_preload and s["lines"]:
                (preload_dir / f"{img.stem}.txt").write_text(
                    "\n".join(s["lines"]) + "\n"
                )
            min_conf = min(s["confs"]) if s["confs"] else 0.0
            max_conf = max(s["confs"]) if s["confs"] else 0.0
            tsv_lines.append(
                f"{img.stem}\t{b}\t{s['n_total']}\t{s['n_patient']}\t"
                f"{min_conf:.3f}\t{max_conf:.3f}\t{json.dumps(s['classes_present'])}"
            )
    (out_root / "candidates.tsv").write_text("\n".join(tsv_lines) + "\n")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_name": run_name,
        "cases": case_list,
        "model": str(model),
        "frame_interval": frame_interval,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "uncertain_threshold": uncertain_threshold,
        "caps": caps,
        "seed": seed,
        "n_extracted_frames": len(all_frames),
        "pool_sizes": {b: len(by_bucket.get(b, [])) for b in BUCKETS},
        "selected_sizes": {b: len(selected[b]) for b in BUCKETS},
        "label_names": dict(enumerate(LABELS)),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.success(
        f"HN candidates ready: {out_root}\n"
        + "\n".join(
            f"  {b:<14} {len(selected[b]):>4} frames -> "
            f"{selected_root / b}"
            for b in BUCKETS
        )
        + "\nNext: create CVAT task per bucket — "
        "scripts/create_hn_verify_task.py --bucket <name>"
    )


if __name__ == "__main__":
    app()
