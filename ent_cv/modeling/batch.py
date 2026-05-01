"""Run a modeling operation over multiple configurations from a YAML file.

Batch config format — two styles are supported:

  Style A: reference individual config files
    operation: predict
    configs:
      - configs/predict_case1.yaml
      - configs/predict_case2.yaml

  Style B: inline runs with shared defaults
    operation: predict
    defaults:
      weights: /mnt/data/ent_cv/models/best/weights/best.pt
      conf: 0.647
    runs:
      - source: /mnt/data/ent_cv/raw/20260205_01
      - source: /mnt/data/ent_cv/raw/20260208_01

Usage:
    uv run -m ent_cv.modeling.batch configs/batch_predict.yaml
"""
from pathlib import Path
from typing import Any, Optional

from loguru import logger
import typer
import yaml

from ent_cv.config import TRAIN_METRIC_KEYS
from ent_cv.utils import notify

app = typer.Typer(add_completion=False)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_runs(cfg: dict, config_file: Path) -> list[dict]:
    if "configs" in cfg:
        base_dir = config_file.parent
        return [_load_yaml(p if Path(p).is_absolute() else base_dir / p) for p in cfg["configs"]]
    defaults = cfg.get("defaults", {})
    base_runs = cfg.get("runs") or []
    if not base_runs:
        raise ValueError("Batch config requires either 'configs' or 'runs'")
    return [{**defaults, **run} for run in base_runs]


def _fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "N/A"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _predict_summary(results: list[dict]) -> str:
    col_w, src_w = 10, 45
    header = f"  {'#':<4} {'Source':<{src_w}} {'Frames':>{col_w}} {'Changes':>{col_w}} {'CaseTime':>{col_w}}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in results:
        raw_s = r.get("summary")
        s = {"total_frames": raw_s} if isinstance(raw_s, int) else (raw_s or {})
        src = str(r.get("source", ""))[-src_w:]
        note = " [skipped]" if r.get("skipped") else (f" [ERR: {r['error'][:30]}]" if r.get("error") else "")
        lines.append(
            f"  {r['run']:<4} {src:<{src_w}}"
            f"{str(s.get('total_frames', 'N/A')):>{col_w}}"
            f"{str(s.get('label_change_count', 'N/A')):>{col_w}}"
            f"{_fmt_time(s.get('total_case_time_sec')):>{col_w}}{note}"
        )
    return "\n".join(lines)


def _train_summary(results: list[dict]) -> str:
    col_w, name_w = 10, 45
    labels = [label for label, _ in TRAIN_METRIC_KEYS]
    header = f"  {'#':<4} {'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in labels)
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in results:
        name = str(r.get("name", f"run_{r['run']}"))[-name_w:]
        row = f"  {r['run']:<4} {name:<{name_w}}"
        for label in labels:
            v = r.get(label)
            row += f"{v:>{col_w}.4f}" if isinstance(v, float) else f"{'N/A':>{col_w}}"
        if r.get("error"):
            row += f" [ERR: {r['error'][:30]}]"
        lines.append(row)
    lines += ["", "  Best:"]
    for label in labels:
        candidates = [(r["run"], r.get("name", ""), r[label]) for r in results if isinstance(r.get(label), float)]
        if candidates:
            best = max(candidates, key=lambda x: x[2])
            lines.append(f"    {label:<12} Run {best[0]} — {best[1][:35]}  ({best[2]:.4f})")
    return "\n".join(lines)


def _run_predict(run_cfg: dict) -> dict:
    from ent_cv.modeling.predict import run
    result = run(**run_cfg)
    entry: dict = {"source": str(run_cfg.get("source", ""))}
    if result is None:
        entry["skipped"] = True
        entry["summary"] = None
    else:
        out_dir, summary = result
        entry.update({"output_dir": str(out_dir), "summary": summary})
    return entry


def _run_postprocess(run_cfg: dict) -> dict:
    from ent_cv.modeling.postprocess import postprocess
    changes = postprocess(**run_cfg)
    return {"source": str(run_cfg.get("raw_json", "")), "changes": changes}


def _run_train(run_cfg: dict) -> dict:
    from ent_cv.modeling.train import run
    result = run(**run_cfg)
    rd = getattr(result, "results_dict", {})
    save_dir = Path(str(getattr(result, "save_dir", "")))
    entry: dict = {"name": save_dir.name or "?"}
    for label, key in TRAIN_METRIC_KEYS:
        entry[label] = rd.get(key)
    return entry


def _postprocess_summary(results: list[dict]) -> str:
    col_w, src_w = 10, 45
    header = f"  {'#':<4} {'Source':<{src_w}} {'Classes kept':>{col_w}} {'Dropped%':>{col_w}}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in results:
        src = str(r.get("source", ""))[-src_w:]
        changes = r.get("changes") or {}
        if r.get("error"):
            lines.append(f"  {r['run']:<4} {src:<{src_w}} [ERR: {r['error'][:30]}]")
            continue
        total_raw = sum(v["raw_frames"] for v in changes.values())
        total_filt = sum(v["filtered_frames"] for v in changes.values())
        dropped_pct = f"{100 * (total_raw - total_filt) / total_raw:.1f}%" if total_raw else "N/A"
        lines.append(
            f"  {r['run']:<4} {src:<{src_w}}"
            f"{str(len(changes)):>{col_w}}"
            f"{dropped_pct:>{col_w}}"
        )
    return "\n".join(lines)


def _batch_results_fn(result: Any) -> tuple[str, list[Path]]:
    n = result.get("n_runs", "?")
    lines = [f"  Completed runs: {n}"]
    if summary_text := result.get("summary_text", ""):
        lines += ["", summary_text]
    return "\n".join(lines), []


@app.command()
@notify("Batch", results_fn=_batch_results_fn)
def main(config_file: Path = typer.Argument(..., help="Path to YAML batch config")) -> dict:
    """Run a modeling operation over multiple configurations."""
    if not config_file.exists():
        logger.error(f"Config not found: {config_file}")
        raise typer.Exit(1)

    cfg = _load_yaml(config_file)
    operation = cfg.get("operation", "").lower()
    if operation not in {"predict", "train", "postprocess", "export", "tune"}:
        logger.error(f"Unknown operation '{operation}'. Must be: predict | train | postprocess | export | tune")
        raise typer.Exit(1)

    runs = _build_runs(cfg, config_file)
    logger.info(f"Starting {operation} batch — {len(runs)} run(s)")
    run_results: list[dict] = []

    for i, run_cfg in enumerate(runs, 1):
        logger.info(f"[{i}/{len(runs)}] {run_cfg}")
        try:
            if operation == "predict":
                entry = _run_predict(run_cfg)
            elif operation == "train":
                entry = _run_train(run_cfg)
            elif operation == "postprocess":
                entry = _run_postprocess(run_cfg)
            else:
                logger.warning(f"Operation '{operation}' not yet supported in batch mode.")
                entry = {"result": "skipped"}
            entry["run"] = str(i)
            run_results.append(entry)
        except Exception as exc:
            logger.error(f"Run {i} failed: {exc}")
            run_results.append({"run": i, "source": str(run_cfg.get("source", "")), "error": str(exc)})

    if operation == "predict":
        summary_text = _predict_summary(run_results)
    elif operation == "train":
        summary_text = _train_summary(run_results)
    elif operation == "postprocess":
        summary_text = _postprocess_summary(run_results)
    else:
        summary_text = "\n".join(f"  Run {r['run']}: {r.get('result', r.get('error', 'OK'))}" for r in run_results)

    logger.info(f"\n{'=' * 60}\nBATCH SUMMARY\n{'=' * 60}\n{summary_text}")
    logger.success(f"Batch complete — {len(runs)} run(s)")
    return {"n_runs": len(runs), "run_results": run_results, "summary_text": summary_text}


if __name__ == "__main__":
    app()
