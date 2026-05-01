"""Compare all trained models in MODELS_DIR and rank them by validation metrics.

Usage:
    uv run -m ent_cv.modeling.compare_models
    uv run -m ent_cv.modeling.compare_models path/to/compare_models.yaml
"""

import csv
from pathlib import Path
import shutil
from typing import Optional

from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table
import typer
import yaml

from ent_cv.config import MODELS_DIR

app = typer.Typer()
console = Console()

SORT_KEYS = {
    "map50":     "map50",
    "map50-95":  "map50_95",
    "map":       "map50_95",
    "precision": "precision",
    "recall":    "recall",
}
_SORT_HELP = "Metric to rank by: map50 | map50-95 | precision | recall"

# Arg keys to hide from --diff (metadata, IO paths, run identity, display flags).
_DIFF_EXCLUDE = {
    "save_dir", "name", "project", "data", "task", "mode", "save", "plots",
    "show", "visualize", "save_frames", "save_conf", "save_crop",
    "show_labels", "show_conf", "show_boxes", "line_width", "format",
    "keras", "optimize", "int8", "dynamic", "simplify", "opset", "workspace",
    "nms", "verbose", "save_txt", "save_json", "save_period", "exist_ok",
    "device", "tracker", "embed", "retina_masks", "dnn", "half",
    "stream_buffer", "vid_stride", "source", "resume", "cfg", "profile",
    "compile",
}

_TYPE_STYLE = {
    "scratch":    "blue",
    "pretrained": "cyan",
    "fine-tune":  "magenta",
    "unknown":    "dim",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _detect_model_type(base_model_arg: str) -> tuple[str, Optional[str]]:
    """Classify a training run by its ``model:`` arg.

    Returns (model_type, parent_run). parent_run is only set for fine-tune runs.

    - ``.yaml`` → ``scratch`` (random init from architecture file)
    - ``.pt`` with a path containing ``models/`` → ``fine-tune`` (local weights)
    - ``.pt`` without such a path → ``pretrained`` (Ultralytics COCO download)
    """
    base_str = str(base_model_arg)
    if base_str.endswith(".yaml"):
        return "scratch", None
    if "/" in base_str and "models" in base_str:
        parts = Path(base_str).parts
        parent = None
        for i, part in enumerate(parts):
            if part == "models" and i + 1 < len(parts):
                parent = parts[i + 1]
                break
        if parent is None:
            parent = Path(base_str).parent.parent.name
        return "fine-tune", parent
    if base_str.endswith(".pt"):
        return "pretrained", None
    return "unknown", None

def _read_nc(model_dir: Path) -> int | None:
    """Read number of classes from the best.pt checkpoint."""
    weights = model_dir / "weights" / "best.pt"
    if not weights.exists():
        return None
    try:
        import torch

        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        model = ckpt.get("model")
        if hasattr(model, "nc"):
            return int(model.nc)
        if hasattr(model, "names"):
            return len(model.names)
    except Exception:
        return None
    return None


def _parse_args_yaml(model_dir: Path) -> dict:
    args_file = model_dir / "args.yaml"
    if not args_file.exists():
        return {}
    with args_file.open() as f:
        return yaml.safe_load(f) or {}


def _parse_results_csv(model_dir: Path) -> dict:
    """Return metrics at the best mAP50 epoch plus total epochs from results.csv."""
    results_file = model_dir / "results.csv"
    if not results_file.exists():
        return {}

    best: dict = {}
    best_map50 = -1.0
    last_row: dict = {}

    with results_file.open(newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [k.strip() for k in (reader.fieldnames or [])]
        for row in reader:
            try:
                map50 = float(row.get("metrics/mAP50(B)", "nan").strip())
            except ValueError:
                continue
            last_row = dict(row)
            if map50 > best_map50:
                best_map50 = map50
                best = dict(row)

    def _clean(d: dict) -> dict:
        out: dict = {}
        for k, v in d.items():
            try:
                out[k.strip()] = float(v.strip()) if v.strip() else None
            except (ValueError, AttributeError):
                out[k.strip()] = v
        return out

    result = _clean(best)
    if last_row:
        try:
            result["_total_epochs"] = int(float(last_row.get("epoch", "0").strip()))
        except (ValueError, AttributeError):
            pass
    if best:
        try:
            result["_best_epoch"] = int(float(best.get("epoch", "0").strip()))
        except (ValueError, AttributeError):
            pass
    return result


def _collect_model_info(model_dir: Path) -> Optional[dict]:
    """Return a dict of info for one model directory, or None if no weights exist."""
    if not (model_dir / "weights" / "best.pt").exists():
        return None

    args = _parse_args_yaml(model_dir)
    metrics = _parse_results_csv(model_dir)

    base_model_arg = args.get("model", "unknown")
    model_type, parent_run = _detect_model_type(str(base_model_arg))
    is_finetune = model_type == "fine-tune"
    base_stem = Path(str(base_model_arg)).stem or str(base_model_arg)

    return {
        "name":          model_dir.name,
        "path":          model_dir,
        "base_model":    base_stem,
        "model_type":    model_type,
        "is_finetune":   is_finetune,  # kept for backward-compat inside this module
        "parent_run":    parent_run,
        "nc":            _read_nc(model_dir),
        "epochs_trained": metrics.get("_total_epochs") or int(args.get("epochs", 0) or 0),
        "best_epoch":    metrics.get("_best_epoch"),
        "epochs_config": int(args.get("epochs", 0) or 0),
        "imgsz":         int(args.get("imgsz", 0) or 0),
        "batch":         args.get("batch", "?"),
        "rect":          args.get("rect", False),
        "scale":         args.get("scale", 0.0),
        "optimizer":     args.get("optimizer", "auto"),
        "dataset":       (Path(str(args.get("data", ""))).parent.name if args.get("data") else "?"),
        "map50":         metrics.get("metrics/mAP50(B)"),
        "map50_95":      metrics.get("metrics/mAP50-95(B)"),
        "precision":     metrics.get("metrics/precision(B)"),
        "recall":        metrics.get("metrics/recall(B)"),
        "val_box_loss":  metrics.get("val/box_loss"),
        "val_cls_loss":  metrics.get("val/cls_loss"),
        "evaluated":     (model_dir / "evaluation" / "run").exists(),
        "raw_args":      args,
        "raw_metrics":   metrics,
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(models_dir: Path, sort_by: str) -> list[dict]:
    """Scan models_dir recursively, collect info, and return all models sorted best→worst."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models: list[dict] = []
    for best_pt in sorted(models_dir.rglob("weights/best.pt")):
        model_dir = best_pt.parent.parent
        info = _collect_model_info(model_dir)
        if info is None:
            continue
        # Derive source from relative path (e.g. "sweep_.../phase1_scout" or "" for standalone)
        rel = model_dir.relative_to(models_dir)
        parts = rel.parts
        info["source"] = str(Path(*parts[:-1])) if len(parts) > 1 else ""
        models.append(info)

    if not models:
        raise RuntimeError(f"No valid models found in {models_dir}")

    key = SORT_KEYS.get(sort_by.lower(), "map50")
    return sorted(models, key=lambda m: (m.get(key) or -1), reverse=True)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 4) -> str:
    if val is None or val != val:
        return "\u2014"
    return f"{val:.{decimals}f}"


def _make_ranking_table(ranked: list[dict], sort_by: str = "map50") -> Table:
    table = Table(
        title=f"Model Comparison \u2014 ranked by {sort_by.upper()}",
        box=box.ROUNDED, header_style="bold cyan",
        show_lines=True, expand=True,
    )
    table.add_column("#",        style="bold",         justify="right",  width=3,  no_wrap=True)
    table.add_column("Model",    style="white",         min_width=20)
    table.add_column("Source",                          min_width=12)
    table.add_column("Type",                            justify="center", width=10, no_wrap=True)
    table.add_column("mAP50",    style="green bold",    justify="right",  min_width=8,  no_wrap=True)
    table.add_column("mAP50-95", style="green",         justify="right",  min_width=9,  no_wrap=True)
    table.add_column("Prec",                            justify="right",  min_width=8,  no_wrap=True)
    table.add_column("Recall",                          justify="right",  min_width=8,  no_wrap=True)
    table.add_column("Ep (t/b)",                        justify="right",  width=9,  no_wrap=True)
    table.add_column("Imgsz",                           justify="right",  width=6,  no_wrap=True)
    table.add_column("NC",                              justify="right",  width=4,  no_wrap=True)
    table.add_column("Dataset",                         min_width=12)

    best_map50 = ranked[0].get("map50") if ranked else None

    for rank, m in enumerate(ranked, 1):
        map50_val = m.get("map50")
        is_best = map50_val is not None and map50_val == best_map50
        model_type = m.get("model_type", "unknown")
        type_style = _TYPE_STYLE.get(model_type, "dim")
        best_ep = m.get("best_epoch")
        ep_str = str(m["epochs_trained"]) + (f"/{best_ep}" if best_ep and best_ep != m["epochs_trained"] else "")
        source = m.get("source", "")
        table.add_row(
            str(rank), m["name"], source or "\u2014",
            f"[{type_style}]{model_type}[/{type_style}]",
            _fmt(map50_val), _fmt(m.get("map50_95")),
            _fmt(m.get("precision")), _fmt(m.get("recall")),
            ep_str, str(m["imgsz"]),
            str(m["nc"]) if m.get("nc") is not None else "\u2014",
            m["dataset"],
            style="bold yellow" if is_best else "",
        )
    return table


def _make_detail_table(models: list[dict]) -> Table:
    table = Table(
        title="Training Configuration Details",
        box=box.SIMPLE_HEAD, header_style="bold cyan",
        show_lines=False, expand=True,
    )
    table.add_column("Model",      min_width=20)
    table.add_column("Source",     min_width=12)
    table.add_column("Base",       width=14,  no_wrap=True)
    table.add_column("Parent run", min_width=20)
    table.add_column("Ep cfg",     justify="right", width=7,  no_wrap=True)
    table.add_column("Batch",      justify="right", width=6,  no_wrap=True)
    table.add_column("Rect",       justify="center", width=5, no_wrap=True)
    table.add_column("Scale",      justify="right", width=7,  no_wrap=True)
    table.add_column("Optim",      width=8, no_wrap=True)
    table.add_column("Eval?",      justify="center", width=6, no_wrap=True)
    for m in models:
        parent = m.get("parent_run") or "\u2014"
        if len(parent) > 32:
            parent = parent[:29] + "..."
        source = m.get("source", "")
        table.add_row(
            m["name"], source or "\u2014", m["base_model"], parent,
            str(m["epochs_config"]), str(m["batch"]),
            "\u2713" if m["rect"] else "\u2717",
            _fmt(m.get("scale"), 2), str(m["optimizer"]),
            "[green]\u2713[/green]" if m["evaluated"] else "[dim]\u2014[/dim]",
        )
    return table


def _print_summary(ranked: list[dict]) -> None:
    console.rule("[bold]Summary & Recommendation")
    console.print()
    if not ranked:
        console.print("[red]No models found.[/red]")
        return

    best, worst = ranked[0], ranked[-1]
    by_type: dict[str, list[dict]] = {}
    for m in ranked:
        by_type.setdefault(m.get("model_type", "unknown"), []).append(m)

    console.print(f"  [bold green]Best:[/bold green]  [yellow]{best['name']}[/yellow]")
    console.print(f"    mAP@50:    {_fmt(best['map50'])}")
    console.print(f"    mAP@50-95: {_fmt(best['map50_95'])}")
    console.print(f"    Precision: {_fmt(best['precision'])}")
    console.print(f"    Recall:    {_fmt(best['recall'])}")
    console.print(f"    Trained:   {best['epochs_trained']} epochs @ {best['imgsz']}px")
    if best["is_finetune"]:
        console.print(f"    From:      [cyan]{best['parent_run']}[/cyan]")
    console.print()
    console.print(f"  [bold red]Weakest:[/bold red]  {worst['name']}  (mAP50={_fmt(worst['map50'])})")
    console.print()
    type_counts = "  ".join(
        f"[{_TYPE_STYLE.get(t, 'dim')}]{len(ms)} {t}[/{_TYPE_STYLE.get(t, 'dim')}]"
        for t, ms in by_type.items()
    )
    console.print(f"  [bold]Total:[/bold] {len(ranked)}  ({type_counts})")

    nc_values = {m["nc"] for m in ranked if m.get("nc") is not None}
    if len(nc_values) > 1:
        console.print(
            f"  [bold yellow]Mixed class counts across models: "
            f"{sorted(nc_values)}[/bold yellow]"
        )
        console.print(
            "    Metrics are not directly comparable between models "
            "trained on different numbers of classes.\n"
        )

    if len(by_type) > 1:
        lines = []
        for t, ms in by_type.items():
            vals = [m["map50"] for m in ms if m["map50"] is not None]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            style = _TYPE_STYLE.get(t, "dim")
            lines.append(f"[{style}]{t}: {avg:.4f}[/{style}]")
        if lines:
            console.print("  Avg mAP50 — " + "  ".join(lines))
    console.print()

    # Fine-tune lineage
    def _chain_lines(m: dict, depth: int = 0) -> list[str]:
        lines = [" " * (depth * 2) + f"\u2192 {m['name']}  (mAP50={_fmt(m['map50'])})"]
        for child in sorted([c for c in ranked if c.get("parent_run") == m["name"]], key=lambda x: x["name"]):
            lines.extend(_chain_lines(child, depth + 1))
        return lines

    roots = [m for m in ranked if any(c.get("parent_run") == m["name"] for c in ranked)]
    if roots:
        console.print("  [bold]Fine-tune lineage:[/bold]")
        for root in roots:
            for line in _chain_lines(root):
                console.print("   " + line)
        console.print()
    console.rule()


# ---------------------------------------------------------------------------
# Diff view & CSV export
# ---------------------------------------------------------------------------

def _fmt_arg(value) -> str:
    if value is None:
        return "\u2014"
    if isinstance(value, bool):
        return "\u2713" if value else "\u2717"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _find_differing_args(models: list[dict]) -> list[str]:
    """Return sorted arg keys whose values differ across *models*."""
    all_keys: set[str] = set()
    for m in models:
        all_keys |= set(m.get("raw_args", {}).keys())
    all_keys -= _DIFF_EXCLUDE

    differing: list[str] = []
    _missing = object()
    for k in sorted(all_keys):
        vals = set()
        for m in models:
            v = m.get("raw_args", {}).get(k, _missing)
            vals.add(repr(v))
        if len(vals) > 1:
            differing.append(k)
    return differing


def _make_diff_table(models: list[dict]) -> Optional[Table]:
    """Table showing only hyperparameters that differ across runs."""
    keys = _find_differing_args(models)
    if not keys:
        return None

    table = Table(
        title=f"Hyperparameter diff \u2014 {len(keys)} param{'s' if len(keys) != 1 else ''} differ across {len(models)} run{'s' if len(models) != 1 else ''}",
        box=box.SIMPLE_HEAD, header_style="bold cyan",
        show_lines=False, expand=True,
    )
    table.add_column("Model", min_width=20)
    for k in keys:
        table.add_column(k, justify="right", no_wrap=True)

    for m in models:
        args = m.get("raw_args", {})
        row = [m["name"]] + [_fmt_arg(args.get(k)) for k in keys]
        table.add_row(*row)
    return table


def _write_csv(models: list[dict], csv_path: Path, sort_by: str) -> int:
    """Write one CSV row per model with all metrics and args. Returns row count."""
    core_cols = [
        "rank", "name", "source", "path", "model_type", "base_model",
        "parent_run", "nc", "dataset", "imgsz", "map50", "map50_95",
        "precision", "recall", "epochs_trained", "best_epoch", "epochs_config",
    ]

    metric_keys: set[str] = set()
    arg_keys: set[str] = set()
    for m in models:
        metric_keys |= {k for k in m.get("raw_metrics", {}).keys() if not k.startswith("_")}
        arg_keys |= set(m.get("raw_args", {}).keys())

    metric_cols = sorted(metric_keys)
    arg_cols = sorted(arg_keys)
    all_cols = core_cols + metric_cols + [f"arg.{k}" for k in arg_cols]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_cols)
        for rank, m in enumerate(models, 1):
            row = [
                rank,
                m["name"],
                m.get("source", ""),
                str(m["path"]),
                m.get("model_type", ""),
                m.get("base_model", ""),
                m.get("parent_run") or "",
                m.get("dataset", ""),
                m.get("imgsz", ""),
                m.get("map50") if m.get("map50") is not None else "",
                m.get("map50_95") if m.get("map50_95") is not None else "",
                m.get("precision") if m.get("precision") is not None else "",
                m.get("recall") if m.get("recall") is not None else "",
                m.get("epochs_trained", ""),
                m.get("best_epoch") or "",
                m.get("epochs_config", ""),
            ]
            raw_metrics = m.get("raw_metrics", {})
            for k in metric_cols:
                v = raw_metrics.get(k)
                row.append("" if v is None else v)
            raw_args = m.get("raw_args", {})
            for k in arg_cols:
                v = raw_args.get(k)
                row.append("" if v is None else v)
            writer.writerow(row)
    return len(models)


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _prompt_and_delete(all_ranked: list[dict], default_n: int = 0) -> None:
    """Prompt the user to delete the worst N models or keep only the top N."""
    total = len(all_ranked)
    if total == 0:
        return

    parent_names = {m["parent_run"] for m in all_ranked if m.get("parent_run")}

    console.print()
    console.rule("[bold red]Cleanup")
    console.print()
    console.print(
        f"  [bold]{total}[/bold] models ranked best\u2192worst.\n"
        "   [green]0[/green]   keep all (default)\n"
        "   [green]+N[/green]  keep top N, delete rest\n"
        "   [red]-N[/red]  delete N worst"
    )
    console.print()

    while True:
        raw = typer.prompt("  Keep/delete", default=str(default_n),
                           prompt_suffix=" [0]: ", show_default=False).strip()
        try:
            n = int(raw)
        except ValueError:
            console.print("  [red]Please enter a whole number.[/red]")
            continue

        if n == 0:
            console.print("  No models deleted.")
            return
        elif n > 0:
            if n >= total:
                console.print(f"  [red]Must be < total models ({total}).[/red]")
                continue
            to_delete = all_ranked[n:]
            action_desc = f"keep top {n}, delete {total - n} worst"
        else:
            delete_count = abs(n)
            if delete_count > total:
                console.print(f"  [red]|N| cannot exceed total models ({total}).[/red]")
                continue
            to_delete = all_ranked[total - delete_count:]
            action_desc = f"delete {delete_count} worst"
        break

    console.print()
    console.print(f"  [bold red]To delete — {action_desc}:[/bold red]")
    for m in to_delete:
        warn = "  [bold yellow]\u26a0 parent of a fine-tune[/bold yellow]" if m["name"] in parent_names else ""
        console.print(f"    [red]\u2717[/red] {m['name']}  (mAP50={_fmt(m['map50'])}){warn}")

    parent_warnings = [m for m in to_delete if m["name"] in parent_names]
    if parent_warnings:
        console.print()
        console.print(
            "  [bold yellow]Warning:[/bold yellow] deleting a parent does not affect already-trained "
            "children, but you will lose the weights used as their starting point."
        )

    console.print()
    if typer.prompt("  Type \'yes\' to confirm", default="no").strip().lower() != "yes":
        console.print("  Cancelled.")
        return

    for m in to_delete:
        try:
            shutil.rmtree(m["path"])
            logger.info(f"Deleted: {m['path']}")
            console.print(f"  [red]Deleted:[/red] {m['path']}")
        except Exception as exc:
            logger.error(f"Failed to delete {m['path']}: {exc}")
            console.print(f"  [red]Error:[/red] {exc}")

    console.print()
    console.print(f"  [bold green]Done.[/bold green] Deleted {len(to_delete)} model(s).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    models_dir: Path = typer.Option(MODELS_DIR, help="Directory containing trained model runs"),
    sort_by: str = typer.Option("map50", help="Metric to rank by: map50 | map50-95 | precision | recall"),
    verbose: bool = typer.Option(False, help="Show detailed training config table"),
    diff: bool = typer.Option(False, help="Show only hyperparameters that differ across runs"),
    csv_out: Optional[Path] = typer.Option(
        None, "--csv", help="Write full results + all parameters to this CSV file",
    ),
    weights_suffix: str = typer.Option("best", help="Weights file stem (best or last)"),
    delete: int = typer.Option(0, help="Default cleanup suggestion (0=keep all, +N=keep top N, -N=delete N worst)"),
    top: Optional[int] = typer.Option(None, help="Only display top N models"),
) -> None:
    """Scan MODELS_DIR, compare all trained models, and print a ranked leaderboard."""
    models_dir = Path(models_dir)
    logger.info(f"Scanning {models_dir} …")
    ranked = run(models_dir=models_dir, sort_by=sort_by)
    logger.info(f"Found {len(ranked)} models")
    console.print()

    display = ranked[:top] if top else ranked
    console.print(_make_ranking_table(display, sort_by=sort_by))
    console.print()

    if verbose:
        console.print(_make_detail_table(display))
        console.print()

    if diff:
        diff_table = _make_diff_table(display)
        if diff_table is None:
            console.print("[dim]No differing hyperparameters across displayed runs.[/dim]")
        else:
            console.print(diff_table)
        console.print()

    if csv_out is not None:
        n = _write_csv(ranked, csv_out, sort_by=sort_by)
        console.print(f"[green]Wrote {n} rows to[/green] [yellow]{csv_out}[/yellow]")
        console.print()

    _print_summary(ranked)

    console.print("[bold cyan]Recommended weights:[/bold cyan]")
    console.print(f"  [yellow]{ranked[0]['path'] / 'weights' / (weights_suffix + '.pt')}[/yellow]")

    _prompt_and_delete(ranked, default_n=delete)


if __name__ == "__main__":
    app()
