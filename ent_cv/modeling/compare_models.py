"""Compare all trained models in MODELS_DIR and rank them by validation metrics.

Usage:
    uv run -m ent_cv.modeling.compare_models
    uv run -m ent_cv.modeling.compare_models path/to/compare_models.yaml
"""

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@dataclass
class CompareConfig:
    models_dir: Path
    sort_by: str
    verbose: bool
    weights_suffix: str
    delete: int
    top: Optional[int] = None

    def __post_init__(self):
        self.models_dir = Path(self.models_dir)


def _load_config(config_file: Path) -> CompareConfig:
    with open(config_file) as f:
        d = yaml.safe_load(f) or {}
    known = {k: v for k, v in d.items() if k in CompareConfig.__dataclass_fields__}
    if unknown := set(d) - set(CompareConfig.__dataclass_fields__):
        logger.warning(f"Ignoring unknown config keys: {unknown}")
    return CompareConfig(**known)


_DEFAULT_CONFIG = Path("ent_cv/modeling/configs/compare_models.yaml")

SORT_KEYS = {
    "map50":     "map50",
    "map50-95":  "map50_95",
    "map":       "map50_95",
    "precision": "precision",
    "recall":    "recall",
}
_SORT_HELP = "Metric to rank by: map50 | map50-95 | precision | recall"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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
    is_finetune = "/" in str(base_model_arg) and "models" in str(base_model_arg)

    parent_run = None
    if is_finetune:
        parts = Path(str(base_model_arg)).parts
        for i, part in enumerate(parts):
            if part == "models" and i + 1 < len(parts):
                parent_run = parts[i + 1]
                break
        if parent_run is None:
            parent_run = Path(str(base_model_arg)).parent.parent.name

    return {
        "name":          model_dir.name,
        "path":          model_dir,
        "base_model":    (Path(str(base_model_arg)).stem if is_finetune
                          else str(base_model_arg).replace(".yaml", "").replace(".pt", "")),
        "is_finetune":   is_finetune,
        "parent_run":    parent_run,
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
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _run(models_dir: Path, sort_by: str) -> list[dict]:
    """Scan models_dir, collect info, and return all models sorted best→worst."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models: list[dict] = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        info = _collect_model_info(d)
        if info is None:
            logger.debug(f"Skipping {d.name} — no best.pt")
            continue
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
    table.add_column("Type",                            justify="center", width=10, no_wrap=True)
    table.add_column("mAP50",    style="green bold",    justify="right",  min_width=8,  no_wrap=True)
    table.add_column("mAP50-95", style="green",         justify="right",  min_width=9,  no_wrap=True)
    table.add_column("Prec",                            justify="right",  min_width=8,  no_wrap=True)
    table.add_column("Recall",                          justify="right",  min_width=8,  no_wrap=True)
    table.add_column("Ep (t/b)",                        justify="right",  width=9,  no_wrap=True)
    table.add_column("Imgsz",                           justify="right",  width=6,  no_wrap=True)
    table.add_column("Dataset",                         min_width=12)

    best_map50 = ranked[0].get("map50") if ranked else None

    for rank, m in enumerate(ranked, 1):
        map50_val = m.get("map50")
        is_best = map50_val is not None and map50_val == best_map50
        type_style = "magenta" if m["is_finetune"] else "blue"
        model_type = "fine-tune" if m["is_finetune"] else "scratch"
        best_ep = m.get("best_epoch")
        ep_str = str(m["epochs_trained"]) + (f"/{best_ep}" if best_ep and best_ep != m["epochs_trained"] else "")
        table.add_row(
            str(rank), m["name"],
            f"[{type_style}]{model_type}[/{type_style}]",
            _fmt(map50_val), _fmt(m.get("map50_95")),
            _fmt(m.get("precision")), _fmt(m.get("recall")),
            ep_str, str(m["imgsz"]), m["dataset"],
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
        table.add_row(
            m["name"], m["base_model"], parent,
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
    finetunes = [m for m in ranked if m["is_finetune"]]
    scratch   = [m for m in ranked if not m["is_finetune"]]

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
    console.print(f"  [bold]Total:[/bold] {len(ranked)}  "
                  f"([magenta]{len(finetunes)} fine-tune[/magenta], "
                  f"[blue]{len(scratch)} from scratch[/blue])")

    if finetunes and scratch:
        ft_avg = sum(m["map50"] for m in finetunes if m["map50"] is not None) / len(finetunes)
        sc_avg = sum(m["map50"] for m in scratch if m["map50"] is not None) / len(scratch)
        console.print(f"  Avg mAP50 — fine-tunes: [magenta]{ft_avg:.4f}[/magenta]  "
                      f"from-scratch: [blue]{sc_avg:.4f}[/blue]")
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
def main(config_file: Path = typer.Argument(_DEFAULT_CONFIG, help="Path to YAML config")):
    """Scan MODELS_DIR, compare all trained models, and print a ranked leaderboard."""
    cfg = _load_config(config_file)
    logger.info(f"Scanning {cfg.models_dir} …")
    ranked = _run(models_dir=cfg.models_dir, sort_by=cfg.sort_by)
    logger.info(f"Found {len(ranked)} models")
    console.print()

    display = ranked[:cfg.top] if cfg.top else ranked
    console.print(_make_ranking_table(display, sort_by=cfg.sort_by))
    console.print()

    if cfg.verbose:
        console.print(_make_detail_table(display))
        console.print()

    _print_summary(ranked)

    console.print("[bold cyan]Recommended weights:[/bold cyan]")
    console.print(f"  [yellow]{ranked[0]['path'] / 'weights' / (cfg.weights_suffix + '.pt')}[/yellow]")

    _prompt_and_delete(ranked, default_n=cfg.delete)


if __name__ == "__main__":
    app()
