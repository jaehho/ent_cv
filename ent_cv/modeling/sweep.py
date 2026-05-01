"""Fully automated multi-phase training sweep.

Three phases run sequentially with no manual intervention:

  Phase 1 — Architecture Scout:  short training across architectures and sizes
  Phase 2 — Hyperparameter Sweep: hparam variations on the top combos
  Phase 3 — Final Training:       full training on the overall winner

Usage:
    uv run ent-cv sweep --data /mnt/data/ent_cv/datasets/dataset/data_with_val.yaml
    uv run ent-cv sweep --data ... --dry-run
"""

from dataclasses import dataclass, field
from datetime import datetime
import gc
from pathlib import Path
from typing import Any

from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table
import typer
import yaml

from ent_cv.config import MODELS_DIR, TRAIN_METRIC_KEYS
from ent_cv.utils import notify, send_email

app = typer.Typer(add_completion=False)
console = Console()

# ── Sweep configuration ─────────────────────────────────────────────────────

ARCHITECTURES: list[str] = [
    "yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26",
]

SCOUT_SIZES: list[str] = ["s", "m", "l"]

HPARAM_GRID: list[tuple[int, bool, float]] = [
    # (imgsz, rect, scale)
    (1024, True,  0.0),   # baseline (already run in Phase 1 — skipped)
    (640,  True,  0.0),   # smaller image size
    (1024, False, 0.0),   # rectangular training off
    (1024, True,  0.5),   # scale augmentation
]

SCOUT_BASELINE: tuple[int, bool, float] = (1024, True, 0.0)

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainConfig:
    """One training configuration — hashable for deduplication."""

    model: str
    imgsz: int
    rect: bool
    scale: float

    @property
    def label(self) -> str:
        r = "rect" if self.rect else "norect"
        return f"{self.model}_img{self.imgsz}_{r}_s{self.scale}"


@dataclass
class RunResult:
    """Outcome of a single training run."""

    config: TrainConfig
    epochs: int
    phase: str
    save_dir: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def map50(self) -> float:
        return self.metrics.get("mAP50", -1.0)


@dataclass
class SweepResult:
    """Full sweep outcome, passed to @notify results_fn."""

    phase1_results: list[RunResult]
    phase2_results: list[RunResult]
    final_result: RunResult
    total_runs: int
    failed_runs: int


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_one(
    cfg: TrainConfig,
    *,
    data: Path,
    epochs: int,
    device: str,
    project: Path,
    phase: str,
) -> RunResult:
    """Run a single training job. Never raises — failures are captured in RunResult."""
    from ent_cv.modeling.train import run as train_run

    logger.info(f"[{phase}] Starting: {cfg.label}  epochs={epochs}")
    try:
        result = train_run(
            data=data,
            model=cfg.model,
            epochs=epochs,
            batch=-1,
            imgsz=cfg.imgsz,
            rect=cfg.rect,
            scale=cfg.scale,
            device=device,
            project=project,
        )
        rd = getattr(result, "results_dict", {})
        save_dir = Path(str(getattr(result, "save_dir", "")))
        metrics: dict[str, float] = {}
        for label, key in TRAIN_METRIC_KEYS:
            v = rd.get(key)
            if v is not None:
                metrics[label] = float(v)
        logger.success(f"[{phase}] Done: {cfg.label}  mAP50={metrics.get('mAP50', 'N/A')}")
        return RunResult(
            config=cfg, epochs=epochs, phase=phase,
            save_dir=save_dir, metrics=metrics,
        )
    except Exception as exc:
        logger.error(f"[{phase}] FAILED: {cfg.label}  error={exc}")
        return RunResult(config=cfg, epochs=epochs, phase=phase, error=str(exc))
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return "---"


def _write_progress(
    sweep_dir: Path,
    *,
    phase: int,
    phase_name: str,
    run_idx: int,
    total_runs: int,
    current: str,
    results: list[RunResult],
    winners: list[str] | None = None,
) -> None:
    """Write progress.yaml to the sweep directory for external monitoring."""
    completed = []
    for r in results:
        entry: dict[str, Any] = {"model": r.config.label, "phase": r.phase}
        if r.error:
            entry["status"] = "FAIL"
            entry["error"] = r.error[:80]
        else:
            entry["status"] = "ok"
            entry["mAP50"] = round(r.map50, 4)
        completed.append(entry)

    progress = {
        "phase": phase,
        "phase_name": phase_name,
        "run": f"{run_idx}/{total_runs}",
        "current": current,
        "completed_runs": len(results),
        "failed_runs": sum(1 for r in results if r.error),
        "results": completed,
    }
    if winners is not None:
        progress["winners"] = winners

    progress_file = sweep_dir / "progress.yaml"
    with open(progress_file, "w") as f:
        yaml.dump(progress, f, default_flow_style=False, sort_keys=False)


def _send_phase_email(
    phase: int,
    phase_name: str,
    summary_text: str,
    winners: list[str],
) -> None:
    """Send a notification email after a phase completes."""
    lines = [
        f"Phase {phase} ({phase_name}) complete.",
        "",
        summary_text,
        "",
        f"Winners advancing: {', '.join(winners)}",
    ]
    send_email(
        subject=f"Sweep Phase {phase}: {phase_name} complete",
        body="\n".join(lines),
    )


LOCK_FILE = MODELS_DIR / "sweep.lock"
DEFAULT_DATA = Path("/mnt/data/ent_cv/datasets/dataset/data_with_val.yaml")


def _acquire_lock(sweep_dir: Path) -> None:
    """Write a lock file. Exits if a sweep is already running."""
    if LOCK_FILE.exists():
        active = LOCK_FILE.read_text().strip()
        logger.error(f"A sweep is already running ({active}). Use --status to check progress.")
        raise SystemExit(1)
    LOCK_FILE.write_text(str(sweep_dir))


def _release_lock() -> None:
    """Remove the lock file."""
    LOCK_FILE.unlink(missing_ok=True)


def _find_active_sweep() -> Path | None:
    """Find the active sweep directory via lock file, falling back to latest."""
    if LOCK_FILE.exists():
        sweep_dir = Path(LOCK_FILE.read_text().strip())
        if sweep_dir.is_dir() and (sweep_dir / "progress.yaml").exists():
            return sweep_dir
    # Fallback: find the most recent completed sweep with a progress file
    sweep_dirs = sorted(MODELS_DIR.glob("sweep_*"), reverse=True)
    for d in sweep_dirs:
        if d.is_dir() and (d / "progress.yaml").exists():
            return d
    return None


def _show_status() -> None:
    """Read progress.yaml from the latest sweep and render a Rich status display."""
    sweep_dir = _find_active_sweep()
    if sweep_dir is None:
        console.print("[red]No active sweep found.[/red]")
        console.print(f"  Looked in: {MODELS_DIR}/sweep_*/progress.yaml")
        return

    with open(sweep_dir / "progress.yaml") as f:
        progress = yaml.safe_load(f)

    phase = progress["phase"]
    phase_name = progress["phase_name"]
    run_str = progress["run"]
    current = progress["current"]
    completed = progress.get("completed_runs", 0)
    failed = progress.get("failed_runs", 0)
    winners = progress.get("winners")
    results = progress.get("results", [])

    # Header
    console.rule(f"[bold]Sweep Status — {sweep_dir.name}")
    console.print()
    if current == "done":
        console.print(f"  Phase {phase}/3: [bold]{phase_name}[/bold]  [green]COMPLETE[/green]")
    else:
        console.print(f"  Phase {phase}/3: [bold]{phase_name}[/bold]  Run {run_str}")
        console.print(f"  Training:  [yellow]{current}[/yellow]")
    console.print(f"  Completed: {completed}  Failed: {failed}")
    if winners:
        console.print(f"  Winners:   [green]{', '.join(winners)}[/green]")
    console.print()

    # Results table
    if results:
        table = Table(title="Results (ranked by mAP50)", box=box.ROUNDED,
                      header_style="bold cyan", show_lines=False)
        table.add_column("#", justify="right", width=3)
        table.add_column("Model", min_width=30)
        table.add_column("Phase", width=8)
        table.add_column("mAP50", style="green bold", justify="right", min_width=8)
        table.add_column("Status", width=8)

        sorted_results = sorted(results, key=lambda r: r.get("mAP50", -1), reverse=True)
        for rank, r in enumerate(sorted_results, 1):
            map50 = _fmt(r["mAP50"]) if "mAP50" in r else "---"
            status = "[red]FAIL[/red]" if r["status"] == "FAIL" else "[green]ok[/green]"
            table.add_row(str(rank), r["model"], r["phase"], map50, status)

        console.print(table)

    console.print()
    console.print(f"  [dim]Sweep dir: {sweep_dir}[/dim]")
    console.rule()


def _print_phase_table(
    title: str,
    results: list[RunResult],
    *,
    highlight_top: int = 0,
) -> str:
    """Print a Rich table to console and return plain-text version for email."""
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan", show_lines=True)
    table.add_column("#", justify="right", width=3, no_wrap=True)
    table.add_column("Model", min_width=12)
    table.add_column("imgsz", justify="right", width=6, no_wrap=True)
    table.add_column("rect", justify="center", width=5, no_wrap=True)
    table.add_column("scale", justify="right", width=6, no_wrap=True)
    table.add_column("mAP50", style="green bold", justify="right", min_width=8, no_wrap=True)
    table.add_column("mAP50-95", justify="right", min_width=9, no_wrap=True)
    table.add_column("Prec", justify="right", min_width=8, no_wrap=True)
    table.add_column("Recall", justify="right", min_width=8, no_wrap=True)
    table.add_column("Status", width=8, no_wrap=True)

    sorted_results = sorted(results, key=lambda r: r.map50, reverse=True)
    plain_lines = [title, "-" * len(title)]

    for rank, r in enumerate(sorted_results, 1):
        c = r.config
        status = "[red]FAIL[/red]" if r.error else "[green]OK[/green]"
        style = "bold yellow" if (rank <= highlight_top and not r.error) else ""
        table.add_row(
            str(rank), c.model, str(c.imgsz),
            "Y" if c.rect else "N", f"{c.scale:.1f}",
            _fmt(r.metrics.get("mAP50")), _fmt(r.metrics.get("mAP50-95")),
            _fmt(r.metrics.get("Precision")), _fmt(r.metrics.get("Recall")),
            status, style=style,
        )
        status_plain = "FAIL" if r.error else "OK"
        plain_lines.append(
            f"  {rank:>2}. {c.label:<40} mAP50={_fmt(r.metrics.get('mAP50'))}"
            f"  {status_plain}"
        )

    console.print(table)
    console.print()
    return "\n".join(plain_lines)


def _print_dry_run(
    *,
    architectures: list[str],
    scout_sizes: list[str],
    top_n: int,
    scout_epochs: int,
    sweep_epochs: int,
    final_epochs: int,
) -> None:
    """Print a summary of what would be trained without running anything."""
    scout_models = [f"{arch}{size}" for arch in architectures for size in scout_sizes]
    n_phase1 = len(scout_models)
    n_hparam_new = len(HPARAM_GRID) - 1  # minus baseline already done in Phase 1
    n_phase2 = top_n * n_hparam_new

    console.rule("[bold]Sweep Dry Run")

    console.print(f"\n[bold]Phase 1: Architecture Scout[/bold]  ({scout_epochs} epochs)")
    console.print(f"  Architectures: {', '.join(architectures)}")
    console.print(f"  Sizes:         {', '.join(scout_sizes)}")
    console.print("  Fixed:         imgsz=1024, rect=True, scale=0.0, batch=-1")
    console.print(f"  Runs:          {len(architectures)} arch x {len(scout_sizes)} sizes = {n_phase1}")
    console.print(f"  Select:        top {top_n} by mAP50\n")

    console.print(f"[bold]Phase 2: Hyperparameter Sweep[/bold]  ({sweep_epochs} epochs)")
    console.print("  Hparam grid (per winning model):")
    for imgsz, rect, scale in HPARAM_GRID:
        skip = " [dim](skip: done in Phase 1)[/dim]" if (imgsz, rect, scale) == SCOUT_BASELINE else ""
        console.print(f"    imgsz={imgsz}  rect={rect}  scale={scale}{skip}")
    console.print(f"  Runs:   {top_n} models x {n_hparam_new} new configs = {n_phase2}\n")

    console.print(f"[bold]Phase 3: Final Training[/bold]  ({final_epochs} epochs)")
    console.print("  Runs:   1 (best from Phase 2)\n")

    total = n_phase1 + n_phase2 + 1
    console.print(f"[bold]Total runs: {total}[/bold]")
    console.rule()


# ── Phases ───────────────────────────────────────────────────────────────────


def _phase1_scout(
    *,
    data: Path,
    epochs: int,
    device: str,
    architectures: list[str],
    scout_sizes: list[str],
    top_n: int,
    project: Path,
    sweep_dir: Path,
) -> tuple[list[RunResult], list[TrainConfig]]:
    """Phase 1: short training across architectures and sizes."""
    models = [f"{arch}{size}" for arch in architectures for size in scout_sizes]
    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 1: Architecture Scout  ({len(models)} models, {epochs} epochs)")
    logger.info(f"{'=' * 60}")

    configs = [
        TrainConfig(model=m, imgsz=SCOUT_BASELINE[0], rect=SCOUT_BASELINE[1], scale=SCOUT_BASELINE[2])
        for m in models
    ]
    results: list[RunResult] = []
    for i, cfg in enumerate(configs, 1):
        logger.info(f"Phase 1 [{i}/{len(configs)}]")
        _write_progress(sweep_dir, phase=1, phase_name="Architecture Scout",
                        run_idx=i, total_runs=len(configs), current=cfg.label, results=results)
        results.append(_run_one(cfg, data=data, epochs=epochs, device=device, project=project, phase="scout"))

    summary = _print_phase_table(f"Phase 1: Architecture Scout ({epochs} epochs)", results, highlight_top=top_n)

    successful = sorted([r for r in results if r.error is None], key=lambda r: r.map50, reverse=True)
    top_configs = [r.config for r in successful[:top_n]]

    if not top_configs:
        raise RuntimeError("Phase 1: All runs failed. Cannot proceed.")

    top_names = [c.model for c in top_configs]
    logger.info(f"Phase 1 winners (top {top_n}): {top_names}")
    _write_progress(sweep_dir, phase=1, phase_name="Architecture Scout",
                    run_idx=len(configs), total_runs=len(configs), current="done",
                    results=results, winners=top_names)
    _send_phase_email(1, "Architecture Scout", summary, top_names)
    return results, top_configs


def _phase2_hparam(
    *,
    data: Path,
    epochs: int,
    device: str,
    top_configs: list[TrainConfig],
    phase1_results: list[RunResult],
    project: Path,
    sweep_dir: Path,
) -> tuple[list[RunResult], RunResult]:
    """Phase 2: hparam variations on the top models from Phase 1."""
    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 2: Hyperparameter Sweep  ({len(top_configs)} models, {epochs} epochs)")
    logger.info(f"{'=' * 60}")

    phase1_set = {r.config for r in phase1_results if r.error is None}
    sweep_configs: list[TrainConfig] = []

    for base_cfg in top_configs:
        for imgsz, rect, scale in HPARAM_GRID:
            cfg = TrainConfig(model=base_cfg.model, imgsz=imgsz, rect=rect, scale=scale)
            if cfg not in phase1_set:
                sweep_configs.append(cfg)

    logger.info(f"Phase 2: {len(sweep_configs)} new configs")

    results: list[RunResult] = []
    for i, cfg in enumerate(sweep_configs, 1):
        logger.info(f"Phase 2 [{i}/{len(sweep_configs)}]")
        _write_progress(sweep_dir, phase=2, phase_name="Hyperparameter Sweep",
                        run_idx=i, total_runs=len(sweep_configs), current=cfg.label,
                        results=phase1_results + results)
        results.append(_run_one(cfg, data=data, epochs=epochs, device=device, project=project, phase="hparam"))

    # Combine Phase 1 winners + Phase 2 results for ranking
    all_candidates = [r for r in phase1_results if r.config in phase1_set] + results
    summary = _print_phase_table("Phase 2: Combined Rankings", all_candidates, highlight_top=1)

    successful = sorted([r for r in all_candidates if r.error is None], key=lambda r: r.map50, reverse=True)
    if not successful:
        raise RuntimeError("Phase 2: All runs failed. Cannot proceed.")

    best = successful[0]
    logger.info(f"Phase 2 winner: {best.config.label}  mAP50={best.map50:.4f}")
    _write_progress(sweep_dir, phase=2, phase_name="Hyperparameter Sweep",
                    run_idx=len(sweep_configs), total_runs=len(sweep_configs), current="done",
                    results=phase1_results + results, winners=[best.config.label])
    _send_phase_email(2, "Hyperparameter Sweep", summary, [best.config.label])
    return all_candidates, best


def _phase3_final(
    *,
    data: Path,
    epochs: int,
    device: str,
    best: RunResult,
    project: Path,
    sweep_dir: Path,
    all_prior: list[RunResult],
) -> RunResult:
    """Phase 3: full training on the winning config."""
    cfg = best.config
    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 3: Final Training  ({cfg.label}, {epochs} epochs)")
    logger.info(f"{'=' * 60}")

    _write_progress(sweep_dir, phase=3, phase_name="Final Training",
                    run_idx=1, total_runs=1, current=cfg.label, results=all_prior)

    result = _run_one(cfg, data=data, epochs=epochs, device=device, project=project, phase="final")

    if result.error:
        raise RuntimeError(f"Phase 3 failed: {result.error}")

    _print_phase_table(f"Phase 3: Final Model ({epochs} epochs)", [result])

    logger.success(
        f"SWEEP COMPLETE: {cfg.label}  mAP50={result.map50:.4f}  "
        f"Weights: {result.save_dir / 'weights' / 'best.pt'}"
    )
    _write_progress(sweep_dir, phase=3, phase_name="Final Training",
                    run_idx=1, total_runs=1, current="done",
                    results=all_prior + [result], winners=[cfg.label])
    return result


# ── Core entry point ─────────────────────────────────────────────────────────


def run(
    *,
    data: Path,
    device: str = "0",
    scout_epochs: int = 50,
    sweep_epochs: int = 100,
    final_epochs: int = 200,
    top_n: int = 3,
    architectures: list[str] | None = None,
    dry_run: bool = False,
) -> SweepResult | None:
    """Execute the full 3-phase sweep."""
    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    archs = architectures or ARCHITECTURES

    if dry_run:
        _print_dry_run(
            architectures=archs,
            scout_sizes=SCOUT_SIZES,
            top_n=top_n,
            scout_epochs=scout_epochs,
            sweep_epochs=sweep_epochs,
            final_epochs=final_epochs,
        )
        return None

    # Create timestamped sweep directory and acquire lock
    sweep_dir = MODELS_DIR / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    _acquire_lock(sweep_dir)
    logger.info(f"Sweep directory: {sweep_dir}")

    try:
        # Phase 1
        p1_results, top_configs = _phase1_scout(
            data=data, epochs=scout_epochs, device=device,
            architectures=archs, scout_sizes=SCOUT_SIZES, top_n=top_n,
            project=sweep_dir / "phase1_scout", sweep_dir=sweep_dir,
        )

        # Phase 2
        p2_results, best = _phase2_hparam(
            data=data, epochs=sweep_epochs, device=device,
            top_configs=top_configs, phase1_results=p1_results,
            project=sweep_dir / "phase2_hparam", sweep_dir=sweep_dir,
        )

        # Phase 3
        p2_new = [r for r in p2_results if r.phase != "scout"]
        final = _phase3_final(
            data=data, epochs=final_epochs, device=device,
            best=best, project=sweep_dir / "phase3_final",
            sweep_dir=sweep_dir, all_prior=p1_results + p2_new,
        )

        all_results = p1_results + [r for r in p2_results if r.phase != "scout"] + [final]
        failed = sum(1 for r in all_results if r.error is not None)

        return SweepResult(
            phase1_results=p1_results,
            phase2_results=p2_results,
            final_result=final,
            total_runs=len(all_results),
            failed_runs=failed,
        )
    finally:
        _release_lock()


# ── CLI ──────────────────────────────────────────────────────────────────────


def _results_fn(result: SweepResult | None) -> tuple[str, list[Path]]:
    if result is None:
        return "", []

    f = result.final_result
    lines = [
        f"Sweep complete: {result.total_runs} runs ({result.failed_runs} failed)",
        "",
        "Final model:",
        f"  Config:  {f.config.label}",
        f"  Epochs:  {f.epochs}",
    ]
    for label, _ in TRAIN_METRIC_KEYS:
        v = f.metrics.get(label)
        if v is not None:
            lines.append(f"  {label:<12} {v:.4f}")
    if f.save_dir and f.save_dir.exists():
        lines.append(f"\n  Weights: {f.save_dir / 'weights' / 'best.pt'}")

    attachments: list[Path] = []
    if f.save_dir and f.save_dir.exists():
        for fname in ("results.png", "confusion_matrix_normalized.png", "args.yaml"):
            p = f.save_dir / fname
            if p.exists():
                attachments.append(p)

    return "\n".join(lines), attachments


@app.command()
@notify("Sweep", results_fn=_results_fn)
def main(
    data: Path = typer.Option(DEFAULT_DATA, help="Path to dataset YAML (must include val split)"),
    device: str = typer.Option("0", help="GPU device (e.g. '0', '0,1', 'cpu')"),
    scout_epochs: int = typer.Option(50, help="Epochs for Phase 1 architecture scout"),
    sweep_epochs: int = typer.Option(100, help="Epochs for Phase 2 hparam sweep"),
    final_epochs: int = typer.Option(200, help="Epochs for Phase 3 final training"),
    top_n: int = typer.Option(3, help="Number of models to advance from Phase 1"),
    dry_run: bool = typer.Option(False, help="Print plan without training"),
    status: bool = typer.Option(False, help="Show progress of the latest running sweep"),
) -> SweepResult | None:
    """Fully automated 3-phase training sweep: scout -> hparam search -> final."""
    if status:
        _show_status()
        return None

    return run(
        data=data, device=device,
        scout_epochs=scout_epochs, sweep_epochs=sweep_epochs,
        final_epochs=final_epochs, top_n=top_n, dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
