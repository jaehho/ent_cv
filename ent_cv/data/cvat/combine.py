"""Combine two or more YOLO datasets into a single output directory.

Usage:
    uv run ent_cv/data/cvat/combine.py /path/to/ds1 /path/to/ds2 --output-dir /path/to/combined

If class names differ between datasets, use --label-aliases to normalise them:
    --label-aliases '{"Energy-based hemostasis": "Energy-based hemostasis without suction"}'

The output directory has the standard Ultralytics YOLO layout:
    <output>/
        data.yaml
        train.txt
        images/train/<stem>.<ext>
        labels/train/<stem>.txt
"""
import json
import shutil
from pathlib import Path
from typing import Annotated

from loguru import logger
from rich.console import Console
from rich.table import Table
import typer
import yaml

from ent_cv.config import DATASETS_DIR, LABELS

_console = Console()

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset_names(dataset_dir: Path) -> dict[int, str]:
    """Return {class_id: class_name} from data.yaml, or {} if file is missing."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        logger.warning(f"No data.yaml in {dataset_dir} – class names unknown")
        return {}
    with data_yaml.open() as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _apply_aliases(names: dict[int, str], aliases: dict[str, str]) -> dict[int, str]:
    """Rename any class whose current name is a key in *aliases*."""
    return {k: aliases.get(v, v) for k, v in names.items()}


def _build_unified_labels(per_dataset_names: list[dict[int, str]]) -> list[str]:
    """Build an ordered unified label list.

    Starts with the canonical LABELS from config (preserving index order), then
    appends any extra names found in the datasets that are not already covered.
    """
    unified: list[str] = list(LABELS)
    seen = set(unified)
    for names in per_dataset_names:
        for name in names.values():
            if name not in seen:
                logger.warning(f"Label '{name}' not in canonical LABELS – appending at end")
                unified.append(name)
                seen.add(name)
    return unified


def _build_id_remap(dataset_names: dict[int, str], unified_labels: list[str]) -> dict[int, int]:
    """Return {old_class_id: unified_class_id} for one dataset."""
    name_to_unified = {name: i for i, name in enumerate(unified_labels)}
    remap: dict[int, int] = {}
    for old_id, name in dataset_names.items():
        if name not in name_to_unified:
            raise ValueError(
                f"Class '{name}' (id={old_id}) is not in the unified label list. "
                "Use --label-aliases to map it to a canonical name."
            )
        remap[old_id] = name_to_unified[name]
    return remap


def _interactive_alias_resolution(
    per_dataset_names: list[dict[int, str]],
    existing_aliases: dict[str, str],
) -> dict[str, str]:
    """Prompt the user to resolve every non-canonical class name.

    Returns a new alias dict that supplements *existing_aliases*.
    """
    aliases = dict(existing_aliases)

    # Collect unique names that still aren't canonical after existing aliases
    all_names: set[str] = set()
    for names in per_dataset_names:
        for name in names.values():
            resolved = aliases.get(name, name)
            all_names.add(resolved)

    non_canonical = sorted(all_names - set(LABELS))
    if not non_canonical:
        _console.print("[green]All class names already match canonical labels.[/green]")
        return aliases

    canonical_list = list(LABELS)

    _console.print(
        f"\n[bold yellow]{len(non_canonical)} non-canonical class name(s) need resolution:[/bold yellow]"
    )

    for name in non_canonical:
        _console.print(f"\n  [bold red]{name!r}[/bold red] is not in the canonical label set.")
        _console.print("  Map to:")
        for i, c in enumerate(canonical_list, 1):
            _console.print(f"    [cyan]{i:2d}[/cyan]. {c}")
        keep_idx = len(canonical_list) + 1
        _console.print(f"    [cyan]{keep_idx:2d}[/cyan]. Keep as new label")

        while True:
            raw = typer.prompt(f"  Choice [1-{keep_idx}]")
            try:
                choice = int(raw)
            except ValueError:
                _console.print("  [red]Please enter a number.[/red]")
                continue

            if 1 <= choice <= len(canonical_list):
                target = canonical_list[choice - 1]
                # Record alias from original (pre-existing-alias) name too
                for orig, resolved in [(n, aliases.get(n, n)) for names in per_dataset_names for n in names.values()]:
                    if resolved == name:
                        aliases[orig] = target
                aliases[name] = target
                _console.print(f"  [green]→ '{name}' will be merged into '{target}'[/green]")
                break
            elif choice == keep_idx:
                _console.print(f"  [yellow]→ '{name}' will be added as a new label[/yellow]")
                break
            else:
                _console.print(f"  [red]Invalid choice – enter 1 to {keep_idx}.[/red]")

    return aliases


def _show_remap_summary(
    dataset_dirs: list[Path],
    per_dataset_names: list[dict[int, str]],
    id_remaps: list[dict[int, int]],
    unified_labels: list[str],
) -> bool:
    """Print a unified-label table and per-dataset remap tables, then confirm.

    Returns True if the user confirms, False otherwise.
    """
    # Unified label table
    ul_table = Table(title="Unified label set", show_lines=True)
    ul_table.add_column("ID", style="cyan", justify="right")
    ul_table.add_column("Name")
    ul_table.add_column("Canonical", justify="center")
    for i, name in enumerate(unified_labels):
        canonical = "[green]✓[/green]" if name in LABELS else "[yellow]new[/yellow]"
        ul_table.add_row(str(i), name, canonical)
    _console.print(ul_table)

    # Per-dataset remap tables (only when remapping actually happens)
    for ds_dir, names, remap in zip(dataset_dirs, per_dataset_names, id_remaps):
        changed = {old: new for old, new in remap.items() if old != new}
        if not changed:
            continue
        t = Table(title=f"ID remapping – {ds_dir.name}", show_lines=True)
        t.add_column("Old ID", style="cyan", justify="right")
        t.add_column("Old name")
        t.add_column("New ID", style="cyan", justify="right")
        t.add_column("New name")
        for old_id, new_id in sorted(changed.items()):
            t.add_row(str(old_id), names[old_id], str(new_id), unified_labels[new_id])
        _console.print(t)

    return typer.confirm("\nProceed with these mappings?", default=True)


def _remap_label_file(content: str, remap: dict[int, int]) -> str:
    """Rewrite a YOLO .txt annotation file with remapped class IDs."""
    out_lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            out_lines.append("")
            continue
        parts = line.split()
        new_id = remap.get(int(parts[0]), int(parts[0]))
        out_lines.append(" ".join([str(new_id)] + parts[1:]))
    return "\n".join(out_lines) + ("\n" if content.endswith("\n") else "")


def _resolve_collision(
    out_name: str,
    existing_ds: Path,
    new_ds: Path,
    policy: str | None,
) -> tuple[str, str | None]:
    """Prompt the user how to resolve a file name collision.

    Returns (resolution, new_policy) where resolution is 'keep_existing' or
    'keep_new', and new_policy is the updated always-apply policy (or None to
    continue asking per-collision).

    For 3+ datasets that share the same filename, each new occurrence is
    compared against whichever version is currently on disk.  The 'apply-to-all'
    options therefore mean:
      keep_existing → always keep the first dataset that wrote this name
      keep_new      → always overwrite with the latest dataset encountered
    """
    if policy is not None:
        return policy, policy

    _console.print(
        f"\n[bold yellow]Collision:[/bold yellow] [cyan]{out_name}[/cyan] "
        "exists in multiple datasets."
    )
    _console.print(f"  [1] Keep [bold]{existing_ds.name}[/bold]  [dim](already on disk)[/dim]")
    _console.print(f"  [2] Use  [bold]{new_ds.name}[/bold]  [dim](overwrite)[/dim]")
    _console.print( "  [3] Always keep whichever dataset was processed first  "
                    "[dim](skip all future duplicates)[/dim]")
    _console.print( "  [4] Always overwrite with the later dataset  "
                    "[dim](last one wins)[/dim]")

    choice_map: dict[str, tuple[str, str | None]] = {
        "1": ("keep_existing", None),
        "2": ("keep_new", None),
        "3": ("keep_existing", "keep_existing"),
        "4": ("keep_new", "keep_new"),
    }
    while True:
        raw = typer.prompt("  Choice [1-4]", default="1")
        if raw in choice_map:
            resolution, new_policy = choice_map[raw]
            if new_policy is not None:
                label = (
                    "first-dataset-wins (skip duplicates)"
                    if new_policy == "keep_existing"
                    else "last-dataset-wins (always overwrite)"
                )
                _console.print(f"  [green]→ '{label}' will be applied to all remaining collisions[/green]")
            return resolution, new_policy
        _console.print("  [red]Please enter 1, 2, 3, or 4.[/red]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    dataset_dirs: Annotated[
        list[Path],
        typer.Argument(help="Two or more dataset directories to combine."),
    ] = [DATASETS_DIR / "exports" / "batch1", DATASETS_DIR / "exports" / "batch2"],
    output_dir: Path = typer.Option(
        DATASETS_DIR / "dataset",
        "--output-dir", "-o",
        help="Destination directory for the combined dataset.",
    ),
    label_aliases: str = typer.Option(
        default="{}",
        help=(
            'JSON mapping non-canonical class names → canonical names, applied to every '
            'dataset before merging. '
            'Example: \'{"Energy-based hemostasis": "Energy-based hemostasis without suction"}\''
        ),
    ),
):
    if len(dataset_dirs) < 2:
        logger.error("Provide at least two dataset directories to combine.")
        raise typer.Exit(1)

    aliases: dict[str, str] = json.loads(label_aliases)
    if aliases:
        logger.info(f"Label aliases: {aliases}")

    # --------------------------------------------------------- resolve labels
    per_dataset_names: list[dict[int, str]] = []
    for ds_dir in dataset_dirs:
        names = _apply_aliases(_load_dataset_names(ds_dir), aliases)
        per_dataset_names.append(names)
        logger.info(
            f"  {ds_dir.name}: {len(names)} classes – "
            + ", ".join(f"{i}:{n}" for i, n in sorted(names.items()))
        )

    # --------------------------------------------------------- interactive alias resolution
    aliases = _interactive_alias_resolution(per_dataset_names, aliases)
    # Re-apply the (now-expanded) alias map
    per_dataset_names = [
        _apply_aliases(_load_dataset_names(ds_dir), aliases)
        for ds_dir in dataset_dirs
    ]

    unified_labels = _build_unified_labels(per_dataset_names)
    logger.info(
        f"Unified label set ({len(unified_labels)}): "
        + ", ".join(f"{i}:{n}" for i, n in enumerate(unified_labels))
    )

    id_remaps: list[dict[int, int]] = []
    for ds_dir, names in zip(dataset_dirs, per_dataset_names):
        remap = _build_id_remap(names, unified_labels)
        changed = {k: v for k, v in remap.items() if k != v}
        if changed:
            logger.info(f"  {ds_dir.name}: remapping class IDs {changed}")
        id_remaps.append(remap)

    # --------------------------------------------------------- interactive confirmation
    confirmed = _show_remap_summary(dataset_dirs, per_dataset_names, id_remaps, unified_labels)
    if not confirmed:
        _console.print("[red]Aborted.[/red]")
        raise typer.Exit(0)

    # --------------------------------------------------------- prepare output dir
    if output_dir.exists() and any(output_dir.iterdir()):
        typer.confirm(f"'{output_dir}' is not empty. Overwrite?", abort=True)
        shutil.rmtree(output_dir)

    images_out = output_dir / "images" / "train"
    labels_out = output_dir / "labels" / "train"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    # --------------------------------------------------------- copy files
    train_entries: list[str] = []
    total_collisions = 0
    collision_policy: str | None = None          # None = ask each time
    written_files: dict[str, int] = {}           # out_name -> ds_idx that wrote it

    for ds_idx, (ds_dir, remap) in enumerate(zip(dataset_dirs, id_remaps)):
        images_root = ds_dir / "images"
        labels_root = ds_dir / "labels"

        image_files = sorted(images_root.rglob("*.jpg")) + sorted(images_root.rglob("*.png"))
        if not image_files:
            logger.error(f"No images found under {images_root}")
            raise typer.Exit(1)

        label_stems = {lf.stem: lf for lf in sorted(labels_root.rglob("*.txt"))}

        for img in image_files:
            out_stem = img.stem
            out_name = out_stem + img.suffix
            dest_img = images_out / out_name

            if dest_img.exists():
                total_collisions += 1
                existing_ds_idx = written_files.get(out_name)
                existing_ds = dataset_dirs[existing_ds_idx] if existing_ds_idx is not None else ds_dir
                resolution, collision_policy = _resolve_collision(
                    out_name, existing_ds, ds_dir, collision_policy
                )
                if resolution == "keep_new":
                    shutil.copy2(img, dest_img)
                    written_files[out_name] = ds_idx
                    lf = label_stems.get(img.stem)
                    dest_lbl = labels_out / f"{out_stem}.txt"
                    if lf is not None:
                        dest_lbl.write_text(_remap_label_file(lf.read_text(), remap))
                    else:
                        dest_lbl.write_text("")
                continue

            shutil.copy2(img, dest_img)
            train_entries.append(f"images/train/{out_name}")
            written_files[out_name] = ds_idx

            # Matching label file (may not exist for unannotated frames)
            lf = label_stems.get(img.stem)
            dest_lbl = labels_out / f"{out_stem}.txt"
            if lf is not None:
                dest_lbl.write_text(_remap_label_file(lf.read_text(), remap))
            else:
                dest_lbl.write_text("")  # empty = no annotations

        logger.info(f"  {ds_dir.name}: copied {len(image_files)} images")

    if total_collisions:
        logger.warning(f"{total_collisions} collision(s) encountered")

    # --------------------------------------------------------- write data.yaml + train.txt
    data_yaml = {
        "path": ".",
        "train": "train.txt",
        "names": {i: name for i, name in enumerate(unified_labels)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.dump(data_yaml, default_flow_style=False, allow_unicode=True)
    )
    (output_dir / "train.txt").write_text("\n".join(train_entries) + "\n")

    total_images = len(train_entries)
    logger.success(
        f"Done! Combined dataset ({total_images} images, {len(unified_labels)} classes) "
        f"written to: {output_dir}"
    )


if __name__ == "__main__":
    app()
