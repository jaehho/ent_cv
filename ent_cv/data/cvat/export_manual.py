"""Export manual annotations from a CVAT job as YOLO label files.

When a task has both auto-imported predictions and hand-drawn corrections,
this script filters by the ``source`` field on each shape so you can extract
only the manual work.

If the CVAT task uses labels that are not in ``config.LABELS``, the script
prompts interactively for each unknown label and either aliases it to an
existing canonical label, appends it as a new canonical label (patching
``ent_cv/config.py``), or drops its annotations from this export. In
non-interactive environments the script refuses to guess and exits with a
non-zero status.

Pass ``--dry-run`` to preview what would happen without touching the
filesystem or ``config.py``: it connects to CVAT, diffs labels against
``config.LABELS``, counts how many rectangles would be exported, and exits
non-zero if drift is detected so the flag is usable as a pre-flight check.

Usage:
    uv run ent-cv cvat export-manual 21              # job 21, manual only
    uv run ent-cv cvat export-manual 21 --source all # everything
    uv run ent-cv cvat export-manual 21 --dry-run    # pre-flight, no writes
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys
from typing import Optional

from cvat_sdk import make_client
from cvat_sdk.api_client.api.labels_api import LabelsApi
from cvat_sdk.api_client.api.tasks_api import TasksApi
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer
import yaml

from ent_cv.config import DATASETS_DIR, LABELS

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)

_CONFIG_PY = Path(__file__).resolve().parents[2] / "config.py"

# Sentinel meaning "do not filter by source". Any other value is matched
# literally against shape.source after validating it exists on the job.
_SOURCE_ALL = "all"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_all_cvat_labels(labels_api: LabelsApi, task_id: int) -> list:
    """Fetch every label for a task, handling pagination."""
    results: list = []
    page = 1
    while True:
        (paginated, _) = labels_api.list(task_id=task_id, page=page)
        results.extend(paginated.results)
        if getattr(paginated, "next", None) is None:
            return results
        page += 1


def _append_label_to_config(new_label: str) -> int:
    """Append ``new_label`` to the LABELS list in ``ent_cv/config.py``.

    Returns the new label's index. Preserves the file's formatting by inserting
    a new entry immediately before the closing ``]`` of the LABELS list. Raises
    RuntimeError if the expected format cannot be found.
    """
    text = _CONFIG_PY.read_text()

    pattern = re.compile(r"^LABELS\s*=\s*\[\n(.*?)^\]$", re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(
            f"Could not locate the LABELS list in {_CONFIG_PY}. "
            "Append the label manually and re-run."
        )

    body = match.group(1)
    existing_indices = re.findall(r'^ *"([^"]*)"', body, re.MULTILINE)
    if new_label in existing_indices:
        return existing_indices.index(new_label)

    new_index = len(existing_indices)
    body_rstripped = body.rstrip()
    if not body_rstripped.endswith(","):
        body_rstripped += ","
    new_body = body_rstripped + "\n" + f'    "{new_label}",\n'
    new_text = text[: match.start(1)] + new_body + text[match.end(1) :]
    _CONFIG_PY.write_text(new_text)
    logger.info(f"Appended '{new_label}' to config.LABELS at index {new_index}")
    return new_index


def _resolve_unknown_labels(
    unknown_names: list[str],
    canonical: list[str],
) -> dict[str, Optional[int]]:
    """Interactively map each unknown CVAT label to a canonical YOLO class id.

    For each name in ``unknown_names`` (not already in ``canonical``), prompt
    the user to alias it, append it, or drop it. Updates ``canonical`` in place
    when a label is appended. Returns a dict ``{unknown_name: class_id | None}``
    where ``None`` means "drop annotations with this label".

    Fails hard if stdin is not a TTY.
    """
    resolutions: dict[str, Optional[int]] = {}
    if not unknown_names:
        return resolutions

    logger.warning(
        f"Found {len(unknown_names)} label(s) used in this export that are not "
        f"in config.LABELS: {unknown_names}"
    )

    if not sys.stdin.isatty():
        logger.error(
            "Running non-interactively — refusing to guess. Append these labels "
            "to ent_cv/config.py:LABELS and re-run:\n  " + "\n  ".join(unknown_names)
        )
        raise typer.Exit(2)

    for unknown in unknown_names:
        typer.echo()
        typer.secho(
            f"Label {unknown!r} is used in CVAT but not in config.LABELS.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        typer.echo("How should this label be handled?")
        for i, name in enumerate(canonical, 1):
            typer.echo(f"  {i:2d}. Alias to {name!r}")
        append_idx = len(canonical) + 1
        drop_idx = len(canonical) + 2
        typer.echo(
            f"  {append_idx:2d}. Append {unknown!r} as a new canonical label "
            "(patches config.py)"
        )
        typer.echo(
            f"  {drop_idx:2d}. Drop — exclude all of its annotations from this export"
        )

        while True:
            raw = typer.prompt(f"Choice [1-{drop_idx}]")
            try:
                choice = int(raw)
            except ValueError:
                typer.echo("Please enter a number.")
                continue

            if 1 <= choice <= len(canonical):
                target = canonical[choice - 1]
                resolutions[unknown] = choice - 1
                typer.secho(
                    f"→ {unknown!r} aliased to {target!r} (class {choice - 1})",
                    fg=typer.colors.GREEN,
                )
                break
            if choice == append_idx:
                new_index = _append_label_to_config(unknown)
                # Keep the in-memory canonical list in sync with the patched file.
                while len(canonical) < new_index:
                    canonical.append(f"<placeholder_{len(canonical)}>")
                if len(canonical) == new_index:
                    canonical.append(unknown)
                else:
                    canonical[new_index] = unknown
                resolutions[unknown] = new_index
                typer.secho(
                    f"→ {unknown!r} appended to config.LABELS at index {new_index}. "
                    "Remember to commit ent_cv/config.py.",
                    fg=typer.colors.GREEN,
                )
                break
            if choice == drop_idx:
                resolutions[unknown] = None
                typer.secho(
                    f"→ {unknown!r} will be dropped from this export",
                    fg=typer.colors.YELLOW,
                )
                break
            typer.echo(f"Invalid choice — enter 1 to {drop_idx}.")

    return resolutions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    job_id: int = typer.Argument(..., help="CVAT job ID to export from."),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory for labels. Defaults to exports/<task_name>_<source>.",
    ),
    source: str = typer.Option(
        "manual",
        help=(
            "Annotation source filter. Pass 'all' to export every shape, or "
            "any source string that actually exists on the job (commonly "
            "'manual' for hand-drawn, 'auto' for auto-annotated, or 'file' "
            "for shapes uploaded by upload-predictions / import-dataset). "
            "Use --dry-run to see which sources the job has. Typos fail fast."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Connect to CVAT and report what would be exported without writing "
            "any files or patching config.py. Exits non-zero if CVAT has labels "
            "that are not in config.LABELS."
        ),
    ),
) -> None:
    """Export annotations from a CVAT job, filtering by source."""
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        job = client.jobs.retrieve(job_id)
        task = client.tasks.retrieve(job.task_id)
        logger.info(f"Job {job_id} belongs to task '{task.name}' (ID {task.id})")

        # 1. Fetch all CVAT labels defined on this task (paginated).
        labels_api = LabelsApi(client.api_client)
        cvat_labels = _fetch_all_cvat_labels(labels_api, task.id)
        cvat_label_by_id: dict[int, str] = {lbl.id: lbl.name for lbl in cvat_labels}

        # 2. Frame dimensions from the task's data meta.
        tasks_api = TasksApi(client.api_client)
        (meta, _) = tasks_api.retrieve_data_meta(task.id)
        if not meta.frames:
            logger.error("Task has no frames — nothing to export.")
            raise typer.Exit(1)
        width = meta.frames[0].width
        height = meta.frames[0].height
        logger.info(f"Frame size: {width}x{height}")

        # 3. Fetch annotations and show a source breakdown.
        annotations = job.get_annotations()
        all_shapes = annotations.shapes

        source_counts: dict[str, int] = {}
        for s in all_shapes:
            source_counts[s.source] = source_counts.get(s.source, 0) + 1
        logger.info(f"Total shapes: {len(all_shapes)}")
        for src, count in sorted(source_counts.items()):
            marker = (
                " ← selected" if (source != _SOURCE_ALL and src == source) else ""
            )
            logger.info(f"  source='{src}': {count}{marker}")

        # 4. Validate the requested source against what this job actually has.
        #    'all' is always valid; any literal source must match one of the
        #    keys in source_counts. This replaces a fixed enum so CVAT can
        #    introduce new source values without requiring a code change.
        if all_shapes and source != _SOURCE_ALL and source not in source_counts:
            available = sorted(source_counts.keys()) + [_SOURCE_ALL]
            logger.error(
                f"No shapes in this job have source={source!r}. "
                f"Available sources: {available}"
            )
            raise typer.Exit(1)

        # 5. Apply the source filter.
        if source == _SOURCE_ALL:
            shapes = list(all_shapes)
        else:
            shapes = [s for s in all_shapes if s.source == source]

        if not shapes:
            logger.warning("No annotations match the filter — nothing to export")
            raise typer.Exit(0)

        # 5. Resolve labels — prompt interactively for any unknown-but-used name.
        #    Non-rectangle shapes are skipped entirely, so they do not influence
        #    which labels count as "used".
        used_label_names: set[str] = set()
        for shape in shapes:
            if shape.type.value != "rectangle":
                continue
            name = cvat_label_by_id.get(shape.label_id)
            if name is not None:
                used_label_names.add(name)

        canonical = list(LABELS)
        unknown_used = sorted(used_label_names - set(canonical))

        if dry_run:
            logger.info("=== DRY RUN — no files written, config.py untouched ===")

            cvat_names = set(cvat_label_by_id.values())
            canonical_set = set(canonical)
            unused_canonical = sorted(canonical_set - cvat_names)

            logger.info(
                f"CVAT labels defined on task: {len(cvat_names)} | "
                f"config.LABELS: {len(canonical)}"
            )
            if unknown_used:
                logger.warning(
                    f"Used in CVAT but missing from config.LABELS "
                    f"({len(unknown_used)}): {unknown_used}"
                )
            if unused_canonical:
                logger.info(
                    f"In config.LABELS but absent from CVAT "
                    f"({len(unused_canonical)}): {unused_canonical}"
                )

            # Project the shapes through the same rectangle/label filter used
            # by the real export so the counts match what would actually land.
            exportable = 0
            pending = 0
            non_rect = 0
            for shape in shapes:
                if shape.type.value != "rectangle":
                    non_rect += 1
                    continue
                name = cvat_label_by_id.get(shape.label_id)
                if name in canonical_set:
                    exportable += 1
                else:
                    pending += 1

            logger.info(
                f"After filter (source={source}): {len(shapes)} shape(s)"
            )
            logger.info(f"  would export now:         {exportable} rectangle(s)")
            if pending:
                logger.warning(
                    f"  pending label resolution: {pending} rectangle(s)"
                )
            if non_rect:
                logger.info(f"  non-rectangles (skipped): {non_rect}")

            target = output_dir
            if target is None:
                suffix = f"_{source}" if source != _SOURCE_ALL else ""
                target = DATASETS_DIR / "exports" / f"{task.name}{suffix}"
            logger.info(f"Would write to: {target}")

            if unknown_used:
                logger.warning(
                    f"Drift detected — re-run without --dry-run to resolve "
                    f"{len(unknown_used)} unknown label(s) interactively."
                )
                raise typer.Exit(1)

            logger.success("Dry run OK — safe to re-run without --dry-run.")
            return

        resolutions = _resolve_unknown_labels(unknown_used, canonical)

        # Build label_map: CVAT label_id -> (canonical_name, yolo_class_id).
        # Labels resolved to None (dropped) are omitted.
        label_map: dict[int, tuple[str, int]] = {}
        for lbl in cvat_labels:
            if lbl.name in canonical:
                label_map[lbl.id] = (lbl.name, canonical.index(lbl.name))
                continue
            if lbl.name in resolutions:
                resolved = resolutions[lbl.name]
                if resolved is None:
                    continue
                label_map[lbl.id] = (canonical[resolved], resolved)
            # Labels defined in CVAT but not used in this export are fine to
            # leave out of label_map; they won't be referenced.

        # 6. Group rectangles by frame; count what we skip and why.
        frames: dict[int, list] = {}
        skipped_types: dict[str, int] = {}
        skipped_dropped = 0
        for shape in shapes:
            if shape.type.value != "rectangle":
                skipped_types[shape.type.value] = (
                    skipped_types.get(shape.type.value, 0) + 1
                )
                continue
            if shape.label_id not in label_map:
                skipped_dropped += 1
                continue
            frames.setdefault(shape.frame, []).append(shape)

        if skipped_types:
            logger.info(f"Skipped non-rectangle shapes: {skipped_types}")
        if skipped_dropped:
            logger.info(
                f"Skipped {skipped_dropped} shapes belonging to dropped labels"
            )

        if not frames:
            logger.warning("No rectangles left after filtering — nothing to export")
            raise typer.Exit(0)

        # 7. Prepare output directory.
        if output_dir is None:
            suffix = f"_{source}" if source != _SOURCE_ALL else ""
            output_dir = DATASETS_DIR / "exports" / f"{task.name}{suffix}"

        if output_dir.exists() and any(output_dir.iterdir()):
            typer.confirm(f"'{output_dir}' is not empty. Overwrite?", abort=True)
            shutil.rmtree(output_dir)

        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 8. Write YOLO-format label files, one per frame that has rectangles.
        class_counts: dict[str, int] = {}
        total_boxes = 0
        for frame_num in sorted(frames):
            lines: list[str] = []
            for shape in frames[frame_num]:
                label_name, class_id = label_map[shape.label_id]
                x1, y1, x2, y2 = shape.points

                x_center = max(0.0, min(1.0, (x1 + x2) / 2.0 / width))
                y_center = max(0.0, min(1.0, (y1 + y2) / 2.0 / height))
                bw = max(0.0, min(1.0, (x2 - x1) / width))
                bh = max(0.0, min(1.0, (y2 - y1) / height))

                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                )
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
                total_boxes += 1

            label_file = labels_dir / f"frame_{frame_num:06d}.txt"
            label_file.write_text("\n".join(lines) + "\n")

        # 9. Write data.yaml so downstream tools (combine.py, YOLO) see an
        #    explicit class-id → name mapping that matches the label files.
        data_yaml = {
            "path": ".",
            "names": {i: name for i, name in enumerate(canonical)},
        }
        (output_dir / "data.yaml").write_text(
            yaml.dump(data_yaml, default_flow_style=False, allow_unicode=True)
        )

        # 10. Summary.
        logger.info(f"Exported {total_boxes} boxes across {len(frames)} frames:")
        for name in sorted(class_counts):
            logger.info(f"  {name}: {class_counts[name]}")
        logger.info(f"Frame range: {min(frames)}–{max(frames)}")

        logger.success(f"Output → {output_dir}")


if __name__ == "__main__":
    app()
