"""Create a CVAT project with canonical labels and migrate existing tasks into it.

After running this once, all task-creation commands (``upload``,
``upload-predictions``, ``import``) will use the project for label
inheritance — labels are defined once at the project level and every task
in the project inherits them automatically.

Usage:
    uv run ent-cv cvat setup-project                   # create project + migrate
    uv run ent-cv cvat setup-project --project-id 5    # migrate tasks into existing project
"""

from __future__ import annotations

from cvat_sdk import make_client
from cvat_sdk.api_client.api.labels_api import LabelsApi
from cvat_sdk.api_client.api.projects_api import ProjectsApi
from cvat_sdk.api_client.api.tasks_api import TasksApi
from cvat_sdk.api_client.model.patched_label_request import PatchedLabelRequest
from cvat_sdk.api_client.model.patched_task_write_request import PatchedTaskWriteRequest
from cvat_sdk.api_client.model.project_write_request import ProjectWriteRequest
from dotenv import find_dotenv, load_dotenv
from loguru import logger
import typer

from ent_cv.config import LABELS
from ent_cv.data.cvat.export_manual import _fetch_all_cvat_labels

load_dotenv(find_dotenv())

app = typer.Typer(add_completion=False)


def _fetch_all_tasks(client) -> list:
    """Fetch every task, handling pagination."""
    results: list = []
    page = 1
    while True:
        (paginated, _) = TasksApi(client.api_client).list(page=page)
        results.extend(paginated.results)
        if getattr(paginated, "next", None) is None:
            return results
        page += 1


def _get_project_label_names(client, project_id: int) -> set[str]:
    """Return the set of label names defined on a project."""
    project = client.projects.retrieve(project_id)
    return {lbl.name for lbl in project.get_labels()}


def _add_labels_to_project(
    projects_api: ProjectsApi, project_id: int, names: list[str]
) -> None:
    """Append missing label names to an existing project."""
    from cvat_sdk.api_client.model.patched_project_write_request import (
        PatchedProjectWriteRequest,
    )

    labels = [PatchedLabelRequest(name=n) for n in names]
    projects_api.partial_update(
        project_id,
        patched_project_write_request=PatchedProjectWriteRequest(labels=labels),
    )


def _move_task(
    tasks_api: TasksApi,
    labels_api: LabelsApi,
    task,
    project_id: int,
    project_label_names: set[str],
    projects_api: ProjectsApi,
) -> bool:
    """Move a single task into the project.

    If the task has labels missing from the project, prompts the user to add
    them. Returns True on success, False on skip.
    """
    task_labels = _fetch_all_cvat_labels(labels_api, task.id)
    task_label_names = {lbl.name for lbl in task_labels}
    missing = sorted(task_label_names - project_label_names)

    if missing:
        logger.warning(
            f"  Task '{task.name}' (ID: {task.id}) has labels missing from the "
            f"project: {missing}"
        )
        if typer.confirm(
            f"  Add {missing} to the project and proceed?", default=True
        ):
            _add_labels_to_project(projects_api, project_id, missing)
            project_label_names.update(missing)
            logger.info(f"  Added {missing} to project")
        else:
            logger.info(f"  Skipped '{task.name}' (ID: {task.id})")
            return False

    tasks_api.partial_update(
        task.id,
        patched_task_write_request=PatchedTaskWriteRequest(project_id=project_id),
    )
    logger.info(f"  Moved '{task.name}' (ID: {task.id})")
    return True


@app.command()
def main(
    project_name: str = typer.Option(
        "ENT CV", help="Name for the new CVAT project."
    ),
    project_id: int | None = typer.Option(
        None,
        envvar="CVAT_PROJECT_ID",
        help=(
            "Existing CVAT project ID to migrate tasks into. "
            "If omitted, a new project is created."
        ),
    ),
    host: str = typer.Option("https://cvat.jaehho.com"),
    port: int = typer.Option(443),
    username: str = typer.Option(default=..., envvar="CVAT_USERNAME", prompt=True),
    password: str = typer.Option(
        default=..., envvar="CVAT_PASSWORD", prompt=True, hide_input=True
    ),
) -> None:
    """Create a CVAT project with canonical labels and migrate orphan tasks."""
    with make_client(host=host, port=port, credentials=(username, password)) as client:
        projects_api = ProjectsApi(client.api_client)
        tasks_api = TasksApi(client.api_client)
        labels_api = LabelsApi(client.api_client)

        # -------------------------------------------------------------- project
        if project_id is not None:
            project = client.projects.retrieve(project_id)
            logger.info(
                f"Using existing project '{project.name}' (ID: {project.id})"
            )
        else:
            labels = [PatchedLabelRequest(name=name) for name in LABELS]
            (project, _) = projects_api.create(
                project_write_request=ProjectWriteRequest(
                    name=project_name, labels=labels,
                ),
            )
            logger.info(f"Created project '{project_name}' (ID: {project.id})")
            logger.info(f"  Labels: {list(LABELS)}")

        # Collect the current project label names (may grow during migration).
        project_label_names = {lbl.name for lbl in project.get_labels()}
        logger.info(
            f"  Project has {len(project_label_names)} labels: "
            f"{sorted(project_label_names)}"
        )

        # -------------------------------------------------------------- tasks
        all_tasks = _fetch_all_tasks(client)
        orphans = [
            t for t in all_tasks
            if getattr(t, "project_id", None) is None
        ]
        already_in = [
            t for t in all_tasks
            if getattr(t, "project_id", None) == project.id
        ]

        logger.info(f"Total tasks on server: {len(all_tasks)}")
        logger.info(f"  Already in this project: {len(already_in)}")
        logger.info(f"  Orphan (no project): {len(orphans)}")

        if not orphans:
            logger.success("No orphan tasks to migrate.")
        else:
            typer.echo()
            typer.echo("Orphan tasks to migrate:")
            for t in orphans:
                typer.echo(f"  [{t.id:4d}]  {t.name}")
            typer.echo()

            if typer.confirm(
                f"Move all {len(orphans)} orphan task(s) into project "
                f"'{project.name}' (ID: {project.id})?",
                default=True,
            ):
                moved = 0
                skipped = 0
                for t in orphans:
                    ok = _move_task(
                        tasks_api, labels_api, t, project.id,
                        project_label_names, projects_api,
                    )
                    if ok:
                        moved += 1
                    else:
                        skipped += 1
                logger.success(
                    f"Migrated {moved} task(s)"
                    + (f", skipped {skipped}" if skipped else "")
                )
            else:
                typer.echo("Skipped.")

        # -------------------------------------------------------------- next steps
        typer.echo()
        logger.success(
            f"Project ID: {project.id}\n"
            f"  Add to .env so task-creation commands inherit it automatically:\n"
            f"    CVAT_PROJECT_ID={project.id}"
        )


if __name__ == "__main__":
    app()
