from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from requests_toolbelt import MultipartEncoder
from tqdm import tqdm

from ent_cv import config, utils
from ent_cv.config import logger

load_dotenv()


def make_server_paths(data_root: Path, target_subdir: str) -> list[str]:
    target_dir = data_root / target_subdir

    if not target_dir.exists():
        logger.error(f"Target directory does not exist: {target_dir}")
        sys.exit(1)

    files = sorted(
        p
        for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
    if not files:
        logger.error(f"No .mp4 files found under {target_dir}")
        sys.exit(1)

    rel_paths = [p.relative_to(data_root).as_posix() for p in files]
    logger.info(f"Found {len(rel_paths)} .mp4 file(s) under {target_dir}")
    return rel_paths


def get_project_id(session: requests.Session, host: str, name: str) -> int:
    logger.info(f"Looking up project id for project '{name}'")
    resp = session.get(f"{host}/api/projects")
    resp.raise_for_status()
    for p in resp.json().get("results", []):
        if p.get("name") == name:
            pid = int(p["id"])
            logger.info(f"Found project '{name}' with id={pid}")
            return pid

    logger.error(f"Project not found: {name}")
    sys.exit(1)


def get_function_id(session: requests.Session, host: str, model_name: str) -> str:
    """Retrieves the serverless function ID (ARN) by its display name."""
    logger.info(f"Looking up model/function ID for '{model_name}'")
    resp = session.get(f"{host}/api/lambda/functions")
    resp.raise_for_status()

    for func in resp.json():
        if func.get("name") == model_name:
            func_id = func["id"]
            logger.info(f"Found function '{model_name}' with id='{func_id}'")
            return str(func_id)

    logger.error(f"Function/Model not found: {model_name}")
    logger.info("Available functions: " + ", ".join([f["name"] for f in resp.json()]))
    sys.exit(1)


def trigger_annotation(
    session: requests.Session,
    host: str,
    task_id: int,
    function_id: str,
    cfg: dict[str, Any],
) -> None:
    """Triggers the serverless function to auto-annotate the specific task."""
    logger.debug(f"Triggering auto-annotation for task {task_id}...")

    payload = {
        "function": function_id,
        "task": task_id,
        # "job": 0,
        # "max_distance": 0,
        "threshold": cfg.get("threshold", 50),
        "cleanup": cfg.get("cleanup", False),
        "convMaskToPoly": cfg.get("conv_mask_to_poly", False),
        "conv_mask_to_poly": cfg.get("conv_mask_to_poly", False),
        # "mapping": {}
    }

    resp = session.post(f"{host}/api/lambda/requests", json=payload)

    try:
        resp.raise_for_status()
        job_id = resp.json().get("id")
        logger.info(f"Started annotation job {job_id} on task {task_id}")
    except requests.exceptions.HTTPError:
        logger.error(f"Failed to trigger task {task_id}: {resp.text}")
        # Continue with other tasks
        pass


def create_task(
    session: requests.Session,
    host: str,
    project_id: int,
    server_file: str,
    cfg: dict,
) -> int:
    task_name = f"{cfg.get('task_prefix', '')}{Path(server_file).stem}"
    logger.debug(f"Creating task '{task_name}' for file '{server_file}'")

    # 1) Create the task
    resp = session.post(
        f"{host}/api/tasks",
        json={"name": task_name, "project_id": project_id},
    )
    resp.raise_for_status()
    task_id = int(resp.json()["id"])

    # 2) Attach data (video) to the task
    fields = {
        # "chunk_size": 2147483647,
        "image_quality": str(cfg.get("image_quality", 70)),
        # "start_frame": 2147483647,
        # "stop_frame": 2147483647,
        "frame_filter": "step=" + str(cfg.get("frame_step", 1)),
        # "client_files": [],
        "server_files[0]": server_file,
        # "remote_files": [],
        "use_zip_chunks": str(cfg.get("use_zip_chunks", False)),
        # "server_files_exclude": [],
        # "cloud_storage_id": 0,
        "use_cache": str(cfg.get("use_cache", False)),
        "copy_data": str(cfg.get("copy_data", False)),
        "storage_method": "cache",
        "sorting_method": cfg.get("sorting_method", "lexicographical"),
        # "filename_pattern": "string",
        # "job_file_mapping": [["string"]],
        # "upload_file_order": ["string"],
        # "validation_params": {...}
    }

    encoder = MultipartEncoder(fields=fields)
    resp2 = session.post(
        f"{host}/api/tasks/{task_id}/data/",
        data=encoder,
        headers={"Content-Type": encoder.content_type},
    )
    resp2.raise_for_status()

    logger.info(f"Created task id={task_id} for file '{server_file}'")
    return task_id


def get_task_state(session: requests.Session, host: str, task_id: int) -> str:
    """
    Returns the task creation state from /api/tasks/{id}/status.
    Expected values (case-insensitive) include: 'Queued', 'Started', 'Finished', 'Failed'.
    """
    resp = session.get(f"{host}/api/tasks/{task_id}/status")
    resp.raise_for_status()
    data = resp.json()
    # CVAT uses "state", older versions might use "status"
    state = (data.get("state") or data.get("status") or "").lower()
    return state


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = utils.load_config(config_path)

    anno_cfg = cfg.get("auto_annotate", {})
    if not anno_cfg:
        logger.error("No 'auto_annotate' section found in config.yaml")
        sys.exit(1)

    HOST = os.getenv("CVAT_HOST", "http://localhost:8080")
    USERNAME = os.getenv("CVAT_USERNAME", "username")
    PASSWORD = os.getenv("CVAT_PASSWORD", "password")

    data_dir = config.DATA_DIR
    target_subdir = cfg["target_subdir"]

    logger.info(f"Using HOST={HOST}")
    logger.info(f"Using data_root={data_dir}")
    logger.info(f"Using target_subdir={target_subdir}")

    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    project_name = cfg["project_name"]
    project_id = get_project_id(session, HOST, project_name)

    server_files = make_server_paths(data_root=data_dir, target_subdir=target_subdir)
    logger.info(f"Server files to process: {server_files}")

    # PHASE 1: create all tasks (ingestion happens asynchronously in CVAT)
    task_ids: list[int] = []
    for sf in tqdm(server_files, desc="Creating CVAT tasks"):
        tid = create_task(
            session=session,
            host=HOST,
            project_id=project_id,
            server_file=sf,
            cfg=cfg,
        )
        task_ids.append(tid)

    logger.info(f"All tasks created: {task_ids}")

    # Resolve model/function ID once
    model_name = anno_cfg["model_name"]
    function_id = get_function_id(session, HOST, model_name)

    # PHASE 2: poll all tasks and trigger annotation as soon as each is ready
    poll_interval = float(anno_cfg.get("poll_interval", 5.0))
    finished_states = {"finished", "ready", "completed"}
    failed_states = {"failed", "error"}

    pending = set(task_ids)
    annotated: set[int] = set()

    logger.info("Waiting for tasks to finish data processing, then triggering auto-annotation...")
    with tqdm(total=len(task_ids), desc="Auto-annotation queued") as pbar:
        while pending:
            progressed = False
            for tid in list(pending):
                try:
                    state = get_task_state(session, HOST, tid)
                except Exception as e:
                    logger.warning(f"Failed to get status for task {tid}: {e}")
                    continue

                if state in finished_states:
                    trigger_annotation(
                        session=session,
                        host=HOST,
                        task_id=tid,
                        function_id=function_id,
                        cfg=anno_cfg,
                    )
                    pending.remove(tid)
                    annotated.add(tid)
                    pbar.update(1)
                    progressed = True
                elif state in failed_states:
                    logger.warning(f"Task {tid} finished with failure state '{state}', skipping auto-annotation.")
                    pending.remove(tid)
                    pbar.update(1)
                    progressed = True
                # else: still queued/started, keep waiting

            if pending and not progressed:
                time.sleep(poll_interval)

    logger.info(f"Auto-annotation triggered for tasks: {sorted(annotated)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
