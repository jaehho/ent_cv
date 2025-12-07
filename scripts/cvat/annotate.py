from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from ent_cv import config, utils
from ent_cv.config import logger

load_dotenv()


def get_project_id(session: requests.Session, host: str, name: str) -> int:
    """Retrieves the project ID by name."""
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


def get_project_tasks(session: requests.Session, host: str, project_id: int) -> list[int]:
    """Retrieves all task IDs associated with a specific project."""
    logger.info(f"Fetching tasks for project id={project_id}")

    # Increase page_size to capture more tasks; for production, implement proper pagination loops
    resp = session.get(f"{host}/api/tasks", params={"project_id": project_id, "page_size": 1000})
    resp.raise_for_status()

    data = resp.json()
    task_ids = [t["id"] for t in data.get("results", [])]

    if not task_ids:
        logger.warning(f"No tasks found for project id={project_id}")
    else:
        logger.info(f"Found {len(task_ids)} task(s) in project id={project_id}")

    return sorted(task_ids)


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
    logger.info("Available functions: " + ", ".join([f['name'] for f in resp.json()]))
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
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to trigger task {task_id}: {resp.text}")
        pass


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

    logger.info(f"Using HOST={HOST}")

    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # Determine Target Tasks
    specified_ids = anno_cfg.get("task_ids")

    if specified_ids and isinstance(specified_ids, list):
        logger.info(f"Using specified task IDs from config: {specified_ids}")
        task_ids = specified_ids
    else:
        logger.info("No specific task IDs found in config. Falling back to Project lookup.")
        project_name = cfg["project_name"]

        project_id = get_project_id(session, HOST, project_name)
        task_ids = get_project_tasks(session, HOST, project_id)

    # Resolve Model ID
    model_name = anno_cfg["model_name"]
    function_id = get_function_id(session, HOST, model_name)

    # Trigger Annotation
    for tid in tqdm(task_ids, desc="Triggering Auto-Annotation"):
        trigger_annotation(
            session=session,
            host=HOST,
            task_id=tid,
            function_id=function_id,
            cfg=anno_cfg
        )
        # Brief pause to be gentle on the API
        time.sleep(0.5)

if __name__ == "__main__":
    main()