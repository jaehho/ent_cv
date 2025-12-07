from __future__ import annotations

import os
import sys
import time
import yaml
from pathlib import Path
from typing import Final, Optional

import requests
from dotenv import load_dotenv
from requests_toolbelt import MultipartEncoder

load_dotenv()

HOST: Final = os.getenv("CVAT_HOST")
USERNAME: Final = os.getenv("CVAT_USERNAME")
PASSWORD: Final = os.getenv("CVAT_PASSWORD")


def load_config(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except yaml.YAMLError as exc:
        sys.exit(f"Invalid YAML in config file: {exc}")


def make_server_paths(data_root: Path, target_subdir: str) -> list[str]:
    target_dir = data_root / target_subdir

    if not target_dir.exists():
        sys.exit(f"Target directory does not exist: {target_dir}")

    files = sorted(p for p in target_dir.rglob("*.mp4") if p.is_file())
    if not files:
        sys.exit(f"No .mp4 files found under {target_dir}")

    return [p.relative_to(data_root).as_posix() for p in files]


def get_project_id(session: requests.Session, host: str, name: str) -> int:
    resp = session.get(f"{host}/api/projects")
    resp.raise_for_status()
    for p in resp.json().get("results", []):
        if p.get("name") == name:
            return int(p["id"])
    sys.exit(f"Project not found: {name}")


def get_model_urn(session: requests.Session, host: str, model_name: str) -> str:
    """Finds the Unique Resource Name (URN) for a specific serverless function/model."""
    resp = session.get(f"{host}/api/lambda/functions")
    resp.raise_for_status()
    
    # List all available models to debug if needed
    available_models = []
    
    for func in resp.json():
        name = func.get("name")
        available_models.append(name)
        if name == model_name:
            return func["id"]  # The 'id' field usually contains the URN
            
    sys.exit(f"Model '{model_name}' not found on server. Available models: {available_models}")


def create_task(
    session: requests.Session,
    host: str,
    project_id: int,
    server_file: str,
    cfg: dict,
) -> int:
    task_name = f"{cfg.get('task_prefix', '')}{Path(server_file).stem}"
    resp = session.post(f"{host}/api/tasks", json={"name": task_name, "project_id": project_id})
    resp.raise_for_status()
    task_id = int(resp.json()["id"])

    fields = {
        "image_quality": str(cfg["image_quality"]),
        "frame_filter": "step=" + str(cfg["frame_step"]),
        "server_files[0]": server_file,
        "use_zip_chunks": str(cfg["use_zip_chunks"]),
        "use_cache": str(cfg["use_cache"]),
        "copy_data": str(cfg["copy_data"]),
        "storage_method": "cache",
        "sorting_method": cfg["sorting_method"],
    }

    encoder = MultipartEncoder(fields=fields)
    resp2 = session.post(
        f"{host}/api/tasks/{task_id}/data/",
        data=encoder,
        headers={"Content-Type": encoder.content_type},
    )
    resp2.raise_for_status()

    return task_id


def wait_for_task_processing(session: requests.Session, host: str, task_id: int) -> None:
    """Polls the task status until data processing is complete."""
    print(f"Waiting for task {task_id} data processing...", end="", flush=True)
    while True:
        resp = session.get(f"{host}/api/tasks/{task_id}")
        resp.raise_for_status()
        data = resp.json()
        
        # Check specific status fields. Note: API structure can vary slightly by version.
        # Usually checking 'status' or 'jobs' status is required.
        # If 'size' is 0, it's definitely not ready.
        if data.get("size", 0) > 0:
             # Often CVAT returns immediately, but we need to ensure the jobs are created
            if data.get("jobs") and len(data["jobs"]) > 0:
                print(" Done.")
                return
        
        time.sleep(2)
        print(".", end="", flush=True)


def trigger_auto_annotation(
    session: requests.Session, 
    host: str, 
    task_id: int, 
    model_urn: str,
    cleanup: bool = False
) -> None:
    """Triggers the lambda function (AI model) on the specified task."""
    print(f"Starting automatic annotation for task {task_id}...")
    
    payload = {
        "function": model_urn,
        "task": task_id,
        "cleanup": cleanup,  # If True, removes existing annotations
        "convMaskToPoly": True, # Useful for segmentation models
        "threshold": 0.5 # Confidence threshold (optional, depends on model)
    }
    
    resp = session.post(f"{host}/api/lambda/requests", json=payload)
    
    # CVAT might return 200 (OK) or 202 (Accepted)
    if resp.status_code not in [200, 201, 202]:
        print(f"Failed to trigger annotation: {resp.text}")
        resp.raise_for_status()
        
    # The response usually contains a 'job' ID (the background inference job, not a CVAT labeling job)
    request_id = resp.json().get("id")
    print(f"Annotation request {request_id} started successfully.")


def main() -> None:
    if not HOST or not USERNAME or not PASSWORD:
        sys.exit("Missing CVAT host or credentials in .env")

    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)

    data_root = Path(cfg["data_root"])
    target_subdir = cfg["target_subdir"]
    model_name = cfg.get("model_name") # Add this to your YAML

    if not model_name:
        sys.exit("Please add 'model_name' (e.g., 'YOLO v7') to your config.yaml")

    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # 1. Get Project ID
    project_name = cfg["project_name"]
    project_id = get_project_id(session, HOST, project_name)

    # 2. Get Model URN (ID)
    model_urn = get_model_urn(session, HOST, model_name)
    print(f"Found model '{model_name}' with URN: {model_urn}")

    server_files = make_server_paths(data_root=data_root, target_subdir=target_subdir)

    print(f"data_root: {data_root}")
    print(f"target_subdir: {target_subdir}")
    print(f"server_files: {server_files}")

    for sf in server_files:
        # 3. Create Task
        tid = create_task(
            session=session,
            host=HOST,
            project_id=project_id,
            server_file=sf,
            cfg=cfg,
        )
        print(f"Created task {tid} for {sf}")

        # 4. Wait for video processing (CRITICAL STEP)
        wait_for_task_processing(session, HOST, tid)

        # 5. Run Auto Annotation
        trigger_auto_annotation(session, HOST, tid, model_urn, cleanup=False)


if __name__ == "__main__":
    main()