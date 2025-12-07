from __future__ import annotations

import os
import sys
import yaml
from pathlib import Path
from typing import Final

import requests
from dotenv import load_dotenv
from requests_toolbelt import MultipartEncoder

from ent_cv import config, utils

load_dotenv()

def make_server_paths(data_root: Path, target_subdir: str) -> list[str]:
    target_dir = data_root / target_subdir

    if not target_dir.exists():
        sys.exit(f"Target directory does not exist: {target_dir}")

    files = sorted(
        p
        for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
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


def create_task(
    session: requests.Session,
    host: str,
    project_id: int,
    server_file: str,
    cfg: dict,
) -> int:
    task_name = f"{cfg.get('task_prefix','')}{Path(server_file).stem}"
    resp = session.post(f"{host}/api/tasks", json={"name": task_name, "project_id": project_id})
    resp.raise_for_status()
    task_id = int(resp.json()["id"])

    fields = {
    # "chunk_size": 2147483647,
    "image_quality": str(cfg["image_quality"]),
    # "start_frame": 2147483647,
    # "stop_frame": 2147483647,
    "frame_filter": "step=" + str(cfg["frame_step"]),
    # "client_files": [],
    "server_files[0]": server_file,
    # "remote_files": [], # This could be used with onedrive links maybe
    "use_zip_chunks": str(cfg["use_zip_chunks"]),
    # "server_files_exclude": [],
    # "cloud_storage_id": 0,
    "use_cache": str(cfg["use_cache"]),
    "copy_data": str(cfg["copy_data"]),
    "storage_method": "cache", # Enum: "cache" "file_system"
    "sorting_method": cfg["sorting_method"], # Enum: "lexicographical" "natural" "predefined" "random"
    # "filename_pattern": "string",
    # "job_file_mapping": [
    #     [
    #     "string"
    #     ]
    # ],
    # "upload_file_order": [
    #     "string"
    # ],
    # "validation_params": {
    #     "mode": "gt",
    #     "frame_selection_method": "random_uniform",
    #     "random_seed": 0,
    #     "frames": [
    #     "string"
    #     ],
    #     "frame_count": 1,
    #     "frame_share": 0.1,
    #     "frames_per_job_count": 1,
    #     "frames_per_job_share": 0.1
    # }
    }

    encoder = MultipartEncoder(fields=fields)
    resp2 = session.post(
        f"{host}/api/tasks/{task_id}/data/",
        data=encoder,
        headers={"Content-Type": encoder.content_type},
    )
    resp2.raise_for_status()

    return task_id

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = utils.load_config(config_path)
    
    HOST = os.getenv("CVAT_HOST", "http://localhost:8080")
    USERNAME = os.getenv("CVAT_USERNAME", "username")
    PASSWORD = os.getenv("CVAT_PASSWORD", "password")

    data_dir = config.DATA_DIR
    target_subdir = cfg["target_subdir"]

    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    project_name = cfg["project_name"]
    project_id = get_project_id(session, HOST, project_name)

    server_files = make_server_paths(data_root=data_dir, target_subdir=target_subdir)

    print(f"data_root: {data_dir}")
    print(f"target_subdir: {target_subdir}")
    print(f"server_files: {server_files}")

    for sf in server_files:
        tid = create_task(
            session=session,
            host=HOST,
            project_id=project_id,
            server_file=sf,
            cfg=cfg,
        )
        print(f"Created task {tid} for {sf}")


if __name__ == "__main__":
    main()
