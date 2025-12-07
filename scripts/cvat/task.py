from __future__ import annotations

import os
import sys
from pathlib import Path

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


def create_task(
    session: requests.Session,
    host: str,
    project_id: int,
    server_file: str,
    cfg: dict,
) -> int:
    task_name = f"{cfg.get('task_prefix', '')}{Path(server_file).stem}"
    logger.debug(f"Creating task '{task_name}' for file '{server_file}'")

    resp = session.post(
        f"{host}/api/tasks",
        json={"name": task_name, "project_id": project_id},
    )
    resp.raise_for_status()
    task_id = int(resp.json()["id"])

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


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = utils.load_config(config_path)

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

    for sf in tqdm(server_files, desc="Creating CVAT tasks"):
        create_task(
            session=session,
            host=HOST,
            project_id=project_id,
            server_file=sf,
            cfg=cfg,
        )

    logger.info("All tasks created.")


if __name__ == "__main__":
    main()
