from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import yaml
from cvat_sdk import make_client
from dotenv import load_dotenv

from ent_cv import config, utils

load_dotenv()

def load_labels(path: Path | None) -> list[Any]:
    if not path:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Label file not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"Invalid JSON in label file {path}: {exc}")


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    cfg = utils.load_config(config_path)

    HOST = os.getenv("CVAT_HOST", "http://localhost:8080")
    USERNAME = os.getenv("CVAT_USERNAME", "username")
    PASSWORD = os.getenv("CVAT_PASSWORD", "password")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    name = cfg.get("project_name") or f"project_{now}"
    description = cfg.get("project_description") or f"created_{now}"

    label_file = cfg.get("labels")
    labels = load_labels(Path(label_file) if label_file else None)

    try:
        with make_client(HOST, credentials=(USERNAME, PASSWORD)) as client:
            project = client.projects.create(
                {
                    "name": name,
                    "description": description,
                    "labels": labels,
                }
            )
    except Exception as exc:
        sys.exit(f"Failed to create project: {exc}")

    print(f"id={project.id} name={project.name}")


if __name__ == "__main__":
    main()
