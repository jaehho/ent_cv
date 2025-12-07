import sys
import yaml
from pathlib import Path

def load_config(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except yaml.YAMLError as exc:
        sys.exit(f"Invalid YAML in config file: {exc}")