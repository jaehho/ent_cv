import os
import sys
import yaml
import requests
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

load_dotenv()

HOST: Final = os.getenv("CVAT_HOST")
USERNAME: Final = os.getenv("CVAT_USERNAME")
PASSWORD: Final = os.getenv("CVAT_PASSWORD")

def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Error: Config file not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        sys.exit(f"Error parsing YAML: {exc}")

def get_model_urn(session, host, model_name):
    """Finds the unique ID (URN) for the AI model by its name."""
    try:
        resp = session.get(f"{host}/api/lambda/functions")
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        sys.exit(f"Connection Error: {e}")

    for func in resp.json():
        if func.get("name") == model_name:
            return func["id"]
            
    sys.exit(f"Error: Model '{model_name}' not found. Please check the 'Models' tab in CVAT.")

def trigger_annotation(session, host, task_id, model_urn, threshold, cleanup):
    """Triggers the inference serverless function."""
    print(f"[-] Task {task_id}: Requesting annotation (Thresh: {threshold}, Cleanup: {cleanup})...", end=" ")
    
    payload = {
        "function": model_urn,
        "task": task_id,
        "cleanup": cleanup,
        "threshold": threshold,
        "convMaskToPoly": True 
    }

    try:
        resp = session.post(f"{host}/api/lambda/requests", json=payload)
        resp.raise_for_status()
        
        request_id = resp.json().get("id")
        print(f"Success! (Request ID: {request_id})")
        
    except requests.exceptions.HTTPError as e:
        print(f"Failed. Status: {e.response.status_code}. Msg: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():

    # 1. Load Config
    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)

    task_ids = cfg.get("task_ids")
    model_name = cfg.get("model_name")
    
    # Read new settings with defaults
    threshold = cfg.get("confidence_threshold", 0.5)
    cleanup = cfg.get("cleanup_old_annotations", False)

    if not model_name:
        sys.exit("Error: 'model_name' is missing in config.yaml")
    if not task_ids or not isinstance(task_ids, list):
        sys.exit("Error: 'task_ids' is missing or not a list in config.yaml")

    print(f"Loaded {len(task_ids)} tasks for model '{model_name}'.")

    # 2. Authenticate
    session = requests.Session()
    session.auth = (CVAT_USER, CVAT_PASS)

    # 3. Get Model URN
    model_urn = get_model_urn(session, CVAT_HOST, model_name)

    # 4. Iterate and Trigger
    for tid in task_ids:
        trigger_annotation(
            session=session, 
            host=CVAT_HOST, 
            task_id=tid, 
            model_urn=model_urn,
            threshold=threshold,
            cleanup=cleanup
        )

if __name__ == "__main__":
    main()