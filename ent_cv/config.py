from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path("/mnt") / "data" / "ent_cv"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PREDICTIONS_DIR = DATA_DIR / "predictions"
DATASETS_DIR = DATA_DIR / "datasets"

PRIORITY_CASES = [
    "20251113_02", "20251124_01"
    # "20251113_02", "20251124_01", "20251204_01", "20251208_01", "20251208_02",
    # "20251210_02", "20251210_03", "20251217_02", "20251217_03", "20251218_02",
    # "20260128_01", "20260205_01", "20260205_03", "20260212_02",
]

MODELS_DIR = DATA_DIR / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ORDER IS SIGNIFICANT - index corresponds to YOLO class ID.
LABELS = [
    "Forceps",
    "Non-forcep instruments",
    "Suction",
    "Navigation probe",
    "Navigation suction",
    "Energy-based hemostasis without suction",
    "Saline-enhanced energy delivery",
    "Suction bovie",
    "Microdebrider",
    "Drill",
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass