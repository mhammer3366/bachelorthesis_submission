"""Shared path defaults for thesis reproduction scripts."""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parent.parent))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/ai/AI-DataPool/Datasets"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", DATA_ROOT.parent / "Models"))
# Historical typo on author machine; submission tree uses data_preparation/
LEGACY_PREP = REPO_ROOT / "data_preparation"
