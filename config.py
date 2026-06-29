import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load .env if present (never committed — keeps secrets out of source control)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# --- Dataset paths (override via environment variables or edit directly) ---

# VinDr-Mammo DICOM root directory
VINDR_DIR = Path(os.environ.get("VINDR_DIR", BASE_DIR / "data" / "vindr"))

# RSNA 2022 DICOM root directory
RSNA_DIR = Path(os.environ.get("RSNA_DIR", BASE_DIR / "data" / "rsna"))

# Preprocessed PNG output directories
VINDR_PNG_DIR = Path(os.environ.get("VINDR_PNG_DIR", BASE_DIR / "data" / "vindr_png"))
RSNA_PNG_DIR  = Path(os.environ.get("RSNA_PNG_DIR",  BASE_DIR / "data" / "rsna_png"))

# Model checkpoints and results
MODELS_DIR  = Path(os.environ.get("MODELS_DIR",  BASE_DIR / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", BASE_DIR / "results"))

for d in [VINDR_PNG_DIR, RSNA_PNG_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
