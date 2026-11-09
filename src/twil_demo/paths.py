"""Project paths, independent of the shell's working directory."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("TWIL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "garmin"
CLEAN = ROOT / "data" / "clean"
DOCS = ROOT / "docs"
OUTPUT = ROOT / "output"
CHARTS = OUTPUT / "charts"
DASHBOARD = OUTPUT / "dashboard"
INDEX = OUTPUT / "rag_index.npz"


def ensure_directories() -> None:
    for path in (CLEAN, DOCS / "garmin", OUTPUT, CHARTS, DASHBOARD):
        path.mkdir(parents=True, exist_ok=True)
