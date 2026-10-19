"""Duong dan tap trung, doc lap vi tri Terminal (current working directory)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
VIDEOS = ROOT / "videos"
REPORTS = ROOT / "reports"

for directory in (DATA, MODELS, VIDEOS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

def _file(folder: Path, name: str) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def data_file(name: str) -> Path:
    return _file(DATA, name)

def model_file(name: str) -> Path:
    return _file(MODELS, name)

def report_file(name: str) -> Path:
    return _file(REPORTS, name)

def video_file(label: str, name: str) -> Path:
    return _file(VIDEOS, str(Path(label) / name))
