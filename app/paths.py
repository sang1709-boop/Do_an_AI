"""Duong dan trong source va ban EXE PyInstaller onedir."""
from pathlib import Path
import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
if FROZEN:
    RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    ROOT = Path(sys.executable).resolve().parent
    WRITE_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AI_Driver_Monitor"
else:
    ROOT = Path(__file__).resolve().parent.parent
    RESOURCE_ROOT = ROOT
    WRITE_ROOT = ROOT
DATA = WRITE_ROOT / "data"
MODELS = RESOURCE_ROOT / "models"
VIDEOS = WRITE_ROOT / "videos"
REPORTS = WRITE_ROOT / "reports"
for folder in (DATA, VIDEOS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)
if FROZEN:
    source_config = RESOURCE_ROOT / "data" / "nguong_phu.json"
    target_config = DATA / "nguong_phu.json"
    if source_config.exists() and not target_config.exists():
        import shutil
        shutil.copyfile(source_config, target_config)

def _file(folder, name):
    p = folder / name
    p.parent.mkdir(parents=True, exist_ok=True) if folder != MODELS else None
    return p

def data_file(name): return _file(DATA, name)
def model_file(name): return _file(MODELS, name)
def report_file(name): return _file(REPORTS, name)
def video_file(label, name): return _file(VIDEOS, str(Path(label) / name))
