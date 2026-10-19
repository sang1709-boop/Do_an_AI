"""Đường dẫn thống nhất, độc lập thư mục đang đứng khi chạy Python."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
VIDEOS = ROOT / "videos"
REPORTS = ROOT / "reports"
for directory in (DATA, MODELS, VIDEOS, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

def data_file(name): return DATA / name
def model_file(name): return MODELS / name
def report_file(name): return REPORTS / name
