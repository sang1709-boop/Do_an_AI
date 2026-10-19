"""Lệnh tương thích: python kiem_tra_du_lieu.py."""
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file
from app.huan_luyen import phan_tich

if __name__ == "__main__":
    phan_tich(chay_so_sanh=False)
