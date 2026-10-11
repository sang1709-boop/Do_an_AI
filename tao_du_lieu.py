"""
BƯỚC 4: Đọc video -> tính EAR/MAR/Head Pose -> cửa sổ trượt -> file CSV.

Cách sắp xếp video (bắt buộc):
    videos/
        tinh_tao/   S01_a.mp4   S02_a.mp4 ...   (video người tỉnh táo)
        buon_ngu/   S01_b.mp4   S02_b.mp4 ...   (video người buồn ngủ)
Phần tên file TRƯỚC dấu gạch dưới đầu tiên là mã người (S01, S02...).
Mã người dùng để chia train/test theo đối tượng ở Bước 5.

Cần cùng thư mục: dac_trung.py và face_landmarker.task
Chạy:  python buoc4_tao_du_lieu.py
Kết quả: du_lieu_dac_trung.csv
"""

import csv
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from dac_trung import (TAN_SO_MAU, TEN_DAC_TRUNG, lay_toa_do, tao_cac_cua_so,
                       tinh_ear, tinh_mar, tinh_tu_the_dau)

# ================= CẤU HÌNH =================
THU_MUC_VIDEO = Path("videos")
NHAN = {"tinh_tao": 0, "buon_ngu": 1}          # tên thư mục -> nhãn
DUOI_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
FILE_CSV = "du_lieu_dac_trung.csv"
CHIEU_RONG_TOI_DA = 640        # thu nhỏ video lớn để chạy nhanh hơn
GIOI_HAN_GIAY = None           # ví dụ 120 = chỉ xử lý 2 phút đầu mỗi video
# ============================================


def tao_detector():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def trich_chuoi_mau(duong_dan, detector):
    """Đọc 1 video, lấy mẫu TAN_SO_MAU khung/giây.

    Trả về mảng (N, 5): [ear, mar, pitch, yaw, roll] cho mỗi mẫu.
    Mẫu nào không thấy mặt thì cả hàng là NaN. Trả về None nếu không mở được video.
    """
    cap = cv2.VideoCapture(str(duong_dan))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:       # fps lỗi hoặc NaN
        fps = 30.0
    buoc = max(1, round(fps / TAN_SO_MAU))     # cứ `buoc` khung hình lấy 1 khung

    mau = []
    i = 0
    while True:
        if not cap.grab():                     # hết video
            break
        if i % buoc == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > CHIEU_RONG_TOI_DA:
                frame = cv2.resize(frame, (CHIEU_RONG_TOI_DA, int(h * CHIEU_RONG_TOI_DA / w)))
                h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(len(mau) * 1000 / TAN_SO_MAU)   # thời gian phải tăng dần
            ket_qua = detector.detect_for_video(mp_image, ts_ms)

            if ket_qua.face_landmarks:
                pts = lay_toa_do(ket_qua.face_landmarks[0], w, h)
                pitch, yaw, roll = tinh_tu_the_dau(ket_qua.facial_transformation_matrixes[0])
                mau.append([tinh_ear(pts), tinh_mar(pts), pitch, yaw, roll])
            else:
                mau.append([np.nan] * 5)

            if GIOI_HAN_GIAY and len(mau) >= GIOI_HAN_GIAY * TAN_SO_MAU:
                break
        i += 1
    cap.release()
    return np.array(mau, dtype=float)


def main():
    danh_sach = []
    for ten_thu_muc, nhan in NHAN.items():
        thu_muc = THU_MUC_VIDEO / ten_thu_muc
        if thu_muc.exists():
            for p in sorted(thu_muc.iterdir()):
                if p.suffix.lower() in DUOI_VIDEO:
                    danh_sach.append((p, nhan))

    if not danh_sach:
        print("Khong tim thay video nao. Hay tao thu muc va bo video vao:")
        for ten in NHAN:
            print(f"   {THU_MUC_VIDEO / ten}/   (ten file dang S01_xxx.mp4)")
        return

    print(f"Tim thay {len(danh_sach)} video. Bat dau xu ly...\n")
    dem_nhan, dem_nguoi = Counter(), Counter()

    with open(FILE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "video", "label", "t_start"] + TEN_DAC_TRUNG)

        for k, (p, nhan) in enumerate(danh_sach, 1):
            nguoi = p.stem.split("_")[0]
            print(f"[{k}/{len(danh_sach)}] {p.name}  (nguoi={nguoi}, nhan={nhan}) ...", end=" ", flush=True)

            detector = tao_detector()          # tạo mới cho mỗi video (reset thời gian)
            chuoi = trich_chuoi_mau(p, detector)
            detector.close()

            if chuoi is None:
                print("KHONG MO DUOC VIDEO, bo qua")
                continue
            co_mat = (~np.isnan(chuoi).any(axis=1)).mean() if len(chuoi) else 0
            so_dong = 0
            for s, vec in tao_cac_cua_so(chuoi):
                w.writerow([nguoi, p.name, nhan, round(s / TAN_SO_MAU, 2)] + [f"{x:.5f}" for x in vec])
                so_dong += 1
            f.flush()

            dem_nhan[nhan] += so_dong
            dem_nguoi[nguoi] += so_dong
            print(f"{len(chuoi)} mau, thay mat {co_mat:.0%}, tao {so_dong} cua so")

    print("\n===== TONG KET =====")
    print("So cua so theo nhan (0=tinh tao, 1=buon ngu):", dict(dem_nhan))
    print("So cua so theo nguoi:", dict(dem_nguoi))
    print("So nguoi:", len(dem_nguoi), "| Da luu:", FILE_CSV)
    if len(dem_nguoi) < 4:
        print("LUU Y: it hon 4 nguoi thi kho chia train/test theo doi tuong o Buoc 5.")


if __name__ == "__main__":
    main()