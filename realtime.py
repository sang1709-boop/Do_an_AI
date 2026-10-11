"""
BƯỚC 6: Phát hiện buồn ngủ theo thời gian thực + cảnh báo âm thanh.

Cần cùng thư mục: model.pkl, face_landmarker.task, dac_trung.py, camera_cfg.py
Chạy:  python buoc6_realtime.py

Phím:  q = thoát | r = xóa bộ nhớ đệm | [ ] = giảm / tăng ngưỡng cảnh báo 0.05

Lưu ý: KHÔNG lật hình (không soi gương) để khớp với dữ liệu huấn luyện.
"""

import threading
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import winsound          # chỉ có trên Windows
except ImportError:
    winsound = None

from camera_cfg import mo_camera
from dac_trung import (CUA_SO, MAT_PHAI, MAT_TRAI, MIENG_DOC, MIENG_NGANG,
                       TAN_SO_MAU, lay_toa_do, tinh_ear, tinh_mar,
                       tinh_tu_the_dau, vector_cua_so)

# ================= CẤU HÌNH =================
FILE_MODEL = "model.pkl"
CHIEU_RONG_TOI_DA = 640     # giống Bước 4
DU_DOAN_MOI = 5             # cứ 5 mẫu (~0.33 giây) dự đoán lại một lần
SO_LAN_LIEN_TIEP = 3        # cần 3 lần dự đoán liên tiếp vượt ngưỡng mới báo động
# ============================================

DIEM_VE = MAT_PHAI + MAT_TRAI + list(MIENG_NGANG) + [i for cap_ in MIENG_DOC for i in cap_]

_dang_keu = False


def keu_canh_bao():
    """Phát tiếng bíp ở luồng riêng để không làm đứng hình."""
    global _dang_keu
    if _dang_keu:
        return
    _dang_keu = True

    def chay():
        global _dang_keu
        try:
            if winsound:
                winsound.Beep(1200, 350)
            else:
                print("\a", end="", flush=True)
            time.sleep(0.25)
        finally:
            _dang_keu = False

    threading.Thread(target=chay, daemon=True).start()


def tao_detector():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    goi = joblib.load(FILE_MODEL)
    mo_hinh, nguong = goi["pipeline"], goi["nguong"]
    detector = tao_detector()
    cap = mo_camera()

    dem = deque(maxlen=CUA_SO)                 # cửa sổ trượt các mẫu gần nhất
    gan_day = deque(maxlen=SO_LAN_LIEN_TIEP)   # các xác suất gần nhất
    dt = 1.0 / TAN_SO_MAU
    t0 = time.time()
    tiep = 0.0                                 # thời điểm lấy mẫu kế tiếp
    ts_cuoi = -1
    mau_tu_lan_cuoi = 0

    xac_suat, bao_dong, thay_mat = None, False, False
    ear = mar = None
    diem_ve = None
    thoi_gian_xl = []                          # ms xử lý mỗi mẫu (cho báo cáo)
    fps_hien, t_truoc = 0.0, time.time()
    so_khung = 0

    print("Dang chay. Bam q de thoat.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Khong doc duoc khung hinh.")
            break
        so_khung += 1
        t = time.time() - t0
        h, w = frame.shape[:2]

        # ---- Lấy mẫu đúng nhịp TAN_SO_MAU (giống lúc huấn luyện) ----
        if t >= tiep:
            bd = time.perf_counter()
            xl = frame
            if w > CHIEU_RONG_TOI_DA:
                xl = cv2.resize(frame, (CHIEU_RONG_TOI_DA, int(h * CHIEU_RONG_TOI_DA / w)))
            hx, wx = xl.shape[:2]
            rgb = cv2.cvtColor(xl, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_cuoi = max(ts_cuoi + 1, int(t * 1000))
            kq = detector.detect_for_video(mp_image, ts_cuoi)

            if kq.face_landmarks:
                lms = kq.face_landmarks[0]
                pts = lay_toa_do(lms, wx, hx)
                ear, mar = tinh_ear(pts), tinh_mar(pts)
                pitch, yaw, roll = tinh_tu_the_dau(kq.facial_transformation_matrixes[0])
                hang = [ear, mar, pitch, yaw, roll]
                diem_ve = [(lms[i].x, lms[i].y) for i in DIEM_VE]
            else:
                hang = [np.nan] * 5
                ear = mar = None
                diem_ve = None

            # Nếu xử lý chậm hơn nhịp lấy mẫu, lặp lại mẫu để cửa sổ vẫn đủ ~2 giây
            so_slot = int((t - tiep) / dt) + 1
            if so_slot > TAN_SO_MAU:
                so_slot = TAN_SO_MAU
                tiep = t + dt
            else:
                tiep += so_slot * dt
            for _ in range(so_slot):
                dem.append(hang)
            mau_tu_lan_cuoi += so_slot

            # ---- Dự đoán ----
            if len(dem) == CUA_SO and mau_tu_lan_cuoi >= DU_DOAN_MOI:
                mau_tu_lan_cuoi = 0
                vec = vector_cua_so(np.array(dem))
                if vec is None:                       # thiếu mặt quá nhiều
                    thay_mat, xac_suat, bao_dong = False, None, False
                    gan_day.clear()
                else:
                    thay_mat = True
                    xac_suat = float(mo_hinh.predict_proba(vec.reshape(1, -1))[0, 1])
                    gan_day.append(xac_suat)
                    bao_dong = (len(gan_day) == SO_LAN_LIEN_TIEP
                                and all(p >= nguong for p in gan_day))
            thoi_gian_xl.append((time.perf_counter() - bd) * 1000)

        if bao_dong:
            keu_canh_bao()

        # ---- Vẽ giao diện ----
        if diem_ve:
            for x, y in diem_ve:
                cv2.circle(frame, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)

        if bao_dong:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "CANH BAO: BUON NGU!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        elif xac_suat is None:
            msg = "Dang thu thap..." if len(dem) < CUA_SO else "Khong thay khuon mat"
            cv2.putText(frame, msg, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "Tinh tao", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

        # thanh xác suất buồn ngủ + vạch ngưỡng
        bx, by, bw, bh = 20, h - 40, 220, 18
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        if xac_suat is not None:
            mau = (0, 0, 255) if xac_suat >= nguong else (0, 200, 0)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * xac_suat), by + bh), mau, -1)
        cv2.line(frame, (bx + int(bw * nguong), by - 4), (bx + int(bw * nguong), by + bh + 4), (0, 255, 255), 2)
        txt_xs = "--" if xac_suat is None else f"{xac_suat:.2f}"
        cv2.putText(frame, f"P(buon ngu)={txt_xs}  nguong={nguong:.2f}", (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if ear is not None:
            cv2.putText(frame, f"EAR {ear:.2f}  MAR {mar:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        now = time.time()
        fps_hien = 0.9 * fps_hien + 0.1 * (1.0 / max(now - t_truoc, 1e-6))
        t_truoc = now
        cv2.putText(frame, f"FPS {fps_hien:.0f}", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Phat hien buon ngu - realtime (q: thoat)", frame)
        phim = cv2.waitKey(1) & 0xFF
        if phim == ord("q"):
            break
        elif phim == ord("r"):
            dem.clear(); gan_day.clear(); xac_suat, bao_dong = None, False
        elif phim == ord("["):
            nguong = max(0.05, nguong - 0.05)
        elif phim == ord("]"):
            nguong = min(0.95, nguong + 0.05)

    cap.release()
    cv2.destroyAllWindows()

    # ---- Tổng kết hiệu năng (dùng cho báo cáo) ----
    tong = time.time() - t0
    if thoi_gian_xl:
        a = np.array(thoi_gian_xl)
        tk = [
            f"Thoi gian chay           : {tong:.1f} s",
            f"So mau da xu ly          : {len(a)}",
            f"FPS hien thi trung binh  : {so_khung / tong:.1f}",
            f"Xu ly 1 mau (trung binh) : {a.mean():.1f} ms",
            f"Xu ly 1 mau (95%)        : {np.percentile(a, 95):.1f} ms",
            f"Do tre cua so            : {CUA_SO / TAN_SO_MAU:.1f} s (do cua so 30 mau)",
        ]
        print("\n=== TONG KET HIEU NANG ===")
        print("\n".join(tk))
        with open("ket_qua_realtime.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(tk))


if __name__ == "__main__":
    main()