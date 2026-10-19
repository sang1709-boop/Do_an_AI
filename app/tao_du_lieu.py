"""
BƯỚC 4: Đọc video -> tính EAR/MAR/Head Pose -> cửa sổ trượt -> file CSV.

Cách sắp xếp video (bắt buộc):
    videos/
        tinh_tao/   S01_ngay_1.mp4   S02_dem_1.mp4 ...   (video người tỉnh táo)
        buon_ngu/   S01_ngay_1.mp4   S02_dem_1.mp4 ...   (video người buồn ngủ, nhắm mắt/ngáp/gật)
        ngu_mat_mo/ S01_ngay_1.mp4 ...   (tùy chọn: giấc ngủ trắng - mắt mở nhưng nhìn trân trân)
Phần tên file TRƯỚC dấu gạch dưới đầu tiên là mã người (S01, S02...).
Mã người dùng để chia train/test theo đối tượng ở Bước 5.
Phần thứ hai (nếu có) là ĐIỀU KIỆN QUAY, một trong:
    ngay      ban ngày
    dem       ban đêm
    kinh      ban ngày, đeo kính
    demkinh   ban đêm, đeo kính (dễ bị loá)
    dui       dụi MỘT mắt (đặt trong thư mục tinh_tao: đây là mẫu "không phải buồn ngủ")
    lai       ngồi như đang lái, nhìn thẳng tập trung vào đường (thư mục tinh_tao). RẤT QUAN TRỌNG
              cho bộ phát hiện giấc ngủ trắng: đây là kiểu "nhìn thẳng ít chớp mắt" mà vẫn tỉnh.
Tên cũ kiểu S01_a.mp4 / S01_1.mp4 vẫn chạy, điều kiện ghi là "khong_ro".

Cần cùng thư mục: dac_trung.py, anh_sang.py và face_landmarker.task
Chạy:  python tao_du_lieu.py
Kết quả: du_lieu_dac_trung.csv (cửa sổ 2 s, tinh_tao / buon_ngu)
         du_lieu_dai.csv       (cửa sổ 10 s, tinh_tao / ngu_mat_mo; chỉ tạo khi có video ngu_mat_mo)
         nguong_phu.json       (ngưỡng lệch hai mắt tính từ dữ liệu)
Video ngu_mat_mo KHÔNG đưa vào CSV 2 s (mắt mở sẽ trông như tỉnh táo và làm nhiễu mô hình đó).

Cách xử lý dụi mắt / kính loá (cùng một hàm hang_mau với realtime.py):
  - Mỗi mẫu tính EAR riêng từng mắt. Nếu hai mắt lệch nhau quá ngưỡng thì mẫu đó dùng mắt MỞ
    HƠN (dụi mắt: một mắt bị che, mắt kia vẫn mở -> không bị coi là nhắm).
  - Ngưỡng lệch KHÔNG đoán: lấy phân vị P_LECH của độ lệch hai mắt trên toàn bộ video
    (trừ video 'dui'), tức là "vượt quá mức lệch tự nhiên của 99.5% mẫu bình thường".
  - Mắt loá (quá nhiều pixel cháy sáng quanh mắt) không được tin, dùng mắt còn lại.
"""
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file

import csv
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app.anh_sang import NGUONG_LOA_MAT, BoXuLyAnh
from app.tai_xe import FILE_POSE, chon_tai_xe, tao_pose_detector, tay_gan_mat
from app.dac_trung import RONG_MAT
from app.dac_trung import (MAT_PHAI, MAT_TRAI, NGUONG_LECH_MAT_MAC_DINH, TAN_SO_MAU,
                       TEN_DAC_TRUNG, TEN_DAC_TRUNG_DAI, ear_hai_mat, hang_mau,
                       huong_nhin, lay_toa_do, lech_hai_mat, luu_nguong_phu,
                       tao_cac_cua_so, tao_cac_cua_so_dai, tinh_mar, tinh_tu_the_dau)

# ================= CẤU HÌNH =================
THU_MUC_VIDEO = VIDEOS
NHAN = {"tinh_tao": 0, "buon_ngu": 1}          # CSV 2 giây: tên thư mục -> nhãn
NHAN_DAI = {"tinh_tao": 0, "ngu_mat_mo": 1}    # CSV 10 giây (giấc ngủ trắng)
DUOI_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
DIEU_KIEN = {"ngay", "dem", "kinh", "demkinh", "dui", "lai"}
FILE_CSV = str(data_file("du_lieu_dac_trung.csv"))
FILE_CSV_DAI = str(data_file("du_lieu_dai.csv"))
CHIEU_RONG_TOI_DA = 640        # thu nhỏ video lớn để chạy nhanh hơn
GIOI_HAN_GIAY = None           # ví dụ 120 = chỉ xử lý 2 phút đầu mỗi video
P_LECH = 99.5                  # phân vị dùng làm ngưỡng "lệch hai mắt bất thường"
SO_MAU_TOI_THIEU_NGUONG = 300  # ít hơn số mẫu này thì không đủ tin cậy, dùng ngưỡng mặc định
# ============================================


def tao_detector():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_file("face_landmarker.task"))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def tach_ten(duong_dan):
    """S01_dem_2.mp4 -> ("S01", "dem");  S01_a.mp4 -> ("S01", "khong_ro")."""
    phan = duong_dan.stem.split("_")
    dk = phan[1] if len(phan) >= 3 and phan[1] in DIEU_KIEN else "khong_ro"
    return phan[0], dk


def trich_chuoi_mau(duong_dan, detector, bo_anh):
    """Đọc 1 video, lấy mẫu TAN_SO_MAU khung/giây.

    Trả về mảng (N, 12): [ear_phai, ear_trai, mar, pitch, yaw, roll, loa_phai, loa_trai, gx, gy, che_phai, che_trai].
    Mẫu nào không thấy mặt thì 6 số đầu và gx, gy là NaN. Trả về None nếu không mở được video.
    """
    cap = cv2.VideoCapture(str(duong_dan))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:       # fps lỗi hoặc NaN
        fps = 30.0

    mau = []
    # Dùng Pose trong khi tạo dữ liệu để xử lý che mắt giống realtime.
    pose_det = tao_pose_detector(FILE_POSE) if Path(FILE_POSE).exists() else None
    pose_tx = None
    i = 0
    mau_ke_tiep = 0
    while True:
        if not cap.grab():                     # hết video
            break
        if (i / fps) + 1e-9 >= mau_ke_tiep / TAN_SO_MAU:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > CHIEU_RONG_TOI_DA:
                frame = cv2.resize(frame, (CHIEU_RONG_TOI_DA, int(h * CHIEU_RONG_TOI_DA / w)))
                h, w = frame.shape[:2]

            anh = bo_anh.xu_ly(frame)          # chuẩn hoá sáng ngày/đêm + giảm loá
            rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(len(mau) * 1000 / TAN_SO_MAU)   # thời gian phải tăng dần
            ket_qua = detector.detect_for_video(mp_image, ts_ms)

            if ket_qua.face_landmarks and ket_qua.facial_transformation_matrixes:
                pts = lay_toa_do(ket_qua.face_landmarks[0], w, h)
                pitch, yaw, roll = tinh_tu_the_dau(ket_qua.facial_transformation_matrixes[0])
                ear_p, ear_t = ear_hai_mat(pts)
                loa_p = bo_anh.ti_le_loa(pts, MAT_PHAI) >= NGUONG_LOA_MAT
                loa_t = bo_anh.ti_le_loa(pts, MAT_TRAI) >= NGUONG_LOA_MAT
                gx, gy = huong_nhin(pts)
                if pose_det is not None and len(mau) % 2 == 0:
                    pose_kq = pose_det.detect_for_video(mp_image, ts_ms)
                    pose_tx = (chon_tai_xe(pose_kq.pose_landmarks, w, h, pts[1])
                               if pose_kq.pose_landmarks else None)
                che_p, che_t = tay_gan_mat(pose_tx, pts, MAT_PHAI, MAT_TRAI, RONG_MAT, w, h)
                mau.append([ear_p, ear_t, tinh_mar(pts), pitch, yaw, roll,
                             float(loa_p), float(loa_t), gx, gy, float(che_p), float(che_t)])
            else:
                pose_tx = None
                mau.append([np.nan] * 6 + [0.0, 0.0, np.nan, np.nan, 0.0, 0.0])

            mau_ke_tiep += 1
            if GIOI_HAN_GIAY and len(mau) >= GIOI_HAN_GIAY * TAN_SO_MAU:
                break
        i += 1
    cap.release()
    if pose_det is not None:
        pose_det.close()
    return np.asarray(mau, dtype=float).reshape(-1, 12)


def ap_dung_quy_tac(tho, nguong_lech):
    """Mảng thô (N, 12) -> (chuoi (N, 5), danh sách lý do từng mẫu).

    Lý do: ok | dui | loa | khong_do_duoc | khong_thay_mat
    """
    chuoi = np.full((len(tho), 5), np.nan)
    ly_do = []
    for i, r in enumerate(tho):
        if np.isnan(r[0]):
            ly_do.append("khong_thay_mat")
            continue
        hang, l = hang_mau(r[0], r[1], r[2], r[3], r[4], r[5],
                           loa_p=bool(r[6]), loa_t=bool(r[7]),
                           che_p=bool(r[10]), che_t=bool(r[11]), nguong_lech=nguong_lech)
        chuoi[i] = hang
        ly_do.append(l)
    return chuoi, ly_do


def tinh_nguong_lech(danh_sach_tho):
    """Ngưỡng lệch hai mắt = phân vị P_LECH của độ lệch trên các mẫu bình thường.

    danh_sach_tho: [(mảng thô, điều kiện)]. Bỏ qua video 'dui' (chúng chính là thứ cần phát hiện)
    và các mẫu có mắt loá (độ lệch do loá, không phải do mắt).
    """
    lech = []
    for tho, dk in danh_sach_tho:
        if dk == "dui":
            continue
        for r in tho:
            if not np.isnan(r[0]) and not r[6] and not r[7]:
                lech.append(lech_hai_mat(r[0], r[1]))
    lech = np.array(lech)
    if len(lech) < SO_MAU_TOI_THIEU_NGUONG:
        print(f"  Chi co {len(lech)} mau (< {SO_MAU_TOI_THIEU_NGUONG}) -> dung nguong mac dinh "
              f"{NGUONG_LECH_MAT_MAC_DINH}")
        return NGUONG_LECH_MAT_MAC_DINH, {"so_mau": int(len(lech)), "mac_dinh": True}
    nguong = float(np.percentile(lech, P_LECH))
    tk = {"so_mau": int(len(lech)), "p50": float(np.percentile(lech, 50)),
          "p95": float(np.percentile(lech, 95)), "p99": float(np.percentile(lech, 99)),
          f"p{P_LECH}": nguong, "max": float(lech.max()), "mac_dinh": False}
    print(f"  Do lech hai mat tren {len(lech)} mau binh thuong: trung vi {tk['p50']:.3f}, "
          f"P95 {tk['p95']:.3f}, P99 {tk['p99']:.3f}, max {tk['max']:.3f}")
    return nguong, tk


def main():
    cac_thu_muc = list(dict.fromkeys(list(NHAN) + list(NHAN_DAI)))   # tinh_tao, buon_ngu, ngu_mat_mo
    danh_sach = []
    for ten_thu_muc in cac_thu_muc:
        thu_muc = THU_MUC_VIDEO / ten_thu_muc
        if thu_muc.exists():
            for p in sorted(thu_muc.iterdir()):
                if p.suffix.lower() in DUOI_VIDEO:
                    danh_sach.append((p, ten_thu_muc))

    if not danh_sach:
        print("Khong tim thay video nao. Hay tao thu muc va bo video vao:")
        for ten in cac_thu_muc:
            print(f"   {THU_MUC_VIDEO / ten}/   (ten file dang S01_ngay_1.mp4)")
        return

    # ---------- Lượt 1: đọc video, lấy số đo thô ----------
    print(f"Tim thay {len(danh_sach)} video. Luot 1/2: trich dac trung tho...\n")
    cac_video = []      # (đường dẫn, thư mục, người, điều kiện, mảng thô)
    for k, (p, ten_thu_muc) in enumerate(danh_sach, 1):
        nguoi, dk = tach_ten(p)
        print(f"[{k}/{len(danh_sach)}] {ten_thu_muc}/{p.name}  (nguoi={nguoi}, dieu kien={dk}) ...",
              end=" ", flush=True)
        detector = tao_detector()          # tạo mới cho mỗi video (reset thời gian)
        bo_anh = BoXuLyAnh("auto")         # mới cho mỗi video (reset độ sáng)
        tho = trich_chuoi_mau(p, detector, bo_anh)
        detector.close()
        if tho is None:
            print("KHONG MO DUOC VIDEO, bo qua")
            continue
        print(f"{len(tho)} mau, che do cuoi: {bo_anh.che_do}")
        cac_video.append((p, ten_thu_muc, nguoi, dk, tho))

    # ---------- Ngưỡng lệch hai mắt từ chính dữ liệu ----------
    print("\nTinh nguong lech hai mat (dui mat) tu du lieu:")
    nguong_lech, tk = tinh_nguong_lech([(v[4], v[3]) for v in cac_video])
    print(f"  => NGUONG LECH HAI MAT = {nguong_lech:.3f}")
    luu_nguong_phu({"nguong_lech_mat": nguong_lech, "phan_vi": P_LECH, "thong_ke": tk})

    # ---------- Lượt 2: áp quy tắc, cửa sổ trượt, ghi CSV ----------
    print("\nLuot 2/2: ap quy tac, tao cua so, ghi CSV...\n")
    co_dai = any(v[1] in NHAN_DAI and NHAN_DAI[v[1]] == 1 for v in cac_video)
    dem_nhan, dem_nguoi, dem_dk, dem_dai = Counter(), Counter(), Counter(), Counter()
    ti_le_dui = {"dui": [], "khac": []}
    f_ngan = open(FILE_CSV, "w", newline="", encoding="utf-8")
    w_ngan = csv.writer(f_ngan)
    w_ngan.writerow(["subject", "video", "label", "t_start", "dieu_kien"] + TEN_DAC_TRUNG)
    f_dai = w_dai = None
    if co_dai:
        f_dai = open(FILE_CSV_DAI, "w", newline="", encoding="utf-8")
        w_dai = csv.writer(f_dai)
        w_dai.writerow(["subject", "video", "label", "t_start", "dieu_kien"] + TEN_DAC_TRUNG_DAI)

    for p, ten_thu_muc, nguoi, dk, tho in cac_video:
        chuoi, ly_do = ap_dung_quy_tac(tho, nguong_lech)
        dem_ly_do = Counter(ly_do)
        n = len(ly_do)
        co_mat = 1 - dem_ly_do["khong_thay_mat"] / n if n else 0
        ti_le_dui["dui" if dk == "dui" else "khac"].append(dem_ly_do["dui"] / max(n, 1))

        so_ngan = so_dai = 0
        if ten_thu_muc in NHAN:
            for s, vec in tao_cac_cua_so(chuoi):
                w_ngan.writerow([nguoi, p.name, NHAN[ten_thu_muc], round(s / TAN_SO_MAU, 2), dk]
                                + [f"{x:.5f}" for x in vec])
                so_ngan += 1
            dem_nhan[NHAN[ten_thu_muc]] += so_ngan
            dem_nguoi[nguoi] += so_ngan
            dem_dk[dk] += so_ngan
        if w_dai is not None and ten_thu_muc in NHAN_DAI:
            dai = np.column_stack([chuoi, tho[:, 8:10]])           # ear, mar, pitch, yaw, roll, gx, gy
            dai[np.isnan(chuoi).any(axis=1), 5:] = np.nan
            for s, vec in tao_cac_cua_so_dai(dai):
                w_dai.writerow([nguoi, p.name, NHAN_DAI[ten_thu_muc], round(s / TAN_SO_MAU, 2), dk]
                               + [f"{x:.5f}" for x in vec])
                so_dai += 1
            dem_dai[NHAN_DAI[ten_thu_muc]] += so_dai
        f_ngan.flush()
        print(f"{ten_thu_muc}/{p.name:20s} thay mat {co_mat:4.0%} | dui {dem_ly_do['dui'] / max(n, 1):4.0%} "
              f"| loa {dem_ly_do['loa'] / max(n, 1):4.0%} "
              f"| ko do duoc {dem_ly_do['khong_do_duoc'] / max(n, 1):4.0%} "
              f"-> {so_ngan} cua so 2s, {so_dai} cua so 10s")

    f_ngan.close()
    if f_dai:
        f_dai.close()

    print("\n===== TONG KET =====")
    print("So cua so 2s theo nhan (0=tinh tao, 1=buon ngu):", dict(dem_nhan))
    print("So cua so 2s theo nguoi:", dict(dem_nguoi))
    print("So cua so 2s theo dieu kien:", dict(dem_dk))
    print("So nguoi:", len(dem_nguoi), "| Da luu:", FILE_CSV, "| nguong_phu.json")
    if co_dai:
        print("So cua so 10s (0=tinh tao, 1=ngu mat mo):", dict(dem_dai), "| Da luu:", FILE_CSV_DAI)
        if not any(v[3] == "lai" for v in cac_video):
            print("LUU Y: chua co video dieu kien 'lai' (nhin thang tap trung nhu dang lai). Thieu no, bo "
                  "phat hien giac ngu trang se bao nham khi tai xe tinh nhung nhin thang it chop mat.")
    else:
        print("Chua co video videos/ngu_mat_mo -> khong tao du_lieu_dai.csv (giac ngu trang).")
    if ti_le_dui["dui"]:
        print(f"Kiem chung dui mat: video 'dui' co {np.mean(ti_le_dui['dui']):.0%} mau bi nhan la dui, "
              f"video thuong chi {np.mean(ti_le_dui['khac']) if ti_le_dui['khac'] else 0:.0%}. "
              "Neu hai so gan nhau thi quy tac chua tach duoc dui mat.")
    else:
        print("Chua co video 'dui' (S01_dui_1.mp4 trong videos/tinh_tao) de kiem chung quy tac dui mat.")
    if len(dem_nguoi) < 4:
        print("LUU Y: it hon 4 nguoi thi kho chia train/test theo doi tuong o Buoc 5.")


if __name__ == "__main__":
    main()
