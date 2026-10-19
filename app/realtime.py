"""
BƯỚC 6: Phát hiện buồn ngủ theo thời gian thực + cảnh báo âm thanh.

Cần cùng thư mục: model.pkl, model_dai.pkl (tùy chọn, giấc ngủ trắng), face_landmarker.task, pose_landmarker_lite.task (tự tải nếu thiếu),
                  dac_trung.py, anh_sang.py, tai_xe.py, camera_cfg.py, nguong_phu.json (nếu có)
Chạy:  python realtime.py

Phím:  q = thoát | r = xóa bộ nhớ đệm | [ ] = giảm / tăng ngưỡng cảnh báo 0.05
       n = đổi chế độ sáng (auto -> ngay -> dem) | c = học lại tư thế lái
       v = bật / tắt việc chỉ cảnh báo khi đang ở tư thế lái

Lưu ý: KHÔNG lật hình (không soi gương) để khớp với dữ liệu huấn luyện.

Có gì mới so với bản cũ:
  - Vẽ thêm viền khuôn mặt và 4 khóe mắt.
  - Ngày / đêm tự động (anh_sang.py): chuẩn hoá ánh sáng, giảm loá kính, loại mắt bị loá.
  - Dụi MỘT mắt không bị cảnh báo: mắt bị tay che / lệch hẳn mắt kia thì dùng mắt còn lại.
  - Điểm vai (MediaPipe Pose) xác định tài xế đang ở tư thế lái (tai_xe.py); chỉ cảnh báo
    khi đang lái.
  - Ngưỡng cảnh báo lấy từ model.pkl, do huan_luyen.py chọn từ đường ROC (không còn 0.5 cảm tính).
  - Giấc ngủ trắng (mắt mở nhưng không xử lý thông tin): model_dai.pkl, cửa sổ 10 giây với đặc
    trưng chớp mắt / ánh nhìn / chuyển động đầu. Có file này thì tự chạy, không có thì bỏ qua.
"""
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file

import threading
import time
from collections import Counter, deque

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

from app.anh_sang import NGUONG_LOA_MAT, BoXuLyAnh
from app.camera_cfg import mo_camera
from app.dac_trung import (CUA_SO, CUA_SO_DAI, KHOE_MAT, MAT_PHAI, MAT_TRAI, MIENG_DOC,
                       MIENG_NGANG, RONG_MAT, TAN_SO_MAU, VIEN_MAT, doc_nguong_lech,
                       ear_hai_mat, hang_mau, huong_nhin, lay_toa_do, tinh_mar,
                       tinh_tu_the_dau, vector_cua_so, vector_cua_so_dai)
from app.tai_xe import (CO_TAY, FILE_POSE, VAI_PHAI, VAI_TRAI, TheoDoiTuTheLai, chon_tai_xe,
                    so_tay_duoi_vai, tao_pose_detector, tay_gan_mat)

# ================= CẤU HÌNH =================
FILE_MODEL = str(model_file("model.pkl"))
FILE_MODEL_DAI = str(model_file("model_dai.pkl"))    # bộ phát hiện giấc ngủ trắng (huan_luyen_dai.py)
DU_DOAN_DAI_MOI = 15        # cứ 15 mẫu (1 giây) dự đoán giấc ngủ trắng một lần
SO_LAN_DAI = 2              # cần 2 lần liên tiếp vượt ngưỡng
CHIEU_RONG_TOI_DA = 640     # giống Bước 4
DU_DOAN_MOI = 5             # cứ 5 mẫu (~0.33 giây) dự đoán lại một lần
SO_LAN_LIEN_TIEP = 3        # cần 3 lần dự đoán liên tiếp vượt ngưỡng mới báo động
DUNG_POSE = True            # False = bỏ qua điểm vai / tay (chạy nhẹ hơn)
POSE_MOI = 2                # cứ 2 mẫu chạy Pose một lần (~7.5 lần/giây)
CHI_CANH_BAO_KHI_LAI = False # mặc định KHÔNG tắt cảnh báo do nhận dạng sai tư thế (phím v để bật)
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
        base_options=mp_python.BaseOptions(model_asset_path=str(model_file("face_landmarker.task"))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _xy(lms, chi_so):
    return [(lms[i].x, lms[i].y) for i in chi_so]


def ve_diem(frame, diem, w, h, ban_kinh, mau):
    for x, y in diem:
        cv2.circle(frame, (int(x * w), int(y * h)), ban_kinh, mau, -1)


def main():
    goi = joblib.load(FILE_MODEL)
    mo_hinh, nguong = goi["pipeline"], goi["nguong"]
    try:
        goi_dai = joblib.load(FILE_MODEL_DAI)
        mo_hinh_dai, nguong_dai = goi_dai["pipeline"], goi_dai["nguong"]
        print(f"Giac ngu trang: bat, nguong = {nguong_dai:.3f}")
    except FileNotFoundError:
        mo_hinh_dai, nguong_dai = None, None
        print(f"Giac ngu trang: tat (chua co {FILE_MODEL_DAI}, xem huan_luyen_dai.py)")
    nguong_lech = doc_nguong_lech()
    detector = tao_detector()
    pose_det = tao_pose_detector(FILE_POSE) if DUNG_POSE else None
    bo_anh = BoXuLyAnh("auto")
    tu_the = TheoDoiTuTheLai()
    chan_theo_tu_the = CHI_CANH_BAO_KHI_LAI and pose_det is not None
    cap = mo_camera()

    dem = deque(maxlen=CUA_SO)                 # cửa sổ trượt các mẫu gần nhất
    gan_day = deque(maxlen=SO_LAN_LIEN_TIEP)   # các xác suất gần nhất
    dem_dai = deque(maxlen=CUA_SO_DAI)         # cửa sổ 10 giây cho giấc ngủ trắng
    gan_dai = deque(maxlen=SO_LAN_DAI)
    mau_dai_tu_lan_cuoi = 0
    xac_suat_dai, bao_dong_dai = None, False
    dt = 1.0 / TAN_SO_MAU
    t0 = time.monotonic()
    tiep = 0.0                                 # thời điểm lấy mẫu kế tiếp
    ts_cuoi = -1
    mau_tu_lan_cuoi = 0
    so_mau = 0

    xac_suat, bao_dong, thay_mat = None, False, False
    ear = mar = None
    ly_do_mat = "ok"                           # ok | dui | che | loa | khong_do_duoc | khong_thay_mat
    dem_ly_do = Counter()
    ve_mat = None                              # điểm mặt để vẽ (toạ độ 0..1)
    ve_vai = None                              # điểm vai / cổ tay để vẽ
    pose_tx = None
    thoi_gian_xl = []                          # ms xử lý mỗi mẫu (cho báo cáo)
    fps_hien, t_truoc = 0.0, time.monotonic()
    so_khung = 0

    print(f"Dang chay. Nguong canh bao = {nguong:.3f}. Bam q de thoat.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Khong doc duoc khung hinh.")
            break
        so_khung += 1
        t = time.monotonic() - t0
        h, w = frame.shape[:2]

        # ---- Lấy mẫu đúng nhịp TAN_SO_MAU (giống lúc huấn luyện) ----
        if t >= tiep:
            bd = time.perf_counter()
            xl = frame
            if w > CHIEU_RONG_TOI_DA:
                xl = cv2.resize(frame, (CHIEU_RONG_TOI_DA, int(h * CHIEU_RONG_TOI_DA / w)))
            hx, wx = xl.shape[:2]
            anh = bo_anh.xu_ly(xl)                       # chuẩn hoá sáng ngày/đêm + giảm loá
            rgb = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_cuoi = max(ts_cuoi + 1, int(t * 1000))
            kq = detector.detect_for_video(mp_image, ts_cuoi)

            pts = lms = None
            if kq.face_landmarks:
                lms = kq.face_landmarks[0]
                pts = lay_toa_do(lms, wx, hx)

            # ---- Điểm vai: xác định tài xế đang ở tư thế lái ----
            if pose_det is not None and so_mau % POSE_MOI == 0:
                kq_pose = pose_det.detect_for_video(mp_image, ts_cuoi)
                tam_mat = pts[1] if pts is not None else None        # điểm 1 = đầu mũi
                pose_tx = (chon_tai_xe(kq_pose.pose_landmarks, wx, hx, tam_mat)
                           if kq_pose.pose_landmarks else None)
                tu_the.cap_nhat(pose_tx, wx, hx, t)
                ve_vai = (None if pose_tx is None else
                          [(pose_tx[i].x, pose_tx[i].y) for i in (VAI_TRAI, VAI_PHAI)
                           if pose_tx[i].visibility >= 0.5] +
                          [(pose_tx[i].x, pose_tx[i].y) for i in CO_TAY
                           if pose_tx[i].visibility >= 0.3])
            so_mau += 1

            # ---- Đặc trưng một mẫu (cùng hàm hang_mau với lúc huấn luyện) ----
            if pts is not None and kq.facial_transformation_matrixes:
                ear_p, ear_t = ear_hai_mat(pts)
                mar_v = tinh_mar(pts)
                pitch, yaw, roll = tinh_tu_the_dau(kq.facial_transformation_matrixes[0])
                loa_p = bo_anh.ti_le_loa(pts, MAT_PHAI) >= NGUONG_LOA_MAT
                loa_t = bo_anh.ti_le_loa(pts, MAT_TRAI) >= NGUONG_LOA_MAT
                che_p, che_t = tay_gan_mat(pose_tx, pts, MAT_PHAI, MAT_TRAI, RONG_MAT, wx, hx)
                hang, ly_do_mat = hang_mau(ear_p, ear_t, mar_v, pitch, yaw, roll,
                                           loa_p, loa_t, che_p, che_t, nguong_lech)
                ear = None if np.isnan(hang[0]) else hang[0]
                mar = None if ear is None else hang[1]
                gx, gy = huong_nhin(pts)
                hang_dai = [np.nan] * 7 if ear is None else hang + [gx, gy]
                ve_mat = {"mat_mieng": _xy(lms, DIEM_VE), "vien": _xy(lms, VIEN_MAT),
                          "khoe": _xy(lms, KHOE_MAT)}
            else:
                hang = [np.nan] * 5
                ly_do_mat = "khong_thay_mat"
                ear = mar = None
                ve_mat = None
                hang_dai = [np.nan] * 7
            dem_ly_do[ly_do_mat] += 1

            # Bỏ qua slot đã lỡ bằng NaN: KHÔNG lặp số đo mắt/đầu cũ.
            # Nhờ vậy chuyển động và tỉ lệ chớp mắt không bị giả tạo khi giật lag.
            so_slot = int((t - tiep) / dt) + 1
            if so_slot > TAN_SO_MAU:
                so_slot = TAN_SO_MAU
                tiep = t + dt
            else:
                tiep += so_slot * dt
            for _ in range(so_slot - 1):
                dem.append([np.nan] * 5)
                dem_dai.append([np.nan] * 7)
            dem.append(hang)
            dem_dai.append(hang_dai)
            mau_tu_lan_cuoi += so_slot
            mau_dai_tu_lan_cuoi += so_slot

            # ---- Dự đoán ----
            if len(dem) == CUA_SO and mau_tu_lan_cuoi >= DU_DOAN_MOI:
                mau_tu_lan_cuoi = 0
                vec = vector_cua_so(np.array(dem))
                if vec is None:                       # thiếu mặt / mắt không đo được quá nhiều
                    thay_mat, xac_suat, bao_dong = False, None, False
                    gan_day.clear()
                else:
                    thay_mat = True
                    xac_suat = float(mo_hinh.predict_proba(vec.reshape(1, -1))[0, 1])
                    gan_day.append(xac_suat)
                    bao_dong = (len(gan_day) == SO_LAN_LIEN_TIEP
                                and all(p >= nguong for p in gan_day))

            # ---- Giấc ngủ trắng (cửa sổ dài) ----
            if (mo_hinh_dai is not None and len(dem_dai) == CUA_SO_DAI
                    and mau_dai_tu_lan_cuoi >= DU_DOAN_DAI_MOI):
                mau_dai_tu_lan_cuoi = 0
                vec_dai = vector_cua_so_dai(np.array(dem_dai))
                if vec_dai is None:
                    xac_suat_dai, bao_dong_dai = None, False
                    gan_dai.clear()
                else:
                    xac_suat_dai = float(mo_hinh_dai.predict_proba(vec_dai.reshape(1, -1))[0, 1])
                    gan_dai.append(xac_suat_dai)
                    bao_dong_dai = (len(gan_dai) == SO_LAN_DAI
                                    and all(q >= nguong_dai for q in gan_dai))
            thoi_gian_xl.append((time.perf_counter() - bd) * 1000)

        # ---- Chỉ cảnh báo khi tài xế đang ở tư thế lái ----
        tam_dung = chan_theo_tu_the and not tu_the.dang_lai
        if tam_dung:
            bao_dong = bao_dong_dai = False
            gan_day.clear()
            gan_dai.clear()

        if bao_dong or bao_dong_dai:
            keu_canh_bao()

        # ---- Vẽ giao diện ----
        if ve_mat:
            ve_diem(frame, ve_mat["vien"], w, h, 1, (255, 200, 0))        # viền khuôn mặt
            ve_diem(frame, ve_mat["mat_mieng"], w, h, 2, (0, 255, 0))     # mắt + miệng
            ve_diem(frame, ve_mat["khoe"], w, h, 4, (0, 255, 255))        # 4 khóe mắt
        if ve_vai:
            pv = [(int(x * w), int(y * h)) for x, y in ve_vai]
            for q in pv:
                cv2.circle(frame, q, 6, (255, 0, 255), -1)
            if len(pv) >= 2:
                cv2.line(frame, pv[0], pv[1], (255, 0, 255), 2)           # đường nối hai vai

        if bao_dong:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "CANH BAO: BUON NGU!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        elif bao_dong_dai:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 100, 255), 12)
            cv2.putText(frame, "CANH BAO: MAT MO - NHIN TRAN TRAN", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 3)
        elif tam_dung:
            cv2.putText(frame, "Khong o tu the lai - tam dung", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        elif xac_suat is None:
            if len(dem) < CUA_SO:
                msg = "Dang thu thap..."
            elif ly_do_mat == "khong_do_duoc":
                msg = "Mat bi loa / bi che - khong do duoc"
            else:
                msg = "Khong thay khuon mat"
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
        # trạng thái ánh sáng / mắt / tư thế
        ten_che_do = f"{bo_anh.che_do.upper()}" + (" (auto)" if bo_anh.che_do_chon == "auto" else " (ep)")
        sang = 0 if bo_anh.do_sang is None else bo_anh.do_sang
        cv2.putText(frame, f"Anh sang: {ten_che_do}  do sang {sang:.0f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if mo_hinh_dai is not None:
            ts = "--" if xac_suat_dai is None else f"{xac_suat_dai:.2f}"
            cv2.putText(frame, f"Giac ngu trang (10s): P={ts} nguong={nguong_dai:.2f}", (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        if ly_do_mat in ("dui", "che", "loa"):
            ten = {"dui": "dui mat 1 ben - dung mat con lai",
                   "che": "tay che 1 mat - dung mat con lai",
                   "loa": "kinh loa 1 mat - dung mat con lai"}[ly_do_mat]
            cv2.putText(frame, "Mat: " + ten, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        if pose_det is not None:
            thay, duoi = so_tay_duoi_vai(pose_tx)
            gate = "" if chan_theo_tu_the else " [khong chan]"
            cv2.putText(frame, f"Tu the: {tu_the.ly_do}  tay duoi vai {duoi}/{thay}{gate}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        now = time.monotonic()
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
            dem_dai.clear(); gan_dai.clear(); xac_suat_dai, bao_dong_dai = None, False
            mau_tu_lan_cuoi = mau_dai_tu_lan_cuoi = 0
        elif phim == ord("["):
            nguong = max(0.05, nguong - 0.05)
        elif phim == ord("]"):
            nguong = min(0.95, nguong + 0.05)
        elif phim == ord("n"):
            print("Che do sang:", bo_anh.doi_che_do())
        elif phim == ord("c"):
            tu_the.dat_lai()
            print("Hoc lai tu the lai: ngoi dung tu the lai trong vai giay.")
        elif phim == ord("v") and pose_det is not None:
            chan_theo_tu_the = not chan_theo_tu_the
            print("Chi canh bao khi dang lai:", "BAT" if chan_theo_tu_the else "TAT")

    cap.release()
    detector.close()
    if pose_det is not None:
        pose_det.close()
    cv2.destroyAllWindows()

    # ---- Tổng kết hiệu năng (dùng cho báo cáo) ----
    tong = time.monotonic() - t0
    if thoi_gian_xl:
        a = np.array(thoi_gian_xl)
        tk = [
            f"Thoi gian chay           : {tong:.1f} s",
            f"So mau da xu ly          : {len(a)}",
            f"FPS hien thi trung binh  : {so_khung / tong:.1f}",
            f"Xu ly 1 mau (trung binh) : {a.mean():.1f} ms",
            f"Xu ly 1 mau (95%)        : {np.percentile(a, 95):.1f} ms",
            f"Do tre cua so            : {CUA_SO / TAN_SO_MAU:.1f} s (do cua so 30 mau)",
            f"Nguong canh bao          : {nguong:.3f}",
            f"Nguong giac ngu trang    : {'tat' if nguong_dai is None else f'{nguong_dai:.3f}'}",
            "Mau theo tinh trang mat  : " + ", ".join(f"{k}={v}" for k, v in sorted(dem_ly_do.items())),
        ]
        print("\n=== TONG KET HIEU NANG ===")
        print("\n".join(tk))
        with open(report_file("ket_qua_realtime.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(tk))


if __name__ == "__main__":
    main()
