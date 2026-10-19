"""
quay_video.py - Quay video huấn luyện trực tiếp từ camera (iPhone / webcam)
và tự lưu đúng thư mục + đúng tên cho Bước 4.

Dùng camera đã chọn trong camera_cfg.py.
Chạy:  python quay_video.py   rồi trả lời 3 câu hỏi.
Bấm q trong lúc quay để hủy clip đó.

Tên file lưu: <mã người>_<điều kiện>_<số>.mp4, ví dụ S01_dem_1.mp4
Điều kiện: ngay | dem | kinh (ngày, đeo kính) | demkinh (đêm, đeo kính) | dui (dụi một mắt)
           | lai (ngồi như đang lái, nhìn thẳng tập trung - nhãn tỉnh táo)
Nhãn 3 = ngu_mat_mo (giấc ngủ trắng): mắt mở, nhìn trân trân, gần như không chớp, đầu bất động.
Quay "dui" vào nhãn tỉnh táo: đó là mẫu để kiểm chứng việc dụi mắt KHÔNG bị cảnh báo.
"""
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file

import time
from pathlib import Path

import cv2

from app.camera_cfg import mo_camera
from app.dac_trung import TAN_SO_MAU

HUONG_DAN = {
    "tinh_tao": "Nhin thang, chop mat binh thuong, co the noi chuyen",
    "buon_ngu": "Nham mat lau 1-2 giay, ngap, gat gu nhu buon ngu",
    "ngu_mat_mo": "Mat MO nhin tran tran vao mot diem, khong chop (hoac rat cham), dau dung yen / gat rat nhe",
}
HUONG_DAN_LAI = "Ngoi nhu dang lai, nhin thang vao duong, tap trung. Chop mat tu nhien, thinh thoang liec guong"
HUONG_DAN_DUI = "Dui MOT mat bang tay (khoang 3-5 giay), nghi, roi dui lai. Mat con lai mo binh thuong"


DIEU_KIEN = {"1": "ngay", "2": "dem", "3": "kinh", "4": "demkinh", "5": "dui", "6": "lai"}


def quay(ma, ten_nhan, giay, dieu_kien="ngay", dem_nguoc=5):
    thu_muc = VIDEOS / ten_nhan
    thu_muc.mkdir(parents=True, exist_ok=True)
    n = 1
    while (thu_muc / f"{ma}_{dieu_kien}_{n}.mp4").exists():
        n += 1
    duong_dan = thu_muc / f"{ma}_{dieu_kien}_{n}.mp4"

    cap = mo_camera()
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Khong doc duoc khung hinh tu camera.")
    h, w = frame.shape[:2]

    # Ghi video ở đúng TAN_SO_MAU (15 khung/giây), bất kể camera chạy nhanh hay chậm,
    # để dòng thời gian trong video luôn chính xác.
    out = cv2.VideoWriter(str(duong_dan), cv2.VideoWriter_fourcc(*"mp4v"),
                          TAN_SO_MAU, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Khong tao duoc file video: {duong_dan}")
    t0 = time.monotonic()
    tiep = dem_nguoc            # thời điểm cần ghi khung kế tiếp (theo lịch cố định)
    so_khung = 0
    bi_huy = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Mat tin hieu camera, dung quay.")
            break
        t = time.monotonic() - t0
        hien = cv2.flip(frame, 1)            # chỉ lật để hiển thị, KHÔNG lật khi lưu

        if t < dem_nguoc:
            dong1 = f"Chuan bi... {int(dem_nguoc - t) + 1}"
            mau = (0, 255, 255)
        else:
            tg = t - dem_nguoc
            if tg >= giay:
                break
            dong1 = f"DANG QUAY - con {int(giay - tg) + 1}s"
            mau = (0, 0, 255)
            # Mỗi 1/15 giây ghi đúng 1 khung. Nếu camera chậm hơn thì lặp lại khung
            # gần nhất, nhờ vậy 1 giây trong video luôn đúng 1 giây ngoài đời.
            # Khung lặp khi camera chậm có thể tạo chớp mắt giả; chỉ ghi khung thật.
            if t >= tiep:
                out.write(frame)
                so_khung += 1
                tiep += 1.0 / TAN_SO_MAU
                if t - tiep > 1.0 / TAN_SO_MAU:
                    tiep = t + 1.0 / TAN_SO_MAU

        cv2.putText(hien, dong1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mau, 2)
        cv2.putText(hien, HUONG_DAN_DUI if dieu_kien == "dui" else
                    HUONG_DAN_LAI if dieu_kien == "lai" else HUONG_DAN[ten_nhan], (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Quay video (bam q de huy)", hien)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            bi_huy = True
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

    if bi_huy or so_khung == 0:
        duong_dan.unlink(missing_ok=True)
        print("Da huy, khong luu clip.")
        return None
    else:
        print(f"Da luu: {duong_dan}  ({so_khung} khung, ~{so_khung / TAN_SO_MAU:.0f}s)")
    return duong_dan


if __name__ == "__main__":
    ma = input("Ma nguoi (vi du S01): ").strip().replace("_", "") or "S01"
    chon = input("Nhan: 1 = tinh tao, 2 = buon ngu (nham mat), 3 = ngu mat mo (giac ngu trang) : ").strip()
    ten_nhan = {"2": "buon_ngu", "3": "ngu_mat_mo"}.get(chon, "tinh_tao")
    print("Dieu kien: 1 = ngay | 2 = dem | 3 = kinh (ngay) | 4 = dem + kinh | 5 = dui mot mat | "
          "6 = lai (nhin thang tap trung)")
    dk = DIEU_KIEN.get(input("Dieu kien (Enter = 1): ").strip() or "1", "ngay")
    if dk in ("dui", "lai") and ten_nhan != "tinh_tao":
        print(f"Video '{dk}' la mau KHONG buon ngu -> tu dong chuyen sang nhan tinh_tao.")
        ten_nhan = "tinh_tao"
    g = input("So giay quay (Enter = 60): ").strip()
    giay = int(g) if g.isdigit() else 60
    huong_dan = (HUONG_DAN_DUI if dk == "dui" else HUONG_DAN_LAI if dk == "lai" else HUONG_DAN[ten_nhan])
    print(f"\nSe quay {giay}s, nguoi={ma}, nhan={ten_nhan}, dieu kien={dk}. Huong dan: {huong_dan}")
    quay(ma, ten_nhan, giay, dk)
