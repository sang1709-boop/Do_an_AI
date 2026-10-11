"""
quay_video.py - Quay video huấn luyện trực tiếp từ camera (iPhone / webcam)
và tự lưu đúng thư mục + đúng tên cho Bước 4.

Dùng camera đã chọn trong camera_cfg.py.
Chạy:  python quay_video.py   rồi trả lời 3 câu hỏi.
Bấm q trong lúc quay để hủy clip đó.
"""

import time
from pathlib import Path

import cv2

from camera_cfg import mo_camera
from dac_trung import TAN_SO_MAU

HUONG_DAN = {
    "tinh_tao": "Nhin thang, chop mat binh thuong, co the noi chuyen",
    "buon_ngu": "Nham mat lau 1-2 giay, ngap, gat gu nhu buon ngu",
}


def quay(ma, ten_nhan, giay, dem_nguoc=5):
    thu_muc = Path("videos") / ten_nhan
    thu_muc.mkdir(parents=True, exist_ok=True)
    n = 1
    while (thu_muc / f"{ma}_{n}.mp4").exists():
        n += 1
    duong_dan = thu_muc / f"{ma}_{n}.mp4"

    cap = mo_camera()
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Khong doc duoc khung hinh tu camera.")
    h, w = frame.shape[:2]

    # Ghi video ở đúng TAN_SO_MAU (15 khung/giây), bất kể camera chạy nhanh hay chậm,
    # để dòng thời gian trong video luôn chính xác.
    out = cv2.VideoWriter(str(duong_dan), cv2.VideoWriter_fourcc(*"mp4v"),
                          TAN_SO_MAU, (w, h))
    t0 = time.time()
    tiep = dem_nguoc            # thời điểm cần ghi khung kế tiếp (theo lịch cố định)
    so_khung = 0
    bi_huy = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Mat tin hieu camera, dung quay.")
            break
        t = time.time() - t0
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
            while t >= tiep:
                out.write(frame)
                so_khung += 1
                tiep += 1.0 / TAN_SO_MAU

        cv2.putText(hien, dong1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mau, 2)
        cv2.putText(hien, HUONG_DAN[ten_nhan], (10, h - 15),
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
    else:
        print(f"Da luu: {duong_dan}  ({so_khung} khung, ~{so_khung / TAN_SO_MAU:.0f}s)")
    return duong_dan


if __name__ == "__main__":
    ma = input("Ma nguoi (vi du S01): ").strip().replace("_", "") or "S01"
    chon = input("Nhan: 1 = tinh tao, 2 = buon ngu : ").strip()
    ten_nhan = "buon_ngu" if chon == "2" else "tinh_tao"
    g = input("So giay quay (Enter = 60): ").strip()
    giay = int(g) if g.isdigit() else 60
    print(f"\nSe quay {giay}s, nguoi={ma}, nhan={ten_nhan}. Huong dan: {HUONG_DAN[ten_nhan]}")
    quay(ma, ten_nhan, giay)
