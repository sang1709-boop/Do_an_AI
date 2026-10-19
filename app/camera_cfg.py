"""
camera_cfg.py - Nơi DUY NHẤT cần sửa để chọn nguồn camera cho mọi bước.
(Đã gộp tim_camera.py vào đây.)

Cách dùng:
    python camera_cfg.py        -> mở camera đang chọn để test (bấm q để thoát)
    python camera_cfg.py tim    -> thử camera 0..4, lưu ảnh reports/camera_0.jpg, reports/camera_1.jpg...
                                   Mở ảnh xem camera nào là iPhone / webcam rồi điền
                                   số đó vào WEBCAM_INDEX bên dưới.
"""
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file

import sys

import cv2

# ============ CHỈ CẦN SỬA PHẦN NÀY ============
# True  = dùng iPhone qua DroidCam (mạng Wi-Fi)
# False = dùng webcam / camera cắm vào máy
DUNG_DROIDCAM = False

# Địa chỉ hiện trong app DroidCam (thường cổng 4747): http://<IP>:<cổng>/video
DROIDCAM_URL = "http://192.168.100.248:4747/video"

# Số thứ tự camera trong máy (0, 1, 2...). Chạy "python camera_cfg.py tim" để biết số nào.
WEBCAM_INDEX = 1
# ==============================================

# Giữ tên NGUON_CAMERA để các file khác (nếu có import) vẫn chạy được
NGUON_CAMERA = DROIDCAM_URL if DUNG_DROIDCAM else WEBCAM_INDEX


def mo_camera():
    """Mở camera theo NGUON_CAMERA, báo lỗi rõ ràng nếu không mở được."""
    print(f"Dang mo camera: {NGUON_CAMERA}")
    if isinstance(NGUON_CAMERA, str):
        cap = cv2.VideoCapture(NGUON_CAMERA)                 # luồng mạng (DroidCam)
    else:
        cap = cv2.VideoCapture(NGUON_CAMERA, cv2.CAP_DSHOW)  # webcam trong máy (Windows)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                      # giảm độ trễ
    if not cap.isOpened():
        cap.release()
        raise SystemExit(
            "Khong mo duoc camera. Kiem tra: (1) app DroidCam tren iPhone dang mo "
            "va man hinh khong tat, (2) iPhone va laptop cung mang Wi-Fi, "
            "(3) IP/cong trong camera_cfg.py dung voi so app dang hien, "
            "(4) neu dung webcam: dong cac ung dung khac dang dung camera."
        )
    return cap


def tim_camera(so_luong=5):
    """Thử camera 0..so_luong-1, in kết quả và lưu ảnh chụp camera_<i>.jpg."""
    for i in range(so_luong):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Camera {i}: khong mo duoc")
            continue
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            path = report_file(f"camera_{i}.jpg")
            cv2.imwrite(str(path), frame)
            print(f"Camera {i}: OK, kich thuoc {w}x{h}, da luu {path}")
        else:
            print(f"Camera {i}: mo duoc nhung khong doc duoc khung hinh")
        cap.release()
    print("\nMo cac anh camera_*.jpg de xem camera nao la camera can dung,"
          " roi dien so do vao WEBCAM_INDEX.")


def test_camera():
    """Hiển thị camera đang chọn, không dùng MediaPipe."""
    cap = mo_camera()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Khong doc duoc khung hinh.")
            break
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{w}x{h}  (bam q de thoat)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Test camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "tim":
        tim_camera()
    else:
        test_camera()