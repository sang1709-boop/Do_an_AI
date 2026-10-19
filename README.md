# AI phát hiện dấu hiệu buồn ngủ qua video khuôn mặt

Dự án sử dụng OpenCV + MediaPipe Tasks + scikit-learn. Đây là **bản nghiên cứu/đồ án**, không phải hệ thống cảnh báo an toàn đã kiểm định. Mô hình cửa sổ dài dự đoán hành vi nhìn trân trân mô phỏng, không chẩn đoán giấc ngủ sinh lý.

## Cấu trúc
- `app/`: toàn bộ 10 module gốc và `paths.py` quản lý đường dẫn chung.
- `camera_cfg.py`, `realtime.py`, `quay_video.py`, `tao_du_lieu.py`, `huan_luyen.py`, ...: entrypoint tương thích với lệnh cũ.
- `models/`: `model.pkl`, `model_dai.pkl` (nếu có), `face_landmarker.task`, `pose_landmarker_lite.task` (KHÔNG upload Git).
- `data/`: `du_lieu_dac_trung.csv`, `du_lieu_dai.csv`, `nguong_phu.json`.
- `videos/`: video theo nhãn `tinh_tao`, `buon_ngu`, `ngu_mat_mo`.
- `reports/`: báo cáo và đồ thị được chương trình xuất ra.

## Thiết lập (PowerShell, Windows)
```powershell
cd D:\do_an_buon_ngu
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python camera_cfg.py
python realtime.py
```

Nếu đã có `venv` hoạt động, **không cần tạo lại**. Cập nhật `app/camera_cfg.py` để chọn camera. Trước khi chạy realtime phải đặt model và asset vào `models/`.

## Thu thập và huấn luyện
```powershell
python quay_video.py
python tao_du_lieu.py
python huan_luyen.py
python huan_luyen_dai.py
python realtime.py
```
Lệnh `huan_luyen_dai.py` chỉ chạy khi đã có nhãn `ngu_mat_mo` và `data/du_lieu_dai.csv`. `python huan_luyen.py phan_tich` dùng để phân tích dữ liệu.

## Di chuyển từ dự án cũ
**Sao lưu toàn bộ dự án.** Giữ thư mục `venv/` và video hiện có. Xem `HUONG_DAN_CHUYEN_DOI.md` để chuyển tệp tới đúng vị trí. Kiểm tra `git status` và `git ls-files` trước khi push.
