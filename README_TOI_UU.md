# Bản tối ưu 10 file — hướng dẫn Windows / VS Code

**Sao lưu dự án trước khi thay thế.** Giữ nguyên các file model.pkl, model_dai.pkl, *.task, CSV và thư mục videos đang có.

1. Giải nén ZIP và chép 10 file `.py` vào thư mục gốc dự án. Không xóa dữ liệu cũ.
2. Mở PowerShell tại thư mục dự án và bật môi trường: `./venv/Scripts/Activate.ps1` (hoặc `.venv`).
3. Cài thư viện theo `python -m pip install -r requirements.txt` của dự án gốc.
4. Chạy `python camera_cfg.py`, chọn chỉ số webcam chính xác trong camera_cfg.py.
5. Nếu đã có model.pkl: chạy ngay `python realtime.py`. Phím q thoát, r reset, [ ] chỉnh ngưỡng, n đổi chế độ ánh sáng, c học lại tư thế, v bật/tắt chặn theo tư thế.
6. Để huấn luyện lại: `python quay_video.py` => `python tao_du_lieu.py` => `python huan_luyen.py` => `python realtime.py`.
7. Nếu có video `videos/ngu_mat_mo/`: sau tạo CSV, chạy `python huan_luyen_dai.py`.
8. Chạy kiểm tra `python kiem_tra_du_lieu.py` hoặc `python huan_luyen.py phan_tich`.

## Thay đổi có ảnh hưởng cần lưu ý

- Mặc định `CHI_CANH_BAO_KHI_LAI=False` để không bỏ cảnh báo do ước lượng tư thế sai. Phím v vẫn cho phép bật lại.
- Video quay mới không lặp frame khi camera chậm; thời lượng file có thể NGẮN hơn thời gian thực nếu FPS camera thiếu. Xem kết quả trên màn hình, nên quay lại bằng camera đủ >=15 FPS.
- `tao_du_lieu.py` dùng Pose phát hiện tay che mắt **nếu có sẵn** `pose_landmarker_lite.task`; nếu thiếu file, vẫn chạy nhưng không dùng thông tin che tay.
- Realtime đánh dấu mẫu trễ là NaN (thay vì nhân bản mẫu), tránh tạo tín hiệu chớp mắt giả. Có thể xuất hiện trạng thái thiếu dữ liệu nếu camera quá chậm.
- Chưa thể xác thực F1/FPS thực tế khi chưa có camera, model và dữ liệu gốc của bạn trong môi trường này. Các mô hình dựa trên biểu hiện bên ngoài, không xác nhận trạng thái sinh lý. Không dùng như hệ thống an toàn lái xe đã được kiểm định.
- `.gitignore` chỉ ngăn thêm file mới. File lớn đã từng commit cần loại khỏi lịch sử Git riêng trước khi push.
