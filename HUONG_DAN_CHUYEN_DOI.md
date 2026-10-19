# Di chuyển bản dự án cũ sang cấu trúc GitHub-ready

1. Sao lưu `D:\do_an_buon_ngu` thành thư mục khác. Không xóa `venv/`, `videos/`, `backup/`.
2. Chép **nội dung** ZIP này vào thư mục dự án sau khi sao lưu, ghi đè các entrypoint Python cũ. Code chính hiện ở `app/`.
3. Di chuyển các tệp hiện có:
   - `model.pkl`, `model_dai.pkl`, `face_landmarker.task`, `pose_landmarker_lite.task` → `models/`
   - `du_lieu_dac_trung.csv`, `du_lieu_dai.csv`, `nguong_phu.json` → `data/`
   - `ma_tran_nham_lan.png`, `do_quan_trong_dac_trung.png`, `nguong_vs_chi_so.png`, `roc_nguong.png`, `ket_qua_*.txt` → `reports/`
   - Giữ nguyên `videos/` và `venv/` tại gốc; KHÔNG commit lên Git.
4. Kiểm thử: `python -m compileall -q app`, `python camera_cfg.py`, `python realtime.py`. Chạy huấn luyện lại nếu đã thay xử lý mẫu, và đo lại F1/Recall.
5. Git: xem hướng dẫn trong README hoặc lệnh dưới đây.

## Git an toàn
```powershell
git status
git remote -v
git fetch origin
git pull --rebase origin main
git add .
git status --short
git commit -m "Refactor project layout and Git ignore"
git push origin main
```
**Nếu `git pull --rebase` có conflict, dừng và xử lý conflict trước, tuyệt đối không force-push.**

**Nếu Git báo file `.venv`/video >100 MB đã từng commit**: `.gitignore` KHÔNG xóa chúng khỏi lịch sử. Trước hết xác định bằng `git ls-files | Select-String '(^|/)(venv|\.venv)/|\.mp4$'`. Để loại bỏ dữ liệu đã theo dõi tại commit hiện tại, dùng `git rm -r --cached --ignore-unmatch venv .venv videos` rồi commit; nếu vẫn bị GitHub chặn do lịch sử cũ, cần quy trình dọn lịch sử riêng sau khi sao lưu và trao đổi với thành viên nhóm.
