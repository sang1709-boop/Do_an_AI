# Du an AI - Vi tri luu san pham

**Khong xoa video, CSV hoac model hien co.** Ban nay cung cap ma nguon, khong kem du lieu rieng.

1. Sao luu `D:\do_an_buon_ngu`. Giai nen ZIP, sao chep `app/` va cac file `.py` khoi dong vao thu muc goc, cho phep ghi de code sau khi da backup.
2. Giu `models/*.pkl`, `models/*.task`, `data/*.csv`, `data/nguong_phu.json`, va `videos/` cu; khong can di chuyen lai neu dung vi tri.
3. Mo VS Code tai `D:\do_an_buon_ngu`, trong terminal chay:

```powershell
.\venv\Scripts\Activate.ps1
python camera_cfg.py tim
python quay_video.py
python tao_du_lieu.py
python huan_luyen.py
python realtime.py
```

Khi co du lieu phu hop: `python huan_luyen_dai.py`.

| Lenh | San pham |
|---|---|
| `python camera_cfg.py tim` | `reports/camera_0.jpg` ... |
| `python quay_video.py` | `videos/<nhan>/S01_ngay_1.mp4` ... |
| `python tao_du_lieu.py` | `data/du_lieu_dac_trung.csv`, `data/du_lieu_dai.csv` (neu co), `data/nguong_phu.json` |
| `python huan_luyen.py` | `models/model.pkl`, `reports/ma_tran_nham_lan.png`, `reports/do_quan_trong_dac_trung.png`, `reports/roc_nguong.png`, `reports/nguong_vs_chi_so.png`, `reports/ket_qua_danh_gia.txt` |
| `python huan_luyen.py phan_tich` | `reports/ket_qua_phan_tich.txt` |
| `python huan_luyen_dai.py` | `models/model_dai.pkl`, `reports/ket_qua_dai.txt` |
| `python realtime.py` | `reports/ket_qua_realtime.txt` (khi thoat binh thuong) |

**Chu y:** Code moi khong tu dong di chuyen file cu con sot o thu muc goc. Di chuyen thu cong neu can. Cac file `.pkl` va `.csv` moi co the ghi de file cu khi huan luyen; sao luu truoc.

`app/paths.py` tinh duong dan theo vi tri file, khong theo vi tri Terminal. Chay cac file khoi dong nam o thu muc goc hoac `python -m app.huan_luyen`.

Khong day cac file trong `videos`, `data`, `models`, `reports` len GitHub. `.gitignore` khong xoa du lieu da duoc Git theo doi truoc do.
