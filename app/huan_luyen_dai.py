"""
Huấn luyện bộ phát hiện GIẤC NGỦ TRẮNG (mắt mở nhưng não không xử lý thông tin).

Cần: du_lieu_dai.csv (từ tao_du_lieu.py, có thư mục videos/ngu_mat_mo), dac_trung.py, huan_luyen.py
Chạy:  python huan_luyen_dai.py
Kết quả: model_dai.pkl, ket_qua_dai.txt

Bộ này tách riêng khỏi model.pkl vì bài toán khác hẳn: cửa sổ 10 giây và đặc trưng HÀNH VI
(chớp mắt, ánh nhìn, chuyển động đầu) thay vì độ nhắm mắt. Cách đánh giá và chọn ngưỡng y hệt
huan_luyen.py: Leave-One-Subject-Out, ngưỡng chọn từ đường ROC (Youden) + bootstrap theo người.

Hai điều cần nhớ khi đọc kết quả:
  - Nhãn 1 là video DIỄN LẠI (ngồi nhìn trân trân, không chớp). Giấc ngủ trắng thật không quay
    được an toàn, nên mô hình học dấu hiệu bên ngoài của nó, không phải trạng thái não.
  - Hãy nhìn bảng theo điều kiện: dòng 'lai' (tỉnh táo nhưng nhìn thẳng tập trung) có nhiều
    báo nhầm là dấu hiệu bộ này chưa phân biệt được "tập trung" với "ngủ mắt mở".
"""
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

import app.huan_luyen as hl
from app.dac_trung import TEN_DAC_TRUNG_DAI

FILE_CSV_DAI = str(data_file("du_lieu_dai.csv"))
FILE_MODEL_DAI = str(model_file("model_dai.pkl"))
FILE_BAO_CAO_DAI = str(report_file("ket_qua_dai.txt"))


def main():
    in_ra = hl.in_ra
    df = pd.read_csv(FILE_CSV_DAI).replace([np.inf, -np.inf], np.nan).dropna(subset=TEN_DAC_TRUNG_DAI)
    X = df[TEN_DAC_TRUNG_DAI].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    g = df["subject"].astype(str).to_numpy()
    nguoi = sorted(set(g))

    in_ra("=" * 70)
    in_ra(f"DU LIEU (cua so 10s): {len(df)} cua so | {len(nguoi)} nguoi {nguoi}")
    in_ra(f"Tinh tao (0): {(y == 0).sum()} | Ngu mat mo (1): {(y == 1).sum()}")
    if len(nguoi) < 3 or len(set(y)) < 2:
        raise SystemExit("Can it nhat 3 nguoi va du ca 2 nhan (tinh_tao, ngu_mat_mo).")
    thieu = [n for n in nguoi if len(set(y[g == n])) < 2]
    if thieu:
        in_ra(f"CANH BAO: nguoi {thieu} chi co 1 nhan; nen quay du ca tinh_tao va ngu_mat_mo cho moi nguoi.")

    # ---------- So sánh với luật đơn giản "bao lâu không chớp mắt" ----------
    cot_gap = TEN_DAC_TRUNG_DAI.index("khoang_khong_chop_max")
    mo_hinh = {
        "Luat don: chi khoang khong chop": lambda: make_pipeline(
            FunctionTransformer(hl.chon_cot, kw_args={"cot": cot_gap}),
            DecisionTreeClassifier(max_depth=1, random_state=hl.SEED)),
        "Random Forest (tinh chinh)": hl.tao_rf,
    }
    in_ra("\n" + "=" * 70)
    in_ra("DANH GIA LEAVE-ONE-SUBJECT-OUT (ngu mat mo la lop duong), nguong Youden rieng tung mo hinh")
    in_ra(f"{'Mo hinh':36s} {'AUC':>6s} {'Nguong':>7s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")
    kq = {}
    for ten, tao in mo_hinh.items():
        print(f"  ...dang chay: {ten}")
        kq[ten] = hl.chay_loso(tao, X, y, g)
        yt, pt = kq[ten][0], kq[ten][1]
        t = hl.chon_nguong(yt, pt, "youden")
        m = hl.chi_so(yt, pt, t)
        in_ra(f"{ten:36s} {roc_auc_score(yt, pt):6.3f} {t:7.3f} {m['prec']:6.3f} {m['rec']:7.3f} "
              f"{m['f1']:6.3f} {m['fp']:4d} {m['fn']:4d}")

    y_that, p, ng, chi_muc = kq["Random Forest (tinh chinh)"]

    # ---------- Chọn ngưỡng ----------
    in_ra("\n" + "=" * 70)
    in_ra("CHON NGUONG CANH BAO (Random Forest)")
    t_chon = hl.chon_nguong(y_that, p, hl.PHUONG_PHAP_NGUONG)
    in_ra(f"=> NGUONG ({hl.PHUONG_PHAP_NGUONG}): {t_chon:.3f}")
    ts = hl.nguong_bootstrap(y_that, p, ng)
    if len(ts) >= 10:
        lo, med, hi = np.percentile(ts, [2.5, 50, 97.5])
        in_ra(f"Do on dinh (bootstrap {len(ts)} lan theo NGUOI): trung vi {med:.3f}, khoang 95% [{lo:.3f}; {hi:.3f}]")
    else:
        lo = med = hi = float("nan")
    d_kq, _ = hl.danh_gia_khach_quan(y_that, p, ng)
    mk = hl.chi_so_nhan(y_that, d_kq)
    in_ra("UOC LUONG KHACH QUAN (nguong moi nguoi chon CHI tu nhung nguoi con lai): "
          f"Prec {mk['prec']:.3f} | Recall {mk['rec']:.3f} | F1 {mk['f1']:.3f} | FP {mk['fp']} | FN {mk['fn']}")

    m = hl.chi_so(y_that, p, t_chon)
    in_ra("\nMA TRAN NHAM LAN tai nguong da chon")
    in_ra(f"  Thuc te Tinh tao   {m['tn']:5d}  {m['fp']:5d}   (FP = canh bao nham)")
    in_ra(f"  Thuc te Ngu mat mo {m['fn']:5d}  {m['tp']:5d}   (FN = bo sot)")

    in_ra("\nKET QUA THEO TUNG NGUOI (nguoi chua tung thay luc huan luyen)")
    in_ra(f"{'Nguoi':>8s} {'So cua so':>10s} {'Acc':>6s} {'Recall':>7s}")
    for n in sorted(set(ng)):
        k = ng == n
        mn = hl.chi_so(y_that[k], p[k], t_chon)
        in_ra(f"{n:>8s} {k.sum():10d} {mn['acc']:6.3f} {mn['rec']:7.3f}")

    if "dieu_kien" in df.columns:
        dk = df["dieu_kien"].astype(str).to_numpy()[chi_muc]
        in_ra("\nKET QUA THEO DIEU KIEN QUAY")
        in_ra(f"{'Dieu kien':>10s} {'So cua so':>10s} {'Acc':>6s} {'Bao nham(FP)':>13s} {'Bo sot(FN)':>11s}")
        for c in sorted(set(dk)):
            k = dk == c
            mc = hl.chi_so(y_that[k], p[k], t_chon)
            in_ra(f"{c:>10s} {k.sum():10d} {mc['acc']:6.3f} {mc['fp']:13d} {mc['fn']:11d}")
        if "lai" not in set(dk):
            in_ra("   CANH BAO: khong co video dieu kien 'lai' (tinh tao nhung nhin thang tap trung). "
                  "Khong the do muc bao nham quan trong nhat cua bo nay.")

    # ---------- Mô hình cuối ----------
    in_ra("\n" + "=" * 70)
    cuoi = hl.tao_rf()
    cuoi.fit(X, y, groups=g)
    in_ra("Sieu tham so tot nhat:", cuoi.best_params_)
    tot = cuoi.best_estimator_
    joblib.dump({"pipeline": tot, "ten_dac_trung": TEN_DAC_TRUNG_DAI, "nguong": float(t_chon),
                 "nguong_info": {"phuong_phap": hl.PHUONG_PHAP_NGUONG,
                                 "auc_loso": float(roc_auc_score(y_that, p)),
                                 "khoang_95": [float(lo), float(hi)], "so_nguoi": len(nguoi)}},
                FILE_MODEL_DAI)
    in_ra(f"Da luu mo hinh: {FILE_MODEL_DAI}  (nguong canh bao = {t_chon:.3f})")
    rf = tot.named_steps["randomforestclassifier"]
    thu_tu = np.argsort(rf.feature_importances_)[::-1]
    in_ra("\nTOP 8 DAC TRUNG QUAN TRONG NHAT:")
    for i in thu_tu[:8]:
        in_ra(f"   {TEN_DAC_TRUNG_DAI[i]:24s} {rf.feature_importances_[i]:.3f}")
    in_ra("\nLUU Y: nhan 1 la video dien lai; do tre phat hien >= 10 giay (do dai cua so).")
    hl.luu_bao_cao(FILE_BAO_CAO_DAI)


if __name__ == "__main__":
    main()
