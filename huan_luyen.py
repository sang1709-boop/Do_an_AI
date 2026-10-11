"""
BƯỚC 5: Huấn luyện + đánh giá + phân tích dữ liệu (đã gộp Bước 5b).

Cần: du_lieu_dac_trung.csv (từ Bước 4), dac_trung.py

Cách chạy:
    python buoc5_huan_luyen.py              -> huấn luyện + đánh giá + lưu model.pkl (mặc định)
    python buoc5_huan_luyen.py kiem_tra     -> kiểm tra dữ liệu có hợp lý không (nhanh)
    python buoc5_huan_luyen.py so_sanh      -> so sánh các nhóm đặc trưng (nhanh)
    python buoc5_huan_luyen.py phan_tich    -> kiem_tra + so_sanh
    python buoc5_huan_luyen.py tat_ca       -> phan_tich rồi huấn luyện

Kết quả:
    huấn luyện : model.pkl, ket_qua_danh_gia.txt, ma_tran_nham_lan.png, do_quan_trong_dac_trung.png
    phân tích  : ket_qua_phan_tich.txt

Cách đánh giá: Leave-One-Subject-Out (LOSO)
  - Lần lượt giữ 1 người làm tập TEST, huấn luyện trên những người còn lại.
  - Người trong tập test KHÔNG BAO GIỜ xuất hiện trong tập huấn luyện.
  - Tinh chỉnh siêu tham số của Random Forest cũng chỉ dùng người trong tập huấn luyện
    (nested cross-validation) nên kết quả không bị "lộ" thông tin từ người test.

Ý nghĩa phần phân tích:
  - kiem_tra: ở MỌI người, clip buồn ngủ thường có EAR thấp hơn (mắt nhắm nhiều) và
    EAR dao động nhiều hơn (std cao). Người nào làm ngược lại -> nghi ngờ nhầm nhãn
    hoặc quay khác điều kiện.
  - so_sanh: nhóm đặc trưng nào giúp mô hình tổng quát hóa tốt cho người mới.
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dac_trung import TEN_DAC_TRUNG

# ================= CẤU HÌNH =================
FILE_CSV = "du_lieu_dac_trung.csv"
FILE_MODEL = "model.pkl"
FILE_BAO_CAO_HUAN_LUYEN = "ket_qua_danh_gia.txt"
FILE_BAO_CAO_PHAN_TICH = "ket_qua_phan_tich.txt"
NGUONG_CANH_BAO = 0.5      # xác suất "buồn ngủ" từ ngưỡng này trở lên thì cảnh báo
SEED = 42

# --- Huấn luyện ---
LUOI_SIEU_THAM_SO = {
    "randomforestclassifier__n_estimators": [100, 300],
    "randomforestclassifier__max_depth": [None, 5, 10],
    "randomforestclassifier__min_samples_leaf": [1, 3],
}

# --- Phân tích: kiem_tra ---
COT_KIEM_TRA = ["ear_mean", "ear_min", "ear_std", "mar_max", "pitch_mean", "yaw_mean"]

# --- Phân tích: so_sanh ---
NHOM = {
    "A. Chi mat (EAR)": [c for c in TEN_DAC_TRUNG if c.startswith("ear_")],
    "B. Mat + mieng (EAR, MAR)": [c for c in TEN_DAC_TRUNG if c.startswith(("ear_", "mar_"))],
    "C. B + do dao dong dau (pitch/yaw/roll _std)": [
        c for c in TEN_DAC_TRUNG
        if c.startswith(("ear_", "mar_")) or c in ("pitch_std", "yaw_std", "roll_std")],
    "D. Tat ca 20 dac trung": list(TEN_DAC_TRUNG),
}
MO_HINH_NHANH = {
    "LogReg": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "RF(bao thu)": lambda: make_pipeline(
        StandardScaler(), RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED)),
}
# ============================================

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

bao_cao = []


def in_ra(*args):
    """In ra màn hình và nhớ lại để ghi vào file báo cáo."""
    dong = " ".join(str(a) for a in args)
    print(dong)
    bao_cao.append(dong)


def luu_bao_cao(ten_file):
    """Ghi báo cáo đã tích lũy ra file rồi xóa bộ nhớ (để chạy tiếp chế độ khác)."""
    with open(ten_file, "w", encoding="utf-8") as f:
        f.write("\n".join(bao_cao))
    bao_cao.clear()
    print(f"Da luu bao cao: {ten_file}")


def doc_du_lieu():
    df = pd.read_csv(FILE_CSV)
    return df.dropna(subset=TEN_DAC_TRUNG)


# =====================================================================
# PHẦN DÙNG CHUNG
# =====================================================================
def chon_cot(X, cot):
    return X[:, [cot]]


def chi_so(y, p, nguong=0.5):
    d = (p >= nguong).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y, d),
        "prec": precision_score(y, d, zero_division=0),
        "rec": recall_score(y, d, zero_division=0),
        "f1": f1_score(y, d, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# =====================================================================
# PHẦN 1: HUẤN LUYỆN + ĐÁNH GIÁ (LOSO) + LƯU MODEL
# =====================================================================
def tao_rf():
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=SEED, class_weight="balanced"),
    )
    return GridSearchCV(pipe, LUOI_SIEU_THAM_SO, cv=LeaveOneGroupOut(),
                        scoring="f1", n_jobs=-1)


def tao_cac_mo_hinh():
    cot_ear_mean = TEN_DAC_TRUNG.index("ear_mean")
    return {
        "Baseline: 1 nguong tren ear_mean": lambda: make_pipeline(
            FunctionTransformer(chon_cot, kw_args={"cot": cot_ear_mean}),
            DecisionTreeClassifier(max_depth=1, random_state=SEED)),
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "SVM (RBF)": lambda: make_pipeline(
            StandardScaler(),
            SVC(probability=True, class_weight="balanced", random_state=SEED)),
        "Random Forest (tinh chinh)": tao_rf,
    }


def chay_loso(tao_mo_hinh, X, y, g):
    """Trả về (nhãn thật, xác suất buồn ngủ, mã người) của toàn bộ các lần giữ-1-người."""
    y_that, xac_suat, nguoi = [], [], []
    for ng in sorted(set(g)):
        te = g == ng
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        mo_hinh = tao_mo_hinh()
        if isinstance(mo_hinh, GridSearchCV):
            mo_hinh.fit(X[tr], y[tr], groups=g[tr])
        else:
            mo_hinh.fit(X[tr], y[tr])
        xac_suat.append(mo_hinh.predict_proba(X[te])[:, 1])
        y_that.append(y[te])
        nguoi.append(g[te])
    return np.concatenate(y_that), np.concatenate(xac_suat), np.concatenate(nguoi)


def huan_luyen():
    df = doc_du_lieu()
    X = df[TEN_DAC_TRUNG].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    g = df["subject"].astype(str).to_numpy()
    nguoi = sorted(set(g))

    in_ra("=" * 70)
    in_ra(f"DU LIEU: {len(df)} cua so | {len(nguoi)} nguoi {nguoi}")
    in_ra(f"Tinh tao (0): {(y == 0).sum()} | Buon ngu (1): {(y == 1).sum()}")
    if len(nguoi) < 3 or len(set(y)) < 2:
        raise SystemExit("Can it nhat 3 nguoi va du ca 2 nhan de danh gia theo nguoi.")

    # ---------- 1. So sánh các mô hình bằng LOSO ----------
    in_ra("\n" + "=" * 70)
    in_ra("DANH GIA LEAVE-ONE-SUBJECT-OUT (nguong 0.5) - nhan 'Buon ngu' la lop duong")
    in_ra(f"{'Mo hinh':40s} {'Acc':>6s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FN':>4s}")
    ket_qua = {}
    for ten, tao in tao_cac_mo_hinh().items():
        print(f"  ...dang chay: {ten}")
        ket_qua[ten] = chay_loso(tao, X, y, g)
        m = chi_so(ket_qua[ten][0], ket_qua[ten][1])
        in_ra(f"{ten:40s} {m['acc']:6.3f} {m['prec']:6.3f} {m['rec']:7.3f} {m['f1']:6.3f} {m['fn']:4d}")

    # ---------- 2. Chi tiết Random Forest ----------
    ten_rf = "Random Forest (tinh chinh)"
    y_that, p, ng = ket_qua[ten_rf]

    in_ra("\n" + "=" * 70)
    in_ra("RANDOM FOREST - MA TRAN NHAM LAN (tong hop cac lan giu-1-nguoi, nguong 0.5)")
    m = chi_so(y_that, p, 0.5)
    in_ra("                  Du doan: Tinh tao   Du doan: Buon ngu")
    in_ra(f"Thuc te Tinh tao   {m['tn']:12d}   {m['fp']:14d}   (FP = canh bao nham)")
    in_ra(f"Thuc te Buon ngu   {m['fn']:12d}   {m['tp']:14d}   (FN = BO SOT, nguy hiem nhat)")

    in_ra("\nANH HUONG CUA NGUONG CANH BAO (ha nguong -> Recall tang, canh bao nham tang)")
    in_ra(f"{'Nguong':>7s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mt = chi_so(y_that, p, t)
        in_ra(f"{t:7.1f} {mt['prec']:6.3f} {mt['rec']:7.3f} {mt['f1']:6.3f} {mt['fp']:4d} {mt['fn']:4d}")

    in_ra("\nKET QUA THEO TUNG NGUOI (mo hinh chua tung thay nguoi nay luc huan luyen)")
    in_ra(f"{'Nguoi':>8s} {'So cua so':>10s} {'Acc':>6s} {'Recall':>7s}")
    for n in sorted(set(ng)):
        k = ng == n
        mn = chi_so(y_that[k], p[k], 0.5)
        in_ra(f"{n:>8s} {k.sum():10d} {mn['acc']:6.3f} {mn['rec']:7.3f}")

    # ---------- 3. Huấn luyện mô hình cuối cùng trên TOÀN BỘ dữ liệu ----------
    in_ra("\n" + "=" * 70)
    in_ra("HUAN LUYEN MO HINH CUOI CUNG (tren toan bo nguoi)")
    cuoi = tao_rf()
    cuoi.fit(X, y, groups=g)
    in_ra("Sieu tham so tot nhat:", cuoi.best_params_)
    tot_nhat = cuoi.best_estimator_
    joblib.dump({"pipeline": tot_nhat, "ten_dac_trung": TEN_DAC_TRUNG,
                 "nguong": NGUONG_CANH_BAO}, FILE_MODEL)
    in_ra(f"Da luu mo hinh: {FILE_MODEL}")

    rf = tot_nhat.named_steps["randomforestclassifier"]
    thu_tu = np.argsort(rf.feature_importances_)[::-1]
    in_ra("\nTOP 10 DAC TRUNG QUAN TRONG NHAT:")
    for i in thu_tu[:10]:
        in_ra(f"   {TEN_DAC_TRUNG[i]:12s} {rf.feature_importances_[i]:.3f}")

    in_ra("\nLUU Y: du lieu it nguoi nen ket qua dao dong nhieu giua cac nguoi; "
          "them nguoi tham gia se dang tin hon.")

    luu_bao_cao(FILE_BAO_CAO_HUAN_LUYEN)

    # ---------- 4. Vẽ hình cho báo cáo (nếu có matplotlib) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ten = ["Tỉnh táo", "Buồn ngủ"]
        ax.set_xticks([0, 1], ten)
        ax.set_yticks([0, 1], ten)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title("Ma trận nhầm lẫn (LOSO)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.tight_layout()
        fig.savefig("ma_tran_nham_lan.png", dpi=150)

        fig, ax = plt.subplots(figsize=(6, 5))
        top = thu_tu[:10][::-1]
        ax.barh([TEN_DAC_TRUNG[i] for i in top], rf.feature_importances_[top])
        ax.set_title("Độ quan trọng của đặc trưng (Random Forest)")
        fig.tight_layout()
        fig.savefig("do_quan_trong_dac_trung.png", dpi=150)
        print("Da luu hinh: ma_tran_nham_lan.png, do_quan_trong_dac_trung.png")
    except ImportError:
        print("(Bo qua ve hinh: chay 'pip install matplotlib' neu muon co hinh cho bao cao)")


# =====================================================================
# PHẦN 2: KIỂM TRA DỮ LIỆU THEO NGƯỜI VÀ THEO VIDEO
# =====================================================================
def kiem_tra_du_lieu(df):
    df = df.copy()
    df["subject"] = df["subject"].astype(str)

    in_ra("=" * 80)
    in_ra("1. TRUNG BINH THEO NGUOI VA NHAN  (nhan 0 = tinh tao, 1 = buon ngu)")
    bang = df.groupby(["subject", "label"])[COT_KIEM_TRA].mean().round(3)
    bang.insert(0, "so_cua_so", df.groupby(["subject", "label"]).size())
    in_ra(bang.to_string())

    in_ra("\n" + "=" * 80)
    in_ra("2. CHENH LECH (buon ngu - tinh tao) CUA TUNG NGUOI")
    in_ra("   Ky vong: ear_mean < 0, ear_min < 0, ear_std > 0")
    dong = []
    for s in sorted(df["subject"].unique()):
        a = df[(df["subject"] == s) & (df["label"] == 0)]
        b = df[(df["subject"] == s) & (df["label"] == 1)]
        if len(a) == 0 or len(b) == 0:
            in_ra(f"   {s}: chi co 1 nhan -> khong so sanh duoc")
            continue
        d = (b[COT_KIEM_TRA].mean() - a[COT_KIEM_TRA].mean()).round(3)
        d.name = s
        dong.append(d)
    if dong:
        chenh = pd.DataFrame(dong)
        chenh["canh_bao"] = ["<-- NGUOC KY VONG" if v >= 0 else "" for v in chenh["ear_mean"]]
        in_ra(chenh.to_string())
    else:
        in_ra("   Khong co nguoi nao du ca 2 nhan.")

    in_ra("\n" + "=" * 80)
    in_ra("3. TUNG VIDEO")
    v = df.groupby(["subject", "video", "label"]).agg(
        so_cua_so=("label", "size"),
        ear_mean=("ear_mean", "mean"),
        ear_min=("ear_min", "mean"),
        mar_max=("mar_max", "mean"),
    ).round(3)
    in_ra(v.to_string())

    in_ra("\nGoi y doc ket qua:")
    in_ra(" - Nguoi nao bi canh bao: mo lai video cua nguoi do, kiem tra clip co nam dung thu muc")
    in_ra("   tinh_tao / buon_ngu khong, va co bi quay khac anh sang / tu the / goc may khong.")
    in_ra(" - Video nao co ear_mean rat khac cac video cung nhan: xem lai video do.")


# =====================================================================
# PHẦN 3: SO SÁNH CÁC NHÓM ĐẶC TRƯNG (LOSO, chạy nhanh, dự đoán nhãn cứng)
# =====================================================================
def loso_nhanh(tao, X, y, g):
    yt, pr, ng = [], [], []
    for n in sorted(set(g)):
        te = g == n
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        m = tao()
        m.fit(X[tr], y[tr])
        pr.append(m.predict(X[te]))
        yt.append(y[te])
        ng.append(g[te])
    return np.concatenate(yt), np.concatenate(pr), np.concatenate(ng)


def so_sanh_dac_trung(df):
    y = df["label"].to_numpy(int)
    g = df["subject"].astype(str).to_numpy()
    nguoi = sorted(set(g))

    in_ra("\n" + "=" * 80)
    in_ra(f"SO SANH NHOM DAC TRUNG: {len(df)} cua so, {len(nguoi)} nguoi. "
          "Danh gia Leave-One-Subject-Out, nguong 0.5")
    in_ra(f"\n{'Nhom dac trung':48s} {'Mo hinh':12s} {'Acc':>6s} {'Prec':>6s} "
          f"{'Recall':>7s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")
    chi_tiet = {}
    for ten_nhom, cot in NHOM.items():
        X = df[cot].to_numpy(float)
        for ten_mh, tao in MO_HINH_NHANH.items():
            yt, pr, ng = loso_nhanh(tao, X, y, g)
            tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
            in_ra(f"{ten_nhom:48s} {ten_mh:12s} {accuracy_score(yt, pr):6.3f} "
                  f"{precision_score(yt, pr, zero_division=0):6.3f} "
                  f"{recall_score(yt, pr, zero_division=0):7.3f} "
                  f"{f1_score(yt, pr, zero_division=0):6.3f} {fp:4d} {fn:4d}")
            chi_tiet[(ten_nhom, ten_mh)] = (yt, pr, ng)

    in_ra("\nDO CHINH XAC THEO TUNG NGUOI (moi nguoi la nguoi 'moi' chua thay luc huan luyen)")
    in_ra(f"{'Nhom dac trung':48s} {'Mo hinh':12s} " + " ".join(f"{n:>6s}" for n in nguoi))
    for (ten_nhom, ten_mh), (yt, pr, ng) in chi_tiet.items():
        acc = [accuracy_score(yt[ng == n], pr[ng == n]) if (ng == n).any() else float("nan")
               for n in nguoi]
        in_ra(f"{ten_nhom:48s} {ten_mh:12s} " + " ".join(f"{a:6.2f}" for a in acc))

    in_ra("\nDoc ket qua: nhom nao co Acc/F1 cao VA do chinh xac deu giua cac nguoi (khong co "
          "nguoi nao ~0) la nhom tong quat hoa tot.")


def phan_tich(chay_kiem_tra=True, chay_so_sanh=True):
    df = doc_du_lieu()
    if chay_kiem_tra:
        kiem_tra_du_lieu(df)
    if chay_so_sanh:
        so_sanh_dac_trung(df)
    luu_bao_cao(FILE_BAO_CAO_PHAN_TICH)


# =====================================================================
def main():
    lenh = sys.argv[1].lower() if len(sys.argv) > 1 else "huan_luyen"
    if lenh == "huan_luyen":
        huan_luyen()
    elif lenh == "kiem_tra":
        phan_tich(chay_so_sanh=False)
    elif lenh == "so_sanh":
        phan_tich(chay_kiem_tra=False)
    elif lenh == "phan_tich":
        phan_tich()
    elif lenh == "tat_ca":
        phan_tich()
        huan_luyen()
    else:
        raise SystemExit("Cach dung: python buoc5_huan_luyen.py "
                         "[huan_luyen | kiem_tra | so_sanh | phan_tich | tat_ca]")


if __name__ == "__main__":
    main()