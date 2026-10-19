"""
BƯỚC 5: Huấn luyện + đánh giá + phân tích dữ liệu (đã gộp Bước 5b).

Cần: du_lieu_dac_trung.csv (từ Bước 4), dac_trung.py

Cách chạy:
    python huan_luyen.py              -> huấn luyện + đánh giá + lưu model.pkl (mặc định)
    python huan_luyen.py kiem_tra     -> kiểm tra dữ liệu có hợp lý không (nhanh)
    python huan_luyen.py so_sanh      -> so sánh các nhóm đặc trưng (nhanh)
    python huan_luyen.py phan_tich    -> kiem_tra + so_sanh
    python huan_luyen.py tat_ca       -> phan_tich rồi huấn luyện

Kết quả:
    huấn luyện : model.pkl, ket_qua_danh_gia.txt, ma_tran_nham_lan.png, do_quan_trong_dac_trung.png
    phân tích  : ket_qua_phan_tich.txt

Ngưỡng cảnh báo KHÔNG còn là 0.5 cảm tính. Nó được chọn từ đường cong ROC của các dự đoán
LOSO (xem chon_nguong), kèm khoảng tin cậy bootstrap theo người và một ước lượng khách quan
(ngưỡng được chọn mà không nhìn thấy người test). Toàn bộ nằm trong ket_qua_danh_gia.txt.

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
from app.paths import DATA, MODELS, VIDEOS, REPORTS, data_file, model_file, report_file

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.dac_trung import TEN_DAC_TRUNG

# ================= CẤU HÌNH =================
FILE_CSV = str(data_file("du_lieu_dac_trung.csv"))
FILE_MODEL = str(model_file("model.pkl"))
FILE_BAO_CAO_HUAN_LUYEN = str(report_file("ket_qua_danh_gia.txt"))
FILE_BAO_CAO_PHAN_TICH = str(report_file("ket_qua_phan_tich.txt"))
SEED = 42

# --- Chọn ngưỡng cảnh báo (xác suất "buồn ngủ" >= ngưỡng thì coi là buồn ngủ) ---
# "youden": cực đại hoá (Recall - Tỉ lệ báo nhầm) = Youden's J. Không cần giả định gì thêm,
#           cân bằng giữa bỏ sót và báo nhầm. Đây là chuẩn phổ biến trong y sinh/phân loại.
# "eer"   : điểm tỉ lệ báo nhầm = tỉ lệ bỏ sót (Equal Error Rate).
# "recall": ngưỡng cao nhất mà vẫn đạt Recall >= RECALL_MUC_TIEU (ưu tiên không bỏ sót).
PHUONG_PHAP_NGUONG = "youden"
RECALL_MUC_TIEU = 0.95
SO_LAN_BOOTSTRAP = 500     # số lần lấy mẫu lại theo NGƯỜI để ước lượng độ ổn định của ngưỡng

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
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=TEN_DAC_TRUNG)


# =====================================================================
# PHẦN DÙNG CHUNG
# =====================================================================
def chon_cot(X, cot):
    return X[:, [cot]]


def chi_so_nhan(y, d):
    """Các chỉ số từ NHÃN DỰ ĐOÁN d (0/1)."""
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y, d),
        "prec": precision_score(y, d, zero_division=0),
        "rec": recall_score(y, d, zero_division=0),
        "f1": f1_score(y, d, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def chi_so(y, p, nguong):
    """Các chỉ số khi cắt xác suất p tại `nguong` (p >= nguong -> buồn ngủ)."""
    return chi_so_nhan(y, (p >= nguong).astype(int))


# =====================================================================
# CHỌN NGƯỠNG CẢNH BÁO TỪ DỮ LIỆU
# =====================================================================
def chon_nguong(y, p, phuong_phap=PHUONG_PHAP_NGUONG, recall_muc_tieu=RECALL_MUC_TIEU):
    """Chọn ngưỡng từ đường cong ROC của (nhãn thật y, xác suất p).

    Mỗi điểm trên đường ROC là một ngưỡng với (TPR = Recall, FPR = tỉ lệ báo nhầm).
    Kết quả được đặt ở GIỮA hai giá trị xác suất liền kề để không "dán" sát vào một mẫu
    cụ thể (dự đoán trên dữ liệu hiện có không đổi, nhưng ngưỡng ổn định hơn).
    """
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    if phuong_phap == "youden":
        i = int(np.argmax(tpr - fpr))
    elif phuong_phap == "eer":
        i = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    elif phuong_phap == "recall":
        du = np.where(tpr >= recall_muc_tieu)[0]
        i = int(du[0]) if len(du) else len(thr) - 1   # FPR nhỏ nhất trong các điểm đạt Recall
    else:
        raise ValueError(f"phuong_phap khong hop le: {phuong_phap}")
    t = thr[i]
    if not np.isfinite(t):                      # điểm đầu tiên của ROC là "không báo gì"
        t = thr[1] if len(thr) > 1 else 0.5
    elif i + 1 < len(thr):
        t = (thr[i] + thr[i + 1]) / 2.0
    return float(t)


def nguong_bootstrap(y, p, ng, phuong_phap=PHUONG_PHAP_NGUONG, so_lan=SO_LAN_BOOTSTRAP):
    """Lấy mẫu lại theo NGƯỜI (có hoàn lại), chọn ngưỡng cho từng lần.

    Phân bố các ngưỡng này cho biết ngưỡng ổn định đến đâu khi đổi bộ người tham gia.
    """
    rng = np.random.default_rng(SEED)
    nguoi = np.array(sorted(set(ng)))
    chi_muc = {n: np.where(ng == n)[0] for n in nguoi}
    ts = []
    for _ in range(so_lan):
        chon = rng.choice(nguoi, size=len(nguoi), replace=True)
        idx = np.concatenate([chi_muc[n] for n in chon])
        if len(set(y[idx])) < 2:
            continue
        ts.append(chon_nguong(y[idx], p[idx], phuong_phap))
    return np.array(ts)


def danh_gia_khach_quan(y, p, ng, phuong_phap=PHUONG_PHAP_NGUONG):
    """Ước lượng không lạc quan: với mỗi người, ngưỡng được chọn CHỈ từ những người khác.

    (Chọn ngưỡng và đo kết quả trên cùng dữ liệu luôn cho số đẹp hơn thực tế.)
    Trả về (nhãn dự đoán d, dict người -> ngưỡng đã dùng).
    """
    d = np.zeros(len(y), dtype=int)
    da_dung = {}
    for n in sorted(set(ng)):
        te = ng == n
        tr = ~te
        t = chon_nguong(y[tr], p[tr], phuong_phap) if len(set(y[tr])) == 2 else 0.5
        da_dung[n] = t
        d[te] = (p[te] >= t).astype(int)
    return d, da_dung


# =====================================================================
# PHẦN 1: HUẤN LUYỆN + ĐÁNH GIÁ (LOSO) + LƯU MODEL
# =====================================================================
def tao_rf():
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=SEED, class_weight="balanced"),
    )
    # scoring="roc_auc": chọn siêu tham số không phụ thuộc vào một ngưỡng cố định nào
    return GridSearchCV(pipe, LUOI_SIEU_THAM_SO, cv=LeaveOneGroupOut(),
                        scoring="roc_auc", error_score=0.5, n_jobs=-1)


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
    """Trả về (nhãn thật, xác suất buồn ngủ, mã người, chỉ mục hàng gốc) của mọi lần giữ-1-người."""
    y_that, xac_suat, nguoi, chi_muc = [], [], [], []
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
        chi_muc.append(np.where(te)[0])
    if not y_that:
        raise ValueError("Khong co fold LOSO hop le: can them nguoi va du 2 nhan cho tap huan luyen.")
    return (np.concatenate(y_that), np.concatenate(xac_suat),
            np.concatenate(nguoi), np.concatenate(chi_muc))


def ve_hinh(m_chon, t_chon, y_that, p, thu_tu, rf):
    """Vẽ 4 hình cho báo cáo (cần matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(Bo qua ve hinh: chay 'pip install matplotlib' neu muon co hinh cho bao cao)")
        return

    # 1. Ma trận nhầm lẫn tại ngưỡng đã chọn
    cm = np.array([[m_chon["tn"], m_chon["fp"]], [m_chon["fn"], m_chon["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ten = ["Tỉnh táo", "Buồn ngủ"]
    ax.set_xticks([0, 1], ten)
    ax.set_yticks([0, 1], ten)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn (LOSO, ngưỡng {t_chon:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(report_file("ma_tran_nham_lan.png")), dpi=150)
    plt.close(fig)

    # 2. Độ quan trọng đặc trưng
    fig, ax = plt.subplots(figsize=(6, 5))
    top = thu_tu[:10][::-1]
    ax.barh([TEN_DAC_TRUNG[i] for i in top], rf.feature_importances_[top])
    ax.set_title("Độ quan trọng của đặc trưng (Random Forest)")
    fig.tight_layout()
    fig.savefig(str(report_file("do_quan_trong_dac_trung.png")), dpi=150)
    plt.close(fig)

    # 3. Đường ROC, đánh dấu ngưỡng 0.5 và ngưỡng đã chọn
    fpr, tpr, thr = roc_curve(y_that, p, drop_intermediate=False)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_that, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    for t, nhan, mau in [(0.5, "ngưỡng 0.5", "tab:gray"), (t_chon, f"đã chọn {t_chon:.2f}", "tab:red")]:
        mt = chi_so(y_that, p, t)
        ax.plot(mt["fp"] / max(mt["fp"] + mt["tn"], 1), mt["rec"], "o", color=mau, label=nhan)
    ax.set_xlabel("Tỉ lệ báo nhầm (FPR)")
    ax.set_ylabel("Recall (tỉ lệ bắt được buồn ngủ)")
    ax.set_title("Đường ROC (LOSO)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(report_file("roc_nguong.png")), dpi=150)
    plt.close(fig)

    # 4. Precision / Recall / F1 theo ngưỡng
    luoi = np.linspace(0.05, 0.95, 91)
    ps, rs, fs = [], [], []
    for t in luoi:
        mt = chi_so(y_that, p, t)
        ps.append(mt["prec"]); rs.append(mt["rec"]); fs.append(mt["f1"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(luoi, ps, label="Precision")
    ax.plot(luoi, rs, label="Recall")
    ax.plot(luoi, fs, label="F1")
    ax.axvline(t_chon, color="tab:red", linestyle="--", label=f"đã chọn {t_chon:.2f}")
    ax.axvline(0.5, color="tab:gray", linestyle=":", label="0.5")
    ax.set_xlabel("Ngưỡng cảnh báo")
    ax.set_title("Chỉ số theo ngưỡng (LOSO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(report_file("nguong_vs_chi_so.png")), dpi=150)
    plt.close(fig)
    print(f"Da luu cac bieu do vao: {REPORTS}")


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
    thieu = [n for n in nguoi if len(set(y[g == n])) < 2]
    if thieu:
        in_ra(f"CANH BAO: nguoi {thieu} chi co 1 nhan -> AUC rieng cua ho khong tinh duoc; "
              "nen quay du ca 2 nhan cho moi nguoi.")

    # ---------- 1. So sánh các mô hình bằng LOSO ----------
    in_ra("\n" + "=" * 70)
    in_ra("DANH GIA LEAVE-ONE-SUBJECT-OUT - nhan 'Buon ngu' la lop duong")
    in_ra("AUC khong phu thuoc nguong. Cac cot con lai tinh tai nguong Youden rieng cua tung mo hinh.")
    in_ra(f"{'Mo hinh':40s} {'AUC':>6s} {'Nguong':>7s} {'Acc':>6s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FN':>4s}")
    ket_qua = {}
    for ten, tao in tao_cac_mo_hinh().items():
        print(f"  ...dang chay: {ten}")
        ket_qua[ten] = chay_loso(tao, X, y, g)
        yt, pt = ket_qua[ten][0], ket_qua[ten][1]
        t_mh = chon_nguong(yt, pt, "youden")
        m = chi_so(yt, pt, t_mh)
        in_ra(f"{ten:40s} {roc_auc_score(yt, pt):6.3f} {t_mh:7.3f} {m['acc']:6.3f} "
              f"{m['prec']:6.3f} {m['rec']:7.3f} {m['f1']:6.3f} {m['fn']:4d}")

    # ---------- 2. Random Forest: chọn ngưỡng ----------
    ten_rf = "Random Forest (tinh chinh)"
    y_that, p, ng, chi_muc = ket_qua[ten_rf]

    in_ra("\n" + "=" * 70)
    in_ra("CHON NGUONG CANH BAO CHO RANDOM FOREST")
    in_ra(f"AUC = {roc_auc_score(y_that, p):.3f} | Average Precision = {average_precision_score(y_that, p):.3f}")
    in_ra("Moi nguong la mot diem tren duong ROC: ha nguong -> Recall tang nhung bao nham tang.")
    in_ra(f"{'Cach chon':34s} {'Nguong':>7s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")
    ung_vien = [("Mac dinh cu (0.5, khong co co so)", 0.5),
                ("Youden J (max Recall - FPR)", chon_nguong(y_that, p, "youden")),
                ("EER (bao nham = bo sot)", chon_nguong(y_that, p, "eer")),
                (f"Recall >= {RECALL_MUC_TIEU:.2f}", chon_nguong(y_that, p, "recall"))]
    for ten_cach, t in ung_vien:
        mt = chi_so(y_that, p, t)
        in_ra(f"{ten_cach:34s} {t:7.3f} {mt['prec']:6.3f} {mt['rec']:7.3f} {mt['f1']:6.3f} "
              f"{mt['fp']:4d} {mt['fn']:4d}")

    t_chon = chon_nguong(y_that, p, PHUONG_PHAP_NGUONG)
    in_ra(f"\n=> NGUONG DUOC CHON ({PHUONG_PHAP_NGUONG}): {t_chon:.3f}")

    ts = nguong_bootstrap(y_that, p, ng)
    if len(ts) >= 10:
        lo, med, hi = np.percentile(ts, [2.5, 50, 97.5])
        in_ra(f"Do on dinh (bootstrap {len(ts)} lan, lay mau lai theo NGUOI): "
              f"trung vi {med:.3f}, khoang 95% [{lo:.3f}; {hi:.3f}]")
        if hi - lo > 0.3:
            in_ra("   -> Khoang rong: nguong con phu thuoc nhieu vao tung nguoi, can them nguoi tham gia.")
    else:
        lo = med = hi = float("nan")
        in_ra("Khong du du lieu de bootstrap do on dinh cua nguong.")

    d_kq, da_dung = danh_gia_khach_quan(y_that, p, ng)
    mk = chi_so_nhan(y_that, d_kq)
    in_ra("\nUOC LUONG KHACH QUAN (nguong cua moi nguoi duoc chon CHI tu nhung nguoi con lai):")
    in_ra(f"   Acc {mk['acc']:.3f} | Prec {mk['prec']:.3f} | Recall {mk['rec']:.3f} | "
          f"F1 {mk['f1']:.3f} | FP {mk['fp']} | FN {mk['fn']}")
    in_ra("   Day moi la so nen bao cao; so o bang tren hoi lac quan vi chon va do tren cung du lieu.")

    # ---------- 3. Chi tiết tại ngưỡng đã chọn ----------
    m0 = chi_so(y_that, p, 0.5)
    m_chon = chi_so(y_that, p, t_chon)
    in_ra("\n" + "=" * 70)
    in_ra("MA TRAN NHAM LAN (tong hop cac lan giu-1-nguoi)")
    for ten_cach, t, mm in [("nguong 0.5 (cu)", 0.5, m0), (f"nguong {t_chon:.3f} (da chon)", t_chon, m_chon)]:
        in_ra(f"  [{ten_cach}]")
        in_ra("                  Du doan: Tinh tao   Du doan: Buon ngu")
        in_ra(f"  Thuc te Tinh tao   {mm['tn']:12d}   {mm['fp']:14d}   (FP = canh bao nham)")
        in_ra(f"  Thuc te Buon ngu   {mm['fn']:12d}   {mm['tp']:14d}   (FN = BO SOT, nguy hiem nhat)")

    in_ra("\nANH HUONG CUA NGUONG CANH BAO")
    in_ra(f"{'Nguong':>7s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")
    for t in sorted(set([0.3, 0.4, 0.5, 0.6, 0.7, t_chon])):
        mt = chi_so(y_that, p, t)
        danh_dau = "  <-- da chon" if t == t_chon else ""
        in_ra(f"{t:7.3f} {mt['prec']:6.3f} {mt['rec']:7.3f} {mt['f1']:6.3f} {mt['fp']:4d} {mt['fn']:4d}{danh_dau}")

    in_ra("\nKET QUA THEO TUNG NGUOI (mo hinh chua tung thay nguoi nay luc huan luyen, nguong da chon)")
    in_ra(f"{'Nguoi':>8s} {'So cua so':>10s} {'Acc':>6s} {'Recall':>7s}")
    for n in sorted(set(ng)):
        k = ng == n
        mn = chi_so(y_that[k], p[k], t_chon)
        in_ra(f"{n:>8s} {k.sum():10d} {mn['acc']:6.3f} {mn['rec']:7.3f}")

    if "dieu_kien" in df.columns:
        dk = df["dieu_kien"].astype(str).to_numpy()[chi_muc]
        in_ra("\nKET QUA THEO DIEU KIEN QUAY (ngay / dem / kinh / demkinh / dui ...), nguong da chon")
        in_ra(f"{'Dieu kien':>10s} {'So cua so':>10s} {'Acc':>6s} {'Bao nham(FP)':>13s} {'Bo sot(FN)':>11s}")
        for c in sorted(set(dk)):
            k = dk == c
            mc = chi_so(y_that[k], p[k], t_chon)
            in_ra(f"{c:>10s} {k.sum():10d} {mc['acc']:6.3f} {mc['fp']:13d} {mc['fn']:11d}")
        in_ra("   ('dui' = quay luc dui mat mot ben, nhan tinh tao: FP o day la bao nham do dui mat)")

    # ---------- 4. Huấn luyện mô hình cuối cùng trên TOÀN BỘ dữ liệu ----------
    in_ra("\n" + "=" * 70)
    in_ra("HUAN LUYEN MO HINH CUOI CUNG (tren toan bo nguoi)")
    cuoi = tao_rf()
    cuoi.fit(X, y, groups=g)
    in_ra("Sieu tham so tot nhat:", cuoi.best_params_)
    tot_nhat = cuoi.best_estimator_
    thong_tin = {"phuong_phap": PHUONG_PHAP_NGUONG, "auc_loso": float(roc_auc_score(y_that, p)),
                 "khoang_95": [float(lo), float(hi)], "so_nguoi": len(nguoi)}
    joblib.dump({"pipeline": tot_nhat, "ten_dac_trung": TEN_DAC_TRUNG,
                 "nguong": float(t_chon), "nguong_info": thong_tin}, FILE_MODEL)
    in_ra(f"Da luu mo hinh: {FILE_MODEL}  (nguong canh bao = {t_chon:.3f})")

    rf = tot_nhat.named_steps["randomforestclassifier"]
    thu_tu = np.argsort(rf.feature_importances_)[::-1]
    in_ra("\nTOP 10 DAC TRUNG QUAN TRONG NHAT:")
    for i in thu_tu[:10]:
        in_ra(f"   {TEN_DAC_TRUNG[i]:12s} {rf.feature_importances_[i]:.3f}")

    in_ra("\nLUU Y: du lieu it nguoi nen ket qua dao dong nhieu giua cac nguoi; "
          "them nguoi tham gia se dang tin hon.")

    luu_bao_cao(FILE_BAO_CAO_HUAN_LUYEN)
    ve_hinh(m_chon, t_chon, y_that, p, thu_tu, rf)


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
    yt, pb, ng = [], [], []
    for n in sorted(set(g)):
        te = g == n
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        m = tao()
        m.fit(X[tr], y[tr])
        pb.append(m.predict_proba(X[te])[:, 1])
        yt.append(y[te])
        ng.append(g[te])
    if not yt:
        raise ValueError("Khong co fold LOSO hop le.")
    yt, pb, ng = np.concatenate(yt), np.concatenate(pb), np.concatenate(ng)
    return yt, (pb >= 0.5).astype(int), ng, pb


def so_sanh_dac_trung(df):
    y = df["label"].to_numpy(int)
    g = df["subject"].astype(str).to_numpy()
    nguoi = sorted(set(g))

    in_ra("\n" + "=" * 80)
    in_ra(f"SO SANH NHOM DAC TRUNG: {len(df)} cua so, {len(nguoi)} nguoi. "
          "Danh gia Leave-One-Subject-Out, nguong 0.5")
    in_ra(f"\n{'Nhom dac trung':48s} {'Mo hinh':12s} {'AUC':>6s} {'Acc':>6s} {'Prec':>6s} "
          f"{'Recall':>7s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")
    in_ra("(AUC khong phu thuoc nguong - dung de xep hang nhom dac trung; cac cot khac cat tai 0.5)")
    chi_tiet = {}
    for ten_nhom, cot in NHOM.items():
        X = df[cot].to_numpy(float)
        for ten_mh, tao in MO_HINH_NHANH.items():
            yt, pr, ng, pb = loso_nhanh(tao, X, y, g)
            tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
            in_ra(f"{ten_nhom:48s} {ten_mh:12s} {roc_auc_score(yt, pb):6.3f} {accuracy_score(yt, pr):6.3f} "
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
        raise SystemExit("Cach dung: python huan_luyen.py "
                         "[huan_luyen | kiem_tra | so_sanh | phan_tich | tat_ca]")


if __name__ == "__main__":
    main()