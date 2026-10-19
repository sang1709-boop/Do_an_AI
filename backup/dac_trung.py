"""
dac_trung.py - Các hàm tính đặc trưng hình học từ điểm mốc khuôn mặt MediaPipe.

File này KHÔNG cần webcam, chỉ cần NumPy. Các bước sau (trích xuất dữ liệu,
chạy realtime) sẽ import lại file này:  from dac_trung import ...
"""

import json
import os

import numpy as np

# ---- Chỉ số (index) các điểm mốc trong lưới 478 điểm của MediaPipe ----
# Mắt: 6 điểm theo thứ tự p1..p6 của công thức EAR
#   p1, p4 = hai khóe mắt (ngang);  p2, p3 = mi trên;  p6, p5 = mi dưới
MAT_PHAI = [33, 160, 158, 133, 153, 144]
MAT_TRAI = [362, 385, 387, 263, 373, 380]

# Miệng (viền môi trong): 3 cặp điểm dọc + 1 cặp khóe miệng (ngang)
MIENG_DOC = [(82, 87), (13, 14), (312, 317)]
MIENG_NGANG = (78, 308)

# ---- Điểm mốc để VẼ / định vị thêm ----
# Khóe mắt: 33, 133 (mắt phải: ngoài, trong) và 362, 263 (mắt trái: trong, ngoài)
KHOE_MAT = [33, 133, 362, 263]
# Viền khuôn mặt (face oval, 36 điểm, theo thứ tự đi vòng quanh mặt)
VIEN_MAT = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
            379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
            234, 127, 162, 21, 54, 103, 67, 109]
# Hai điểm ngoài cùng hai bên má: dùng làm "thước" bề rộng khuôn mặt
RONG_MAT = (234, 454)


def lay_toa_do(landmarks, w, h):
    """Đổi điểm mốc (x, y trong khoảng 0..1) sang tọa độ pixel.

    Bắt buộc nhân với w và h riêng, nếu không khoảng cách bị méo
    vì ảnh webcam không phải hình vuông.
    """
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks])


def _kc(a, b):
    """Khoảng cách Euclid giữa 2 điểm."""
    return float(np.linalg.norm(a - b))


def ear_mot_mat(pts, idx):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    return (_kc(p2, p6) + _kc(p3, p5)) / (2.0 * _kc(p1, p4))


def tinh_ear(pts):
    """EAR trung bình của hai mắt."""
    return (ear_mot_mat(pts, MAT_PHAI) + ear_mot_mat(pts, MAT_TRAI)) / 2.0


def tinh_mar(pts):
    """MAR = tổng 3 khoảng cách dọc của miệng / (3 * khoảng cách ngang)."""
    doc = sum(_kc(pts[a], pts[b]) for a, b in MIENG_DOC)
    ngang = _kc(pts[MIENG_NGANG[0]], pts[MIENG_NGANG[1]])
    return doc / (3.0 * ngang)


def tinh_tu_the_dau(ma_tran):
    """Tính (pitch, yaw, roll) theo độ từ ma trận biến đổi 4x4 của MediaPipe.

    pitch: gật / ngẩng đầu,  yaw: quay trái / phải,  roll: nghiêng đầu sang vai.
    """
    R = np.array(ma_tran)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    # Đưa pitch về khoảng [-90, 90] để không bị nhảy +-180
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return float(pitch), float(yaw), float(roll)


# =====================================================================
# VECTOR ĐẶC TRƯNG THEO CỬA SỔ THỜI GIAN (Bước 4 và Bước 6 dùng chung)
# Quan trọng: lúc huấn luyện và lúc chạy realtime phải tính y hệt nhau.
# =====================================================================

TEN_CHI_SO = ["ear", "mar", "pitch", "yaw", "roll"]
TEN_THONG_KE = ["mean", "min", "max", "std"]
TEN_DAC_TRUNG = [f"{c}_{t}" for c in TEN_CHI_SO for t in TEN_THONG_KE]  # 20 đặc trưng

TAN_SO_MAU = 15          # lấy mẫu 15 khung hình / giây
CUA_SO = 30              # mỗi cửa sổ = 30 mẫu = 2 giây
BUOC_TRUOT = 15          # dịch cửa sổ 15 mẫu (1 giây), chồng lấn 50%
TI_LE_CO_MAT_TOI_THIEU = 0.8   # cửa sổ phải có mặt ở >= 80% số mẫu


def vector_cua_so(cua_so_mau):
    """Tính vector đặc trưng của MỘT cửa sổ.

    cua_so_mau: mảng (n, 5), mỗi hàng = [ear, mar, pitch, yaw, roll].
                Hàng chứa NaN = khung hình không thấy mặt.
    Trả về mảng 20 số (mean, min, max, std của từng chỉ số),
    hoặc None nếu cửa sổ thiếu mặt quá nhiều.
    """
    a = np.asarray(cua_so_mau, dtype=float)
    co_mat = ~np.isnan(a).any(axis=1)
    if len(a) == 0 or co_mat.mean() < TI_LE_CO_MAT_TOI_THIEU:
        return None
    a = a[co_mat]
    ket_qua = []
    for j in range(a.shape[1]):
        c = a[:, j]
        ket_qua += [c.mean(), c.min(), c.max(), c.std()]
    return np.array(ket_qua)


def tao_cac_cua_so(chuoi_mau):
    """Trượt cửa sổ trên cả chuỗi mẫu của một video.

    Sinh ra từng cặp (vị trí mẫu bắt đầu, vector đặc trưng).
    """
    chuoi_mau = np.asarray(chuoi_mau, dtype=float)
    for s in range(0, len(chuoi_mau) - CUA_SO + 1, BUOC_TRUOT):
        v = vector_cua_so(chuoi_mau[s:s + CUA_SO])
        if v is not None:
            yield s, v

# =====================================================================
# MẮT: GỘP EAR HAI MẮT, LOẠI MẮT KHÔNG ĐÁNG TIN (DỤI MẮT / KÍNH LOÁ)
# Dùng chung cho Bước 4 (tao_du_lieu) và Bước 6 (realtime).
# =====================================================================
FILE_NGUONG_PHU = "nguong_phu.json"      # tao_du_lieu.py ghi, realtime.py đọc
NGUONG_LECH_MAT_MAC_DINH = 0.35          # chỉ dùng khi chưa có nguong_phu.json


def ear_hai_mat(pts):
    """Trả về (EAR mắt phải, EAR mắt trái)."""
    return ear_mot_mat(pts, MAT_PHAI), ear_mot_mat(pts, MAT_TRAI)


def lech_hai_mat(ear_p, ear_t):
    """Độ lệch giữa hai mắt = |ln(EAR_phải / EAR_trái)|.

    Dùng tỉ số (log) nên không phụ thuộc mắt to hay nhỏ. 0 = hai mắt giống hệt nhau.
    Hai mắt thật sự buồn ngủ thì nhắm CÙNG LÚC (lệch nhỏ); dụi một mắt thì một mắt bị tay
    che/nheo còn mắt kia vẫn mở (lệch lớn).
    """
    return abs(float(np.log(max(ear_p, 1e-6) / max(ear_t, 1e-6))))


def gop_ear(ear_p, ear_t, loa_p=False, loa_t=False, che_p=False, che_t=False,
            nguong_lech=NGUONG_LECH_MAT_MAC_DINH):
    """Gộp EAR hai mắt thành MỘT số, loại bỏ mắt không đáng tin.

    loa_*: mắt bị kính loá;  che_*: mắt bị tay che (phát hiện bằng điểm tay của Pose).
    Trả về (ear, ly_do), ly_do là một trong:
        "ok"            hai mắt bình thường -> trung bình hai mắt
        "loa" / "che"   một mắt loá / bị che -> dùng mắt còn lại
        "dui"           hai mắt lệch nhau hơn nguong_lech -> dùng mắt MỞ HƠN
                        (nếu thật sự buồn ngủ thì mắt mở hơn cũng sẽ nhắm theo)
        "khong_do_duoc" cả hai mắt đều không đáng tin -> ear = NaN
    """
    if np.isnan(ear_p) or np.isnan(ear_t):
        return float("nan"), "khong_do_duoc"
    xau_p, xau_t = bool(loa_p or che_p), bool(loa_t or che_t)
    if xau_p and xau_t:
        return float("nan"), "khong_do_duoc"
    if xau_p:
        return ear_t, ("che" if che_p else "loa")
    if xau_t:
        return ear_p, ("che" if che_t else "loa")
    if lech_hai_mat(ear_p, ear_t) > nguong_lech:
        return max(ear_p, ear_t), "dui"
    return (ear_p + ear_t) / 2.0, "ok"


def hang_mau(ear_p, ear_t, mar, pitch, yaw, roll, loa_p=False, loa_t=False,
             che_p=False, che_t=False, nguong_lech=NGUONG_LECH_MAT_MAC_DINH):
    """Tạo MỘT hàng [ear, mar, pitch, yaw, roll] cho cửa sổ trượt (cùng logic train/realtime).

    Trả về (hang, ly_do). Nếu mắt không đo được thì cả hàng là NaN (cửa sổ coi như thiếu mặt).
    """
    ear, ly_do = gop_ear(ear_p, ear_t, loa_p, loa_t, che_p, che_t, nguong_lech)
    if np.isnan(ear) or any(np.isnan(v) for v in (mar, pitch, yaw, roll)):
        return [float("nan")] * 5, ly_do
    return [ear, mar, pitch, yaw, roll], ly_do


def luu_nguong_phu(du_lieu, duong_dan=FILE_NGUONG_PHU):
    with open(duong_dan, "w", encoding="utf-8") as f:
        json.dump(du_lieu, f, indent=2, ensure_ascii=False)


def doc_nguong_lech(duong_dan=FILE_NGUONG_PHU):
    """Ngưỡng lệch hai mắt do tao_du_lieu.py tính từ dữ liệu; không có file thì dùng mặc định."""
    if os.path.exists(duong_dan):
        try:
            with open(duong_dan, encoding="utf-8") as f:
                return float(json.load(f)["nguong_lech_mat"])
        except (ValueError, KeyError, OSError):
            pass
    return NGUONG_LECH_MAT_MAC_DINH


# =====================================================================
# GIẤC NGỦ TRẮNG (MICROSLEEP): MẮT VẪN MỞ NHƯNG NÃO KHÔNG XỬ LÝ THÔNG TIN
# Cửa sổ 2 giây không thấy được "không chớp mắt" hay "nhìn trân trân", nên dùng cửa sổ DÀI
# 10 giây với các đặc trưng hành vi: chớp mắt, ánh nhìn (mống mắt), chuyển động đầu.
# Bộ phát hiện này tách riêng (model_dai.pkl), huấn luyện bằng huan_luyen_dai.py.
# =====================================================================
IRIS_PHAI, IRIS_TRAI = 468, 473          # tâm mống mắt trong lưới 478 điểm

CUA_SO_DAI = 150                         # 10 giây ở 15 mẫu/giây
BUOC_DAI = 75                            # dịch 5 giây
TI_LE_DIP_CHOP = 0.8                     # EAR tụt dưới 80% mức mắt-mở của cửa sổ = một lần chớp / sụp mí

TEN_CHI_SO_DAI = ["ear", "mar", "pitch", "yaw", "roll", "gx", "gy"]
TEN_DAC_TRUNG_DAI = [
    "ear_mean", "ear_std", "ear_range", "ear_min_rel",       # mức mở mắt và dao động
    "chop_so", "khoang_khong_chop_max",                      # chớp mắt (lần, giây dài nhất không chớp)
    "mar_mean", "mar_max",                                   # ngáp
    "pitch_std", "yaw_std", "roll_std", "dau_toc_do", "pitch_xu_huong",  # đầu
    "nhin_std_x", "nhin_std_y", "nhin_toc_do",               # ánh nhìn
]


def huong_nhin(pts):
    """Hướng nhìn (gx, gy) = vị trí mống mắt trong mắt, trung bình hai mắt.

    gx: 0 = mống mắt sát khóe trái ảnh, 1 = sát khóe phải ảnh (dọc theo trục khóe-khóe).
    gy: lệch lên/xuống so với tâm mắt, chia cho bề rộng mắt. Đo trong hệ của mắt nên ít bị
    ảnh hưởng khi cả đầu di chuyển.
    """
    gx, gy = [], []
    for idx, iris in ((MAT_PHAI, IRIS_PHAI), (MAT_TRAI, IRIS_TRAI)):
        a = pts[idx[0]]                       # khóe trái ảnh của mắt này
        b = pts[idx[3]]                       # khóe phải ảnh
        truc = b - a
        rong = float(np.linalg.norm(truc))
        if rong < 1e-6:
            return float("nan"), float("nan")
        t = float(np.dot(pts[iris] - a, truc) / (rong ** 2))
        gx.append(t)
        gy.append(float((pts[iris][1] - (a[1] + b[1]) / 2.0) / rong))
    return float(np.mean(gx)), float(np.mean(gy))


def _noi_suy_nan(c):
    """Lấp các mẫu NaN bằng nội suy tuyến tính (cửa sổ đã đảm bảo >= 80% có mặt)."""
    c = np.asarray(c, dtype=float).copy()
    x = np.arange(len(c))
    ok = ~np.isnan(c)
    if ok.sum() == 0:
        return c
    c[~ok] = np.interp(x[~ok], x[ok], c[ok])
    return c


def vector_cua_so_dai(cua_so_mau):
    """Vector đặc trưng của MỘT cửa sổ dài.

    cua_so_mau: mảng (n, 7) = [ear, mar, pitch, yaw, roll, gx, gy]; hàng NaN = không thấy mặt.
    Trả về mảng len(TEN_DAC_TRUNG_DAI) số, hoặc None nếu thiếu mặt quá nhiều.
    """
    a = np.asarray(cua_so_mau, dtype=float)
    if len(a) == 0 or (~np.isnan(a).any(axis=1)).mean() < TI_LE_CO_MAT_TOI_THIEU:
        return None
    ear, mar, pitch, yaw, roll, gx, gy = [_noi_suy_nan(a[:, j]) for j in range(7)]
    n = len(a)
    tan_so = float(TAN_SO_MAU)

    # --- chớp mắt / sụp mí: EAR tụt dưới 80% mức "mắt mở" (phân vị 90) của chính cửa sổ ---
    mo = np.percentile(ear, 90)
    thap = ear < TI_LE_DIP_CHOP * mo
    chop = int(np.sum(thap[1:] & ~thap[:-1]) + (1 if thap[0] else 0))
    dai_nhat, hien_tai = 0, 0
    for v in thap:
        if v:
            dai_nhat = max(dai_nhat, hien_tai)
            hien_tai = 0
        else:
            hien_tai += 1
    dai_nhat = max(dai_nhat, hien_tai)

    def toc_do(*cs):                          # trung bình |thay đổi| mỗi giây, cộng các kênh
        return float(sum(np.abs(np.diff(c)).mean() for c in cs) * tan_so)

    xu_huong = float(np.polyfit(np.arange(n) / tan_so, pitch, 1)[0])   # độ / giây

    return np.array([
        ear.mean(), ear.std(), np.percentile(ear, 95) - np.percentile(ear, 5),
        ear.min() / max(mo, 1e-6),
        chop, dai_nhat / tan_so,
        mar.mean(), mar.max(),
        pitch.std(), yaw.std(), roll.std(), toc_do(pitch, yaw, roll), xu_huong,
        gx.std(), gy.std(), toc_do(gx, gy),
    ])


def tao_cac_cua_so_dai(chuoi_mau):
    """Trượt cửa sổ dài trên cả chuỗi mẫu (n, 7). Sinh ra (vị trí mẫu bắt đầu, vector)."""
    chuoi_mau = np.asarray(chuoi_mau, dtype=float)
    for s in range(0, len(chuoi_mau) - CUA_SO_DAI + 1, BUOC_DAI):
        v = vector_cua_so_dai(chuoi_mau[s:s + CUA_SO_DAI])
        if v is not None:
            yield s, v
