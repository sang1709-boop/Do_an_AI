"""
dac_trung.py - Các hàm tính đặc trưng hình học từ điểm mốc khuôn mặt MediaPipe.

File này KHÔNG cần webcam, chỉ cần NumPy. Các bước sau (trích xuất dữ liệu,
chạy realtime) sẽ import lại file này:  from dac_trung import ...
"""

import numpy as np

# ---- Chỉ số (index) các điểm mốc trong lưới 478 điểm của MediaPipe ----
# Mắt: 6 điểm theo thứ tự p1..p6 của công thức EAR
#   p1, p4 = hai khóe mắt (ngang);  p2, p3 = mi trên;  p6, p5 = mi dưới
MAT_PHAI = [33, 160, 158, 133, 153, 144]
MAT_TRAI = [362, 385, 387, 263, 373, 380]

# Miệng (viền môi trong): 3 cặp điểm dọc + 1 cặp khóe miệng (ngang)
MIENG_DOC = [(82, 87), (13, 14), (312, 317)]
MIENG_NGANG = (78, 308)


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