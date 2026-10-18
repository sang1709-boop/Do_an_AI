"""
tai_xe.py - Dùng điểm VAI (và cổ tay) từ MediaPipe Pose để biết tài xế có đang ở tư thế lái.

Ý tưởng: khi lái xe, hai tay đặt trên vô lăng nên thân người hướng thẳng về phía trước:
hai vai cùng nhìn thấy, đường vai gần ngang, và vị trí / bề rộng vai gần như không đổi.
Mỗi tài xế ngồi một kiểu và đặt camera mỗi nơi một khác, nên KHÔNG dùng con số tuyệt đối.
Thay vào đó chương trình tự "học" tư thế lái của chính người đang ngồi trong vài giây đầu
(vai thấy rõ liên tiếp SO_MAU_HIEU_CHINH lần), sau đó so sánh với tư thế gốc đó.
Bấm 'c' trong realtime để học lại khi bạn chỉnh ghế / camera.

Cách dùng trong realtime:
    - chon_tai_xe(...)        trong xe có thể có 2 người: chọn người có mũi gần mặt đang theo dõi nhất
    - TheoDoiTuTheLai         hiệu chỉnh + kiểm tra tư thế lái, báo có nên cảnh báo hay không
    - tay_gan_mat(...)        bàn tay có đang che sát một mắt không (dụi mắt)

Tải model Pose (chỉ cần làm một lần):  python tai_xe.py
"""

import math
import os
import urllib.request

import numpy as np

URL_MODEL_POSE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
FILE_POSE = "pose_landmarker_lite.task"

# Chỉ số điểm trong 33 điểm của MediaPipe Pose
MUI = 0
VAI_TRAI, VAI_PHAI = 11, 12
TAY_TRAI = [15, 17, 19, 21]      # cổ tay, út, trỏ, cái
TAY_PHAI = [16, 18, 20, 22]
CO_TAY = [15, 16]

# ---------------- Cấu hình ----------------
VIS_VAI = 0.5                 # độ tin cậy tối thiểu để coi vai là "nhìn thấy"
VIS_TAY = 0.3                 # tay gần mặt thường bị che nên cho thấp hơn
SO_MAU_HIEU_CHINH = 30        # số lần thấy vai tốt cần để chốt tư thế lái gốc
DUNG_SAI_RONG = 0.35          # bề rộng vai được lệch +-35% so với tư thế gốc
DUNG_SAI_TAM = 0.75           # tâm hai vai được dịch tối đa 0.75 lần bề rộng vai
GOC_VAI_TOI_DA = 30.0         # độ nghiêng tối đa của đường nối hai vai (độ)
THOI_GIAN_GIU = 1.5           # giây: lệch tư thế ngắn hơn mức này thì chưa coi là ngừng lái
CHAN_KHI_MAT_VAI = False      # True: mất vai là tạm dừng cảnh báo. False (an toàn): vẫn theo dõi
YEU_CAU_TAY_DUOI_VAI = False  # True: thấy cổ tay nằm TRÊN đường vai thì coi là không đặt vô lăng
TI_LE_TAY_MAT = 0.35          # tay cách tâm mắt < 0.35 bề rộng mặt -> coi là đang che mắt
# ------------------------------------------


def tai_model_pose(duong_dan=FILE_POSE):
    """Có sẵn thì dùng, chưa có thì tải từ Google về cạnh chương trình."""
    if os.path.exists(duong_dan):
        return duong_dan
    print(f"Chua co {duong_dan}, dang tai ve (~6 MB)...")
    try:
        urllib.request.urlretrieve(URL_MODEL_POSE, duong_dan)
    except Exception as loi:                      # mạng lỗi, bị chặn...
        if os.path.exists(duong_dan):
            os.remove(duong_dan)
        raise SystemExit(f"Khong tai duoc ({loi}). Hay tai thu cong file nay roi dat canh "
                         f"realtime.py:\n  {URL_MODEL_POSE}")
    print("Da tai xong.")
    return duong_dan


def tao_pose_detector(duong_dan=FILE_POSE):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=tai_model_pose(duong_dan)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,                               # trong xe có thể có cả hành khách
    )
    return vision.PoseLandmarker.create_from_options(options)


def _px(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


def chon_tai_xe(danh_sach_pose, w, h, tam_mat_px=None):
    """Chọn pose của TÀI XẾ trong số những người phát hiện được.

    Có mặt đang theo dõi: chọn người có mũi gần mặt đó nhất.
    Chưa có mặt: chọn người có vai rộng nhất (ngồi gần camera nhất).
    """
    tot_nhat, diem_tot = None, None
    for pose in danh_sach_pose:
        if tam_mat_px is not None:
            diem = float(np.linalg.norm(_px(pose[MUI], w, h) - tam_mat_px))
        else:
            diem = -float(np.linalg.norm(_px(pose[VAI_TRAI], w, h) - _px(pose[VAI_PHAI], w, h)))
        if diem_tot is None or diem < diem_tot:
            tot_nhat, diem_tot = pose, diem
    return tot_nhat


def tay_gan_mat(pose, pts, idx_mat_p, idx_mat_t, idx_rong, w, h, ti_le=TI_LE_TAY_MAT):
    """(che_phai, che_trai): có điểm bàn tay (cổ tay, ngón) nào nằm sát mắt đó không.

    pts: toạ độ pixel lưới mặt, cùng kích thước ảnh (w, h) đã đưa vào Pose.
    Khoảng cách được đo theo bề rộng mặt nên không phụ thuộc xa / gần camera.
    """
    if pose is None:
        return False, False
    rong_mat = float(np.linalg.norm(pts[idx_rong[0]] - pts[idx_rong[1]]))
    kq = []
    for idx_mat in (idx_mat_p, idx_mat_t):
        tam = pts[idx_mat].mean(axis=0)
        gan = False
        for i in TAY_TRAI + TAY_PHAI:
            lm = pose[i]
            if lm.visibility >= VIS_TAY and np.linalg.norm(_px(lm, w, h) - tam) < ti_le * rong_mat:
                gan = True
                break
        kq.append(gan)
    return kq[0], kq[1]


def so_tay_duoi_vai(pose):
    """(số cổ tay nhìn thấy, trong đó số cổ tay nằm DƯỚI đường vai). Chỉ để hiển thị."""
    if pose is None:
        return 0, 0
    y_vai = (pose[VAI_TRAI].y + pose[VAI_PHAI].y) / 2
    thay = [pose[i] for i in CO_TAY if pose[i].visibility >= VIS_TAY]
    return len(thay), sum(1 for lm in thay if lm.y > y_vai)


class TheoDoiTuTheLai:
    """Học tư thế lái gốc rồi theo dõi. Thuộc tính cần dùng: dang_lai (bool), ly_do (str)."""

    def __init__(self):
        self.dat_lai()

    def dat_lai(self):
        self._mau = []             # (rộng, cx, cy) trong lúc hiệu chỉnh
        self.goc = None            # (rộng, cx, cy) tư thế gốc
        self.dang_lai = True       # chưa có tư thế gốc thì KHÔNG chặn cảnh báo
        self.ly_do = "chua thay vai - chua hieu chinh"
        self.tien_do = 0.0         # 0..1 tiến độ hiệu chỉnh
        self._lan_cuoi_ok = 0.0

    @staticmethod
    def _do_vai(pose, w, h):
        ls, rs = pose[VAI_TRAI], pose[VAI_PHAI]
        if min(ls.visibility, rs.visibility) < VIS_VAI:
            return None
        a, b = _px(ls, w, h), _px(rs, w, h)
        rong = float(np.linalg.norm(a - b))
        if rong < 1e-6:
            return None
        goc = math.degrees(math.atan2(abs(a[1] - b[1]), abs(a[0] - b[0])))
        tam = (a + b) / 2
        return rong, float(tam[0]), float(tam[1]), goc

    def cap_nhat(self, pose, w, h, t):
        """pose: 33 điểm của tài xế (hoặc None); (w, h): kích thước ảnh; t: thời gian (giây).

        Trả về self.dang_lai (True = cho phép cảnh báo).
        """
        do = self._do_vai(pose, w, h) if pose is not None else None

        if do is None:                                           # không thấy vai
            if self.goc is None:
                self.ly_do = "chua thay vai - chua hieu chinh"
                return self.dang_lai
            if CHAN_KHI_MAT_VAI and t - self._lan_cuoi_ok >= THOI_GIAN_GIU:
                self.dang_lai, self.ly_do = False, "mat vai"
            elif not CHAN_KHI_MAT_VAI:
                self.dang_lai, self.ly_do = True, "mat vai - van theo doi"
            return self.dang_lai

        rong, cx, cy, goc_nghieng = do

        if self.goc is None:                                     # đang học tư thế gốc
            if goc_nghieng <= GOC_VAI_TOI_DA:
                self._mau.append((rong, cx, cy))
            self.tien_do = min(1.0, len(self._mau) / SO_MAU_HIEU_CHINH)
            self.dang_lai = True
            self.ly_do = f"dang hieu chinh tu the lai {len(self._mau)}/{SO_MAU_HIEU_CHINH}"
            if len(self._mau) >= SO_MAU_HIEU_CHINH:
                self.goc = tuple(np.median(np.array(self._mau), axis=0))
                self._lan_cuoi_ok = t
                self.ly_do = "dang lai"
            return self.dang_lai

        g_rong, g_cx, g_cy = self.goc
        lech_rong = abs(rong / g_rong - 1.0)
        dich = math.hypot(cx - g_cx, cy - g_cy) / g_rong
        loi = []
        if lech_rong > DUNG_SAI_RONG:
            loi.append(f"vai {'gan' if rong > g_rong else 'xa'} camera")
        if dich > DUNG_SAI_TAM:
            loi.append("lech khoi cho ngoi")
        if goc_nghieng > GOC_VAI_TOI_DA:
            loi.append(f"vai nghieng {goc_nghieng:.0f} do")
        if YEU_CAU_TAY_DUOI_VAI:
            thay, duoi = so_tay_duoi_vai(pose)
            if thay > duoi:
                loi.append("tay gio cao")

        if not loi:
            self._lan_cuoi_ok = t
            self.dang_lai, self.ly_do = True, "dang lai"
        elif t - self._lan_cuoi_ok >= THOI_GIAN_GIU:
            self.dang_lai, self.ly_do = False, "khong o tu the lai: " + ", ".join(loi)
        return self.dang_lai


if __name__ == "__main__":
    tai_model_pose()
    print("San sang:", FILE_POSE)
