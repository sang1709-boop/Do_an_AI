"""
anh_sang.py - Chế độ NGÀY / ĐÊM và xử lý kính loá.

Dùng chung cho tao_du_lieu.py (huấn luyện) và realtime.py (chạy thật), để ảnh đưa vào
MediaPipe được xử lý GIỐNG HỆT nhau ở hai nơi.

Việc làm:
  1. Đo độ sáng khung hình (trung vị mức xám, làm mượt) -> tự chọn chế độ NGAY / DEM
     (có "độ trễ" hai mức để không nhấp nháy). Bấm 'n' trong realtime để ép chế độ.
  2. Chế độ ĐÊM: kéo sáng bằng gamma, cân bằng tương phản CLAHE, lọc nhiễu nhẹ.
     Ngày quá chói/ngược sáng: hạ sáng bằng gamma.
  3. Kính loá: các đốm trắng cháy nhỏ (phản chiếu trên mắt kính) được xoá bằng inpaint
     trước khi đưa vào MediaPipe. Sau đó đo riêng tỉ lệ pixel cháy quanh từng mắt trên
     ẢNH GỐC; mắt nào loá nhiều sẽ không được tin (xem dac_trung.gop_ear).

Chạy riêng để lấy số liệu chỉnh ngưỡng sáng từ chính video của bạn:
    python anh_sang.py do  video_ngay.mp4  video_dem.mp4
"""

import sys

import cv2
import numpy as np

# ---------------- Cấu hình (thang mức xám 0..255) ----------------
SANG_VAO_DEM = 70       # độ sáng tụt dưới mức này  -> sang chế độ ĐÊM
SANG_RA_DEM = 95        # độ sáng lên trên mức này  -> về chế độ NGÀY (khác mức vào = có "trễ")
SANG_QUA_CHOI = 175     # sáng hơn mức này (ngược sáng / nắng gắt) -> hạ sáng
SANG_MUC_TIEU = 115     # độ sáng ta muốn đưa ảnh về
HE_SO_LAM_MUOT = 0.1    # làm mượt độ sáng theo thời gian (nhỏ = mượt hơn)

NGUONG_PIXEL_CHAY = 250     # pixel >= mức này coi là "cháy sáng"
TI_LE_CHAY_TOI_DA = 0.03    # nếu quá 3% khung hình cháy (cửa sổ, đèn pha...) thì KHÔNG inpaint
NGUONG_LOA_MAT = 0.10       # >= 10% pixel quanh một mắt bị cháy -> mắt đó bị loá, không tin EAR
# -----------------------------------------------------------------


class BoXuLyAnh:
    """Giữ trạng thái độ sáng / chế độ ngày-đêm. Tạo MỚI cho mỗi video khi huấn luyện."""

    def __init__(self, che_do="auto", giam_loa=True):
        self.che_do_chon = che_do        # "auto" | "ngay" | "dem"
        self.giam_loa = giam_loa
        self.do_sang = None              # độ sáng đã làm mượt
        self._dem = False                # trạng thái ĐÊM khi ở chế độ auto
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gray_goc = None             # ảnh xám GỐC (trước khi xử lý), để đo loá

    # ------------------------------------------------------------
    @property
    def che_do(self):
        """Chế độ đang áp dụng: "ngay" hoặc "dem"."""
        if self.che_do_chon == "auto":
            return "dem" if self._dem else "ngay"
        return self.che_do_chon

    def doi_che_do(self):
        """Xoay vòng: auto -> ngay -> dem -> auto (cho phím 'n')."""
        thu_tu = ["auto", "ngay", "dem"]
        self.che_do_chon = thu_tu[(thu_tu.index(self.che_do_chon) + 1) % 3]
        return self.che_do_chon

    # ------------------------------------------------------------
    def _cap_nhat_do_sang(self, gray):
        do = float(np.median(gray[::4, ::4]))
        if self.do_sang is None:
            self.do_sang = do
        else:
            self.do_sang += HE_SO_LAM_MUOT * (do - self.do_sang)
        if self.che_do_chon == "auto":
            if not self._dem and self.do_sang < SANG_VAO_DEM:
                self._dem = True
            elif self._dem and self.do_sang > SANG_RA_DEM:
                self._dem = False

    @staticmethod
    def _gamma(anh, gamma):
        bang = (255.0 * (np.arange(256) / 255.0) ** gamma).clip(0, 255).astype(np.uint8)
        return cv2.LUT(anh, bang)

    def _xoa_loa(self, anh, gray):
        """Xoá các đốm cháy sáng NHỎ (phản chiếu kính). Bỏ qua nếu vùng cháy quá lớn."""
        mask = (gray >= NGUONG_PIXEL_CHAY).astype(np.uint8) * 255
        ti_le = float(mask.mean()) / 255.0
        if ti_le == 0.0 or ti_le > TI_LE_CHAY_TOI_DA:
            return anh
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        return cv2.inpaint(anh, mask, 3, cv2.INPAINT_TELEA)

    def xu_ly(self, frame):
        """Nhận khung BGR, trả khung BGR đã chuẩn hoá ánh sáng (cùng kích thước)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray_goc = gray
        self._cap_nhat_do_sang(gray)

        anh = frame
        if self.giam_loa:
            anh = self._xoa_loa(anh, gray)

        if self.che_do == "dem":
            # kéo độ sáng về mức mục tiêu (gamma < 1 làm sáng), rồi cân bằng tương phản
            do = min(max(self.do_sang, 5.0), 250.0)
            gamma = np.log(SANG_MUC_TIEU / 255.0) / np.log(do / 255.0)
            anh = self._gamma(anh, float(np.clip(gamma, 0.4, 1.0)))
            lab = cv2.cvtColor(anh, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            anh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            anh = cv2.GaussianBlur(anh, (3, 3), 0)        # giảm nhiễu hạt do kéo sáng
        elif self.do_sang > SANG_QUA_CHOI:
            gamma = np.log(SANG_MUC_TIEU / 255.0) / np.log(min(self.do_sang, 250.0) / 255.0)
            anh = self._gamma(anh, float(np.clip(gamma, 1.0, 2.0)))   # gamma > 1 làm tối
        return anh

    # ------------------------------------------------------------
    def ti_le_loa(self, pts, idx, le=0.1):
        """Tỉ lệ pixel cháy sáng (trên ảnh GỐC) trong hộp bao quanh mắt.

        pts: toạ độ pixel (N, 2) của lưới landmark, idx: 6 chỉ số điểm của một mắt.
        """
        if self.gray_goc is None:
            return 0.0
        h, w = self.gray_goc.shape
        p = pts[idx]
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        dx, dy = (x1 - x0) * le, (y1 - y0) * le
        x0, x1 = int(max(0, x0 - dx)), int(min(w, x1 + dx + 1))
        y0, y1 = int(max(0, y0 - dy)), int(min(h, y1 + dy + 1))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float((self.gray_goc[y0:y1, x0:x1] >= NGUONG_PIXEL_CHAY).mean())


# =====================================================================
# Tiện ích: đo độ sáng của video để chỉnh SANG_VAO_DEM / SANG_RA_DEM
# =====================================================================
def _do_video(duong_dan):
    cap = cv2.VideoCapture(str(duong_dan))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    buoc = int(round(fps)) if fps and fps == fps and fps >= 1 else 15
    kq, i = [], 0
    while cap.grab():
        if i % buoc == 0:
            ok, f = cap.retrieve()
            if ok:
                kq.append(float(np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[::4, ::4])))
        i += 1
    cap.release()
    return np.array(kq)


def do_cac_video(ngay, dem):
    a = [x for x in (_do_video(p) for p in ngay) if x is not None and len(x)]
    b = [x for x in (_do_video(p) for p in dem) if x is not None and len(x)]
    if not a or not b:
        raise SystemExit("Can it nhat 1 video ngay va 1 video dem doc duoc.")
    a, b = np.concatenate(a), np.concatenate(b)
    print(f"NGAY: P5={np.percentile(a, 5):.0f}  trung vi={np.median(a):.0f}  P95={np.percentile(a, 95):.0f}")
    print(f"DEM : P5={np.percentile(b, 5):.0f}  trung vi={np.median(b):.0f}  P95={np.percentile(b, 95):.0f}")
    thap, cao = np.percentile(b, 95), np.percentile(a, 5)
    if cao <= thap:
        print("Do sang ngay va dem CHONG LAN -> khong tach duoc bang do sang; hay quay lai video dem toi hon.")
        return
    giua = (thap + cao) / 2
    nua = (cao - thap) / 4
    print(f"\nGoi y: SANG_VAO_DEM = {giua - nua:.0f}, SANG_RA_DEM = {giua + nua:.0f}  "
          f"(trung diem giua P95 dem va P5 ngay, +-1/4 khoang cach)")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "do":
        # python anh_sang.py do ngay1.mp4[,ngay2.mp4] dem1.mp4[,dem2.mp4]
        do_cac_video(sys.argv[2].split(","), sys.argv[3].split(","))
    else:
        print(__doc__)
