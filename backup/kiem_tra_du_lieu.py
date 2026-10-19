"""
kiem_tra_du_lieu.py - Kiểm tra dữ liệu CSV có "hợp lý" không, theo từng người và từng video.

Chạy:  python kiem_tra_du_lieu.py
Ý tưởng: ở MỌI người, clip buồn ngủ thường có EAR thấp hơn (mắt nhắm nhiều),
EAR dao động nhiều hơn (std cao). Người nào làm ngược lại -> nghi ngờ nhầm nhãn
hoặc quay khác điều kiện.
"""

import pandas as pd

FILE_CSV = "du_lieu_dac_trung.csv"
COT = ["ear_mean", "ear_min", "ear_std", "mar_max", "pitch_mean", "yaw_mean"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

df = pd.read_csv(FILE_CSV).dropna()

print("=" * 80)
print("1. TRUNG BINH THEO NGUOI VA NHAN  (nhan 0 = tinh tao, 1 = buon ngu)")
bang = df.groupby(["subject", "label"])[COT].mean().round(3)
bang.insert(0, "so_cua_so", df.groupby(["subject", "label"]).size())
print(bang)

print("\n" + "=" * 80)
print("2. CHENH LECH (buon ngu - tinh tao) CUA TUNG NGUOI")
print("   Ky vong: ear_mean < 0, ear_min < 0, ear_std > 0")
dong = []
for s in sorted(df["subject"].unique()):
    a = df[(df["subject"] == s) & (df["label"] == 0)]
    b = df[(df["subject"] == s) & (df["label"] == 1)]
    if len(a) == 0 or len(b) == 0:
        print(f"   {s}: chi co 1 nhan -> khong so sanh duoc")
        continue
    d = (b[COT].mean() - a[COT].mean()).round(3)
    d.name = s
    dong.append(d)
chenh = pd.DataFrame(dong)
chenh["canh_bao"] = ["<-- NGUOC KY VONG" if v >= 0 else "" for v in chenh["ear_mean"]]
print(chenh)

print("\n" + "=" * 80)
print("3. TUNG VIDEO")
v = df.groupby(["subject", "video", "label"]).agg(
    so_cua_so=("label", "size"),
    ear_mean=("ear_mean", "mean"),
    ear_min=("ear_min", "mean"),
    mar_max=("mar_max", "mean"),
).round(3)
print(v)

print("\nGoi y doc ket qua:")
print(" - Nguoi nao bi canh bao: mo lai video cua nguoi do, kiem tra clip co nam dung thu muc")
print("   tinh_tao / buon_ngu khong, va co bi quay khac anh sang / tu the / goc may khong.")
print(" - Video nao co ear_mean rat khac cac video cung nhan: xem lai video do.")