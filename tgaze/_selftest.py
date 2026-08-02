"""tgaze パッケージの実画像エンドツーエンド検証。

合成フレームでは顔が取れずキャリブ経路が通らないので、MPIIFaceGaze の実画像を
「較正 → 予測」に流して、公開API だけで妥当な誤差が出ることを確かめる。
(MPIIFaceGaze がある環境でのみ動く。無ければスキップ。)

Usage: .venv/Scripts/python.exe tgaze/_selftest.py
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2, scipy.io as sio
from tgaze import GazeTracker

MPII = ROOT / "MPIIFaceGaze"
if not MPII.exists():
    print("[skip] MPIIFaceGaze が無いのでスキップ"); raise SystemExit(0)

PID = "p00"
ss = sio.loadmat(str(MPII / PID / "Calibration" / "screenSize.mat"))
wpx, hpx = float(np.ravel(ss["width_pixel"])[0]), float(np.ravel(ss["height_pixel"])[0])
wmm, hmm = float(np.ravel(ss["width_mm"])[0]), float(np.ravel(ss["height_mm"])[0])

lines = open(MPII / PID / f"{PID}.txt").read().strip().splitlines()
rs = np.random.RandomState(0)
sel = rs.choice(len(lines), 260, replace=False)
train, test = sel[:200], sel[200:]

def load(i):
    f = lines[i].split()
    img = cv2.imread(str(MPII / PID / f[0]))
    return img, np.array([float(f[1]) / wpx, float(f[2]) / hpx])

t = GazeTracker(video=False)        # 既定=16D+目パッチ / バラバラの静止画なので平滑は切る
ok = 0
for i in train:
    img, tgt = load(i)
    if img is not None and t.add_calibration_sample(img, tgt):
        ok += 1
print(f"較正サンプル: {ok}/{len(train)} 収集")
t.fit()
assert t.is_calibrated, "fit() 後に is_calibrated が False"

errs = []
for i in test:
    img, tgt = load(i)
    if img is None: continue
    p = t.predict(img)
    if not p.face_detected or not np.isfinite(p.x): continue
    d = (np.array([p.x, p.y]) - tgt) * np.array([wmm, hmm]) / 10.0   # 実画面サイズで cm 化
    errs.append(np.linalg.norm(d))

errs = np.array(errs)
print(f"テスト {len(errs)}枚  中央値誤差 = {np.median(errs):.2f} cm  "
      f"(90%tile {np.percentile(errs,90):.2f} cm)")
print(f"較正時 LOO = {t.calibration_error:.4f} (正規化) "
      f"= {t.calibration_error_cm(wmm/10, hmm/10):.2f} cm (この人の実画面)")

# 保存 → 読込 → 同じ予測になるか
mdl = ROOT / "cache" / "_selftest.tgaze"
t.save(mdl)
t2 = GazeTracker(video=False); t2.load(mdl)
img, _ = load(test[0])
a, b = t.predict(img), t2.predict(img)
assert abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9, "save/load で予測が変わった"
print("save/load 一致 OK")
mdl.unlink(missing_ok=True)
t.close(); t2.close()

assert np.median(errs) < 6.0, f"誤差が大きすぎる: {np.median(errs):.2f}cm"
print("[PASS] tgaze 公開APIの実画像エンドツーエンド検証 成功")
