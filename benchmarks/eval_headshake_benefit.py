"""「正面だけキャリブ」vs「首振りキャリブ」で横向き精度がどう変わるか。
同じ多姿勢セッションを 各点で前半=キャリブ / 後半=評価 に分け、キャリブに使う姿勢だけ変える:
  A: 正面キャリブ  … 前半のうち |yaw|<12° のフレームだけでH1
  B: 首振りキャリブ … 前半の全フレーム(全姿勢)でH1
評価は共通(後半・全姿勢)。姿勢bin別 median Euc(cm)。
"""
import sys, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
def euc_cm(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

def fit_h1(X, y):
    h = H1Calibration()
    for f_, t in zip(X, y): h.add(f_, t[0], t[1], 1.0)
    h.fit(); return h

print(f"{'session':16s} {'ｷｬﾘﾌﾞ方式':14s} {'全体':>5s}  |y|0-10 10-20 20-30  30+")
for f in sorted(glob.glob(str(ROOT/"logs"/"session_*_rich16d.npz"))):
    dd = np.load(f); m = dd["has_target"].astype(bool)
    X, y, ts = dd["X"][m][:,:7], dd["y_norm"][m], dd["time_s"][m]
    if len(X) < 300: continue
    yawd = np.abs(np.degrees(X[:,5]))
    if np.mean(yawd > 20) < 0.15: continue   # 横向きが十分あるセッションのみ(=首振り録画)
    uniq, ids = np.unique(np.round(y,4), axis=0, return_inverse=True)
    # 各点 前半=キャリブ/後半=評価
    trm = np.zeros(len(X), bool)
    for p in np.unique(ids):
        s = np.where(ids==p)[0]; order = s[np.argsort(ts[s])]
        trm[order[:len(order)//2]] = True
    tem = ~trm
    front = trm & (yawd < 12)   # 正面キャリブ用
    name = Path(f).name.replace("session_","").replace("_rich16d.npz","")
    Xe, ye, yde = X[tem], y[tem], yawd[tem]
    try:
        hA = fit_h1(X[front], y[front])   # 正面のみ
        hB = fit_h1(X[trm], y[trm])       # 首振り(全姿勢)
    except Exception:
        continue
    for tag, h in [("A:正面のみ", hA), ("B:首振り", hB)]:
        pr = np.array([h.predict(f_) for f_ in Xe])
        e = euc_cm(pr, ye)
        print(f"{name if tag=='A:正面のみ' else '':16s} {tag:14s} {np.median(e):5.2f}  " + " ".join(f"{v:5.2f}" for v in binstats(e, yde)))
    print(f"{'':16s} (正面ｷｬﾘﾌﾞ枚数={front.sum()}, 首振りｷｬﾘﾌﾞ枚数={trm.sum()})")
    print()
print("→ B(首振り)が横向き(20-30/30+)でAより大幅に良ければ、首振りキャリブが横向き崩壊を防ぐ実証。")
