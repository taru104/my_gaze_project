"""良質な多姿勢セッション('m'モード)を合算し、H1で全姿勢の現在地を測る。
点ごとLOO・姿勢bin別。ハッカソンの現在地把握用。
"""
import sys, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

SCREEN = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
ROOT = Path(__file__).parent.parent / "logs"

# 良質な多姿勢セッション('m'モード, loo<4cm)
GOOD = ["20260716_130217", "20260717_165617", "20260717_174621"]

def euc(P,G): return np.sqrt(((P-G)[:,0]*SCREEN[0])**2+((P-G)[:,1]*SCREEN[1])**2)

def load(sid):
    d = np.load(ROOT / f"session_{sid}_rich16d.npz")
    m = d["has_target"].astype(bool)
    return d["X"][m][:, :7], d["y_norm"][m]

def loo_h1(X, y, label):
    uniq, ids = np.unique(np.round(y,4), axis=0, return_inverse=True)
    PM, GM, YD = [], [], []
    for p in np.unique(ids):
        te, tr = ids==p, ids!=p
        if tr.sum() < 20: continue
        m = H1Calibration()
        for feat, tgt in zip(X[tr], y[tr]):
            m.add(feat, tgt[0], tgt[1], 1.0)
        try: m.fit()
        except Exception: continue
        pr = np.array([m.predict(f) for f in X[te]])
        PM.append(np.median(pr,axis=0)); GM.append(np.median(y[te],axis=0))
        # bin用にフレーム単位も
        YD.append((euc(pr, y[te]), np.abs(np.degrees(X[te][:,5]))))
    e_pt = euc(np.array(PM), np.array(GM))
    allе = np.concatenate([a for a,_ in YD]); ally = np.concatenate([b for _,b in YD])
    binv = [np.median(allе[(ally>=lo)&(ally<hi)]) if ((ally>=lo)&(ally<hi)).sum() else np.nan for lo,hi in BINS]
    print(f"  {label:26s} 実効={np.median(e_pt):5.2f}cm frame={np.median(allе):5.2f}  " +
          " ".join(f"{v:5.2f}" for v in binv))

print("H1 点ごとLOO 姿勢bin別 median Euc(cm)   |y|0-10 |y|10-20 |y|20-30 |y|30+")
Xs, ys = [], []
for sid in GOOD:
    try:
        X,y = load(sid); Xs.append(X); ys.append(y)
        loo_h1(X, y, f"{sid} 単体")
    except FileNotFoundError:
        print(f"  {sid}: npz無し(要reprocess)")
if len(Xs) >= 2:
    Xa, ya = np.vstack(Xs), np.vstack(ys)
    print()
    loo_h1(Xa, ya, f"★合算{len(Xs)}セッション")
    print(f"\n  合算 {len(Xa)}フレーム |yaw|>20°={np.mean(np.abs(np.degrees(Xa[:,5]))>20):.2f} |yaw|>30°={np.mean(np.abs(np.degrees(Xa[:,5]))>30):.2f}")
