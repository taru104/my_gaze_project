"""実験1: MPIIの多姿勢傾向で、ユーザの横向き(データ不足)を補えるか。7D線形のまま。
姿勢bin別に3方式を比較:
  A: user個人H1 (点ごとLOO)                     … 正面◎ 横向き弱い
  B: MPII汎用 + userアフィン適応(50点)          … MPIIは横向きも学習済
  C: ゲートブレンド (正面=A, 横向き=B, yawで連続) … 両立狙い
"""
import sys, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
def euc_cm(P, G): return np.linalg.norm((P - G) * CM, axis=1)

# MPII汎用モデル
d = np.load(ROOT/"cache"/"mpii_7d.npz"); Xm, ym = d["X"], d["y"]
sc = StandardScaler().fit(Xm)
gmx = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:,0])
gmy = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:,1])
def gen(X): return np.column_stack([gmx.predict(sc.transform(X)), gmy.predict(sc.transform(X))])

def affine(bt, yt, be):
    A = np.hstack([bt, np.ones((len(bt),1))]); W,*_ = np.linalg.lstsq(A, yt, rcond=None)
    return np.hstack([be, np.ones((len(be),1))]) @ W

def binstats(e, yd):
    return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

def run(f):
    d = np.load(f); m = d["has_target"].astype(bool)
    X, y = d["X"][m][:,:7], d["y_norm"][m]
    if len(X) < 200: return None
    uniq, ids = np.unique(np.round(y,4), axis=0, return_inverse=True)
    rng = np.random.RandomState(0)
    PA, PB, GG, YD = [], [], [], []
    # userH1 は点ごとLOO / MPII+affine は同じ評価点で
    for p in np.unique(ids):
        te, tr = ids==p, ids!=p
        if tr.sum() < 30: continue
        # A: user H1
        h = H1Calibration()
        for f_,t in zip(X[tr], y[tr]): h.add(f_, t[0], t[1], 1.0)
        try: h.fit()
        except Exception: continue
        pa = np.array([h.predict(f_) for f_ in X[te]])
        # B: MPII汎用 + userアフィン(trの50点で適応)
        cal = rng.choice(np.where(tr)[0], min(50, tr.sum()), replace=False)
        pb = affine(gen(X[cal]), y[cal], gen(X[te]))
        PA.append(pa); PB.append(pb); GG.append(y[te]); YD.append(np.abs(np.degrees(X[te][:,5])))
    PA, PB, GG, YD = np.vstack(PA), np.vstack(PB), np.vstack(GG), np.concatenate(YD)
    # C: yawゲート(正面=A, 横向き=B)
    w = np.exp(-np.clip(YD-12,0,None)/12)[:,None]  # 正面でA重視
    PC = w*PA + (1-w)*PB
    eA, eB, eC = euc_cm(PA,GG), euc_cm(PB,GG), euc_cm(PC,GG)
    return binstats(eA,YD), binstats(eB,YD), binstats(eC,YD), np.median(eA), np.median(eB), np.median(eC)

print(f"{'session':16s} {'方式':6s} {'全体':>5s}  |y|0-10 10-20 20-30  30+")
for f in sorted(glob.glob(str(ROOT/"logs"/"session_*_rich16d.npz"))):
    r = run(f)
    if r is None: continue
    (bA,bB,bC,mA,mB,mC) = r
    name = Path(f).name.replace("session_","").replace("_rich16d.npz","")
    for tag,b,m in [("A:H1",bA,mA),("B:MPII",bB,mB),("C:blend",bC,mC)]:
        print(f"{name if tag=='A:H1' else '':16s} {tag:7s} {m:5.2f}  " + " ".join(f"{v:5.2f}" for v in b))
    print()
print("→ 30+でB(MPII)がA(H1)より良ければ、C(ブレンド)で正面◎横向き改善の両立。")
