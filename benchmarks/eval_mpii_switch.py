"""実験: キャリブ姿勢から崩れたら MPII推定に「切り替える」(ハードスイッチ, ブレンドでない)。
正面(キャリブ姿勢近傍)=個人H1, |Δyaw|が閾値超=MPII汎用+アフィン。姿勢bin別に検証。
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

d = np.load(ROOT/"cache"/"mpii_7d.npz"); Xm, ym = d["X"], d["y"]
sc = StandardScaler().fit(Xm)
gmx = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:,0])
gmy = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:,1])
def gen(X): return np.column_stack([gmx.predict(sc.transform(X)), gmy.predict(sc.transform(X))])
def affine(bt, yt, be):
    A=np.hstack([bt,np.ones((len(bt),1))]); W,*_=np.linalg.lstsq(A,yt,rcond=None)
    return np.hstack([be,np.ones((len(be),1))])@W
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

THRESH = 15.0  # キャリブ姿勢からの |Δyaw|(deg) がこれ超でMPIIへ切替
print(f"切替閾値 |Δyaw|>{THRESH}° で MPII に切替\n")
print(f"{'session':16s} {'方式':10s} {'全体':>5s}  |y|0-10 10-20 20-30  30+")
rng = np.random.RandomState(0)
for f in sorted(glob.glob(str(ROOT/"logs"/"session_*_rich16d.npz"))):
    d = np.load(f); m = d["has_target"].astype(bool)
    X, y = d["X"][m][:,:7], d["y_norm"][m]
    if len(X) < 200: continue
    ref_yaw = np.median(np.abs(np.degrees(X[:,5])))  # キャリブ姿勢の代表|yaw|
    uniq, ids = np.unique(np.round(y,4), axis=0, return_inverse=True)
    PA,PB,GG,YD,DYAW = [],[],[],[],[]
    for p in np.unique(ids):
        te, tr = ids==p, ids!=p
        if tr.sum() < 30: continue
        h = H1Calibration()
        for f_,t in zip(X[tr], y[tr]): h.add(f_, t[0], t[1], 1.0)
        try: h.fit()
        except Exception: continue
        pa = np.array([h.predict(f_) for f_ in X[te]])
        cal = rng.choice(np.where(tr)[0], min(50,tr.sum()), replace=False)
        pb = affine(gen(X[cal]), y[cal], gen(X[te]))
        PA.append(pa); PB.append(pb); GG.append(y[te])
        yd = np.abs(np.degrees(X[te][:,5])); YD.append(yd)
        DYAW.append(np.abs(yd - ref_yaw))
    PA,PB,GG,YD,DYAW = np.vstack(PA),np.vstack(PB),np.vstack(GG),np.concatenate(YD),np.concatenate(DYAW)
    # ハードスイッチ: キャリブ姿勢から離れたら(=Δyaw>THRESH) MPII
    use_mpii = (DYAW > THRESH)[:,None]
    PS = np.where(use_mpii, PB, PA)
    name = Path(f).name.replace("session_","").replace("_rich16d.npz","")
    for tag, P in [("A:H1",PA),("B:MPII",PB),("S:switch",PS)]:
        e = euc_cm(P, GG)
        print(f"{name if tag=='A:H1' else '':16s} {tag:10s} {np.median(e):5.2f}  " + " ".join(f"{v:5.2f}" for v in binstats(e,YD)))
    print(f"{'':16s} (切替率={use_mpii.mean():.2f})")
    print()
