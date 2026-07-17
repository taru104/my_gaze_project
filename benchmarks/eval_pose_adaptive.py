"""両立の核心: 姿勢適応。姿勢帯ごとに別H1を学習し、推論時に姿勢で選ぶ。
正面は正面データのモデル、横向きは横向きデータのモデル → 各姿勢で最適。
グローバル単一H1(合算)と姿勢bin別に比較。合算多姿勢データで検証。
"""
import sys, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

SCREEN = np.array([30.9,17.4]); BINS=[(0,10),(10,20),(20,30),(30,90)]
ROOT = Path(__file__).parent.parent / "logs"
GOOD = ["20260716_130217","20260717_165617","20260717_174621"]
# 姿勢帯(|yaw| deg)で訓練を分ける。境界は重ねてデータ不足を防ぐ
POSE_BANDS = [(0,18),(12,32),(25,90)]

def euc(P,G): return np.sqrt(((P-G)[:,0]*SCREEN[0])**2+((P-G)[:,1]*SCREEN[1])**2)

def load(sid):
    d=np.load(ROOT/f"session_{sid}_rich16d.npz"); m=d["has_target"].astype(bool)
    return d["X"][m][:,:7], d["y_norm"][m]

def fit_h1(X,y):
    m=H1Calibration()
    for f_,t in zip(X,y): m.add(f_,t[0],t[1],1.0)
    m.fit(); return m

def band_of(yawdeg):
    # そのyawが属する帯index(最も中心に近い)
    for i,(lo,hi) in enumerate(POSE_BANDS):
        if lo<=yawdeg<hi: return i
    return len(POSE_BANDS)-1

def evaluate(X,y,adaptive):
    uniq,ids=np.unique(np.round(y,4),axis=0,return_inverse=True)
    P,G,YD=[],[],[]
    for p in np.unique(ids):
        te,tr=ids==p,ids!=p
        if tr.sum()<20: continue
        ytr_yaw=np.abs(np.degrees(X[tr][:,5]))
        if adaptive:
            models=[]
            for lo,hi in POSE_BANDS:
                sel=(ytr_yaw>=lo)&(ytr_yaw<hi)
                if len(np.unique(np.round(y[tr][sel],4),axis=0))>=5:
                    try: models.append(fit_h1(X[tr][sel],y[tr][sel]))
                    except Exception: models.append(None)
                else: models.append(None)
            glob_m=fit_h1(X[tr],y[tr])
            pr=[]
            for f_ in X[te]:
                b=band_of(abs(np.degrees(f_[5])))
                mm=models[b] if models[b] is not None else glob_m
                pr.append(mm.predict(f_))
            pr=np.array(pr)
        else:
            m=fit_h1(X[tr],y[tr]); pr=np.array([m.predict(f_) for f_ in X[te]])
        P.append(pr); G.append(y[te]); YD.append(np.abs(np.degrees(X[te][:,5])))
    P,G,YD=np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    bv=[np.median(e[(YD>=lo)&(YD<hi)]) if ((YD>=lo)&(YD<hi)).sum() else np.nan for lo,hi in BINS]
    return np.median(e),bv

Xs,ys=[],[]
for sid in GOOD:
    try: X,y=load(sid); Xs.append(X); ys.append(y)
    except FileNotFoundError: print(f"{sid} 無し")
Xa,ya=np.vstack(Xs),np.vstack(ys)
print(f"合算 {len(Xa)}フレーム\n")
print(f"{'手法':22s} {'frame':>6s}  |y|0-10 |y|10-20 |y|20-30 |y|30+")
fr,bv=evaluate(Xa,ya,False)
print(f"{'グローバルH1(単一)':22s} {fr:6.2f}  "+" ".join(f"{v:5.2f}" for v in bv))
fr,bv=evaluate(Xa,ya,True)
print(f"{'★姿勢適応H1(帯別)':20s} {fr:6.2f}  "+" ".join(f"{v:5.2f}" for v in bv))
