"""7時間耐久 実験3: 特徴エンジニアリング + 姿勢条件モデルで全姿勢1cmへ。
16D Huber(30+ 4.17cm)を軸に:
  (1) 虹彩×頭部姿勢の交互作用項を追加(横向きで虹彩の見え方が変わるのを線形で捉える)
  (2) 姿勢帯ごと専用Huber(yaw帯別モデル、推論時に姿勢で選択)
  (3) 虹彩のみpoly2(眼球回転の非線形)
全て線形Huber(過学習しない)。点ごとLOO 姿勢bin別。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
REPORT = ROOT / "experiments" / "REPORT.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

GOOD = ["20260716_130217", "20260717_165617", "20260717_174621"]
Xs, ys = [], []
for sid in GOOD:
    d = np.load(ROOT/"logs"/f"session_{sid}_rich16d.npz"); m = d["has_target"].astype(bool)
    Xs.append(d["X"][m]); ys.append(d["y_norm"][m])
X16 = np.vstack(Xs); Y = np.vstack(ys)

def mkH(): return HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=800)

def loo(X, Y):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        sc = StandardScaler().fit(X[tr]); a,b = sc.transform(X[tr]), sc.transform(X[te])
        mx = mkH().fit(a, Y[tr,0]); my = mkH().fit(a, Y[tr,1])
        P.append(np.column_stack([mx.predict(b), my.predict(b)])); G.append(Y[te]); YD.append(np.abs(np.degrees(X16[te,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

BANDS = [(0,18),(12,32),(25,90)]
def band(y):
    for i,(lo,hi) in enumerate(BANDS):
        if lo<=y<hi: return i
    return len(BANDS)-1
def loo_posecond(X, Y):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        yawtr = np.abs(np.degrees(X[tr,5]))
        scg = StandardScaler().fit(X[tr]); gx = mkH().fit(scg.transform(X[tr]),Y[tr,0]); gy = mkH().fit(scg.transform(X[tr]),Y[tr,1])
        mods = []
        for lo,hi in BANDS:
            sel = (yawtr>=lo)&(yawtr<hi)
            if len(np.unique(np.round(Y[tr][sel],4),axis=0))>=5:
                sc = StandardScaler().fit(X[tr][sel])
                mods.append((sc, mkH().fit(sc.transform(X[tr][sel]),Y[tr][sel,0]), mkH().fit(sc.transform(X[tr][sel]),Y[tr][sel,1])))
            else: mods.append(None)
        pr=[]
        for f_ in X[te]:
            bi = band(abs(np.degrees(f_[5])))
            if mods[bi] is not None:
                sc,mx,my = mods[bi]; ff=sc.transform(f_.reshape(1,-1)); pr.append([mx.predict(ff)[0],my.predict(ff)[0]])
            else:
                ff=scg.transform(f_.reshape(1,-1)); pr.append([gx.predict(ff)[0],gy.predict(ff)[0]])
        P.append(np.array(pr)); G.append(Y[te]); YD.append(np.abs(np.degrees(X16[te,5])))
    P,G,YD=np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

t0=time.time()
log(f"\n---\n## 実験3: 特徴拡張 + 姿勢条件 (開始)  全体|0-10 10-20 20-30 30+")
med,bv = loo(X16, Y); log(f"  16D baseline       : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
# (1) 虹彩×姿勢 交互作用
iris = X16[:,:4]; pit=X16[:,4:5]; yaw=X16[:,5:6]; dist=X16[:,6:7]
inter = np.hstack([iris*yaw, iris*pit, iris*dist])
Xext = np.hstack([X16, inter])
med,bv = loo(Xext, Y); log(f"  16D+虹彩×姿勢交互(28D): {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
# (2) 虹彩のみpoly2 + 残り16D
pf = PolynomialFeatures(2, include_bias=False)
iris_poly = pf.fit_transform(iris)
Xip = np.hstack([iris_poly, X16[:,4:]])
med,bv = loo(Xip, Y); log(f"  虹彩poly2+姿勢12D    : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
# (3) 姿勢条件
med,bv = loo_posecond(X16, Y); log(f"  姿勢条件(yaw帯別16D) : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
# (4) 交互作用 + 姿勢条件
med,bv = loo_posecond(Xext, Y); log(f"  28D交互+姿勢条件     : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
log(f"\n(実験3 完了 {(time.time()-t0)/60:.1f}分)")
