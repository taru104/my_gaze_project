"""7時間耐久 実験2: exp1で最良だった 16D を軸に全姿勢1cmへ深掘り。
16D Huberハイパラ / MLP深層 / poly2非線形 / 特徴アブレーション(横向きに効く次元) を姿勢bin別に。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
REPORT = ROOT / "experiments" / "REPORT.md"
NAMES16 = ["Lx","Ly","Rx","Ry","pitch","yaw","dist","roll","L_EAR","R_EAR","L_ivert","R_ivert","L_idiam","R_idiam","L_asp","R_asp"]
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

def loo(X, Y, make, poly=False):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        Xtr, Xte = X[tr], X[te]
        sc = StandardScaler().fit(Xtr); a,b = sc.transform(Xtr), sc.transform(Xte)
        if poly:
            pf = PolynomialFeatures(2, include_bias=False); a = pf.fit_transform(a); b = pf.transform(b)
        m = make()
        if hasattr(m, "predict") and "MLP" in type(m).__name__:
            m.fit(a, Y[tr]); pr = m.predict(b)
        else:
            mx = make().fit(a, Y[tr,0]); my = make().fit(a, Y[tr,1]); pr = np.column_stack([mx.predict(b), my.predict(b)])
        P.append(pr); G.append(Y[te]); YD.append(np.abs(np.degrees(X[te][:,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

t0 = time.time()
log(f"\n---\n## 実験2: 16D深掘り (開始)")
log(f"### 16D Huber ハイパラ   全体|0-10 10-20 20-30 30+")
best = (99, None)
for eps in [1.1, 1.35, 2.0]:
    for al in [1e-4, 1e-3, 1e-2]:
        med, bv = loo(X16, Y, lambda eps=eps, al=al: HuberRegressor(epsilon=eps, alpha=al, max_iter=800))
        log(f"  eps={eps} a={al}: {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
        if med < best[0]: best = (med, f"eps={eps},a={al}")
log(f"  → 最良: {best[1]} ({best[0]:.2f}cm)")

log(f"\n### 16D MLP (深層で非線形)   全体|0-10 10-20 20-30 30+")
for arch in [(64,32),(128,64),(128,64,32),(256,128,64)]:
    for al in [1e-3, 1e-2]:
        try:
            med, bv = loo(X16, Y, lambda arch=arch, al=al: MLPRegressor(hidden_layer_sizes=arch, alpha=al, max_iter=2000, early_stopping=True, random_state=0))
            log(f"  {str(arch):16s} a={al}: {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
        except Exception as e:
            log(f"  {arch} a={al}: ERR {e}")

log(f"\n### 16D + poly2 (非線形展開) Huber/Ridge")
for nm, mk in [("Huber", lambda: HuberRegressor(max_iter=800)), ("Ridge", lambda: Ridge(1.0))]:
    med, bv = loo(X16, Y, mk, poly=True)
    log(f"  poly2 {nm}: {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))

log(f"\n### 16D 特徴アブレーション (1つ抜いて 30+ がどう悪化=横向き寄与)")
base_med, base_bv = loo(X16, Y, lambda: HuberRegressor(max_iter=800))
rows = []
for i in range(16):
    cols = [c for c in range(16) if c != i]
    med, bv = loo(X16[:, cols], Y, lambda: HuberRegressor(max_iter=800))
    d30 = (bv[3] - base_bv[3]) if not (np.isnan(bv[3]) or np.isnan(base_bv[3])) else 0
    rows.append((NAMES16[i], bv[3], d30))
for n, v30, d in sorted(rows, key=lambda r: -r[2])[:8]:
    log(f"  -{n:8s} 30+={v30:5.2f} (base {base_bv[3]:.2f}, 悪化{d:+.2f})")
log(f"\n(実験2 完了 {(time.time()-t0)/60:.1f}分)")
