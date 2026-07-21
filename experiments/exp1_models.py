"""7時間耐久 実験1: モデル探索(全姿勢1cmへ)。7D線形の限界を超える手法を系統的に。
ユーザ多姿勢合算データで 点ごとLOO 姿勢bin別 + クロスセッション(汎化=過学習チェック)。
mainのコードは触らず、cache/logs のデータを読むだけ。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
REPORT = ROOT / "experiments" / "REPORT.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

GOOD = ["20260716_130217", "20260717_165617", "20260717_174621"]
Xs, ys, ss = [], [], []
for i, sid in enumerate(GOOD):
    d = np.load(ROOT/"logs"/f"session_{sid}_rich16d.npz"); m = d["has_target"].astype(bool)
    Xs.append(d["X"][m]); ys.append(d["y_norm"][m]); ss.append(np.full(m.sum(), i))
X16 = np.vstack(Xs); Y = np.vstack(ys); SESS = np.concatenate(ss); X7 = X16[:, :7]

def fit_pred(Xtr, ytr, Xte, kind):
    sc = StandardScaler().fit(Xtr); a, b = sc.transform(Xtr), sc.transform(Xte)
    if kind.startswith("MLP"):
        arch = {"MLP-s": (64,32), "MLP-m": (128,64,32), "MLP-l": (256,128,64,32)}[kind]
        m = MLPRegressor(hidden_layer_sizes=arch, max_iter=1500, early_stopping=True, random_state=0).fit(a, ytr)
        return m.predict(b)
    R = {"Huber": lambda: HuberRegressor(max_iter=500), "Ridge": lambda: Ridge(1.0),
         "GBM": lambda: GradientBoostingRegressor(random_state=0),
         "RF": lambda: RandomForestRegressor(120, n_jobs=-1, random_state=0),
         "KNN": lambda: KNeighborsRegressor(20, weights='distance')}[kind]
    mx = R().fit(a, ytr[:,0]); my = R().fit(a, ytr[:,1])
    return np.column_stack([mx.predict(b), my.predict(b)])

def loo_h1(X, Y):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        h = H1Calibration()
        for f_,t in zip(X[tr],Y[tr]): h.add(f_,t[0],t[1],1.0)
        try: h.fit()
        except Exception: continue
        P.append(np.array([h.predict(f_) for f_ in X[te]])); G.append(Y[te]); YD.append(np.abs(np.degrees(X[te][:,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

def loo_eval(X, Y, kind):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        try: pr = fit_pred(X[tr],Y[tr],X[te],kind)
        except Exception: continue
        P.append(pr); G.append(Y[te]); YD.append(np.abs(np.degrees(X[te][:,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

t0 = time.time()
log(f"\n---\n## 実験1: モデル探索 (開始 {int(time.time())})")
log(f"データ: ユーザ多姿勢合算 {len(X16)}フレーム/3セッション。|yaw|>20°={np.mean(np.abs(np.degrees(X16[:,5]))>20):.2f}")
log(f"\n### 点ごとLOO 姿勢bin別 median Euc(cm)   全体 | 0-10 10-20 20-30 30+")
med, bv = loo_h1(X7, Y); log(f"7D  H1(現行)  : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
for feat, Xf in [("7D ", X7), ("16D", X16)]:
    for kind in ["Huber","Ridge","GBM","RF","KNN","MLP-s","MLP-m","MLP-l"]:
        try:
            med, bv = loo_eval(Xf, Y, kind)
            log(f"{feat} {kind:9s}: {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
        except Exception as e:
            log(f"{feat} {kind}: ERR {e}")

# クロスセッション(汎化=過学習チェック): 2セッション学習→1セッション評価
log(f"\n### クロスセッション leave-one-session-out (別条件への汎化=過学習チェック)")
for kind in ["Huber","GBM","MLP-m"]:
    errs = []
    for hold in range(3):
        tr = SESS != hold; te = SESS == hold
        pr = fit_pred(X7[tr], Y[tr], X7[te], kind)
        errs.append(np.median(euc(pr, Y[te])))
    log(f"7D {kind:9s}: 各session保留 median={[f'{e:.1f}' for e in errs]} 平均={np.mean(errs):.2f}cm")
log(f"\n(実験1 完了 {(time.time()-t0)/60:.1f}分)")
