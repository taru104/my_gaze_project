"""7時間耐久 実験5: 統合・過学習チェック・最終結論。
(1) 16D Huber eps アンサンブル(全体最良と横向き最良の両取り)
(2) クロスセッション leave-one-session-out (過学習チェック=別条件でも良いか)
(3) MPII(7D)結合: ユーザ7D + MPII7D を結合学習 → ユーザ評価(汎用データで底上げされるか)
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

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

def H(eps=1.5): return HuberRegressor(epsilon=eps, alpha=1e-3, max_iter=800)
def fit_xy(Xtr, Ytr, Xte, eps=1.5):
    sc = StandardScaler().fit(Xtr); a,b = sc.transform(Xtr), sc.transform(Xte)
    mx = H(eps).fit(a, Ytr[:,0]); my = H(eps).fit(a, Ytr[:,1])
    return np.column_stack([mx.predict(b), my.predict(b)])

def loo_ens(X, Y, epslist):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        preds = [fit_xy(X[tr],Y[tr],X[te],eps) for eps in epslist]
        P.append(np.mean(preds, axis=0)); G.append(Y[te]); YD.append(np.abs(np.degrees(X16[te,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

t0=time.time()
log(f"\n---\n## 実験5: 統合・過学習チェック・結論  全体|0-10 10-20 20-30 30+")
# (1) アンサンブル
for name, epsl in [("単一eps1.1",[1.1]),("単一eps2.0",[2.0]),("アンサンブル1.1+1.5+2.0",[1.1,1.5,2.0])]:
    med,bv = loo_ens(X16, Y, epsl); log(f"  16D {name:20s}: {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))

# (2) クロスセッション(過学習チェック): 2セッション学習→1セッション評価
log(f"\n### クロスセッション leave-one-session-out (別条件汎化=過学習チェック)")
for feat, Xf in [("7D ", X7), ("16D", X16)]:
    errs, b30 = [], []
    for hold in range(3):
        tr = SESS!=hold; te = SESS==hold
        pr = fit_xy(Xf[tr], Y[tr], Xf[te])
        e = euc(pr, Y[te]); yd = np.abs(np.degrees(X16[te,5]))
        errs.append(np.median(e)); bv = binstats(e, yd); b30.append(bv[3])
    log(f"  {feat} Huber: 各session保留 median={[f'{e:.1f}' for e in errs]} 平均={np.mean(errs):.2f}cm  30+平均={np.nanmean(b30):.2f}")

# (3) MPII(7D)結合: ユーザ点ごとLOOの訓練にMPIIを足す
log(f"\n### MPII(7D)結合: ユーザ訓練にMPII 37k を足す(汎用データで底上げ?)")
dm = np.load(ROOT/"cache"/"mpii_7d.npz"); Xm, ym = dm["X"], dm["y"]
uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
for tag, add_mpii in [("ユーザのみ", False), ("ユーザ+MPII", True)]:
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        if add_mpii:
            Xtr = np.vstack([X7[tr], Xm]); Ytr = np.vstack([Y[tr], ym])
        else:
            Xtr, Ytr = X7[tr], Y[tr]
        pr = fit_xy(Xtr, Ytr, X7[te])
        P.append(pr); G.append(Y[te]); YD.append(np.abs(np.degrees(X16[te,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    log(f"  7D {tag:12s}: {np.median(e):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(e,YD)))
log(f"\n(実験5 完了 {(time.time()-t0)/60:.1f}分)")
