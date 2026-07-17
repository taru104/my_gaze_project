"""1cm を目指す精度探索ハーネス（自律実行）。

既存の生ランドマーク由来7D特徴(session_*_rich16d.npz)で、多数の手法を
**点ごと leave-one-point-out** で評価し、median Euclidean(cm) を最小化する。
次元は増やさない方針(ユーザ指示)＝7D以下の特徴のみ。中身(モデル/前処理/後処理)で詰める。

結果は results/exploration_log.md に追記(逐次)。stdout にも表示。

Usage:
    .venv/Scripts/python.exe benchmarks/explore_accuracy.py [session_id] [stage]
    stage: models | features | preprocess | ncalib | all(default)
"""
import sys, json
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

SCREEN = np.array([30.9, 17.4])
CALIB_DISCARD = 1.0
ROOT = Path(__file__).parent.parent
LOG = ROOT / "results" / "exploration_log.md"


def load(sid):
    d = np.load(ROOT / "logs" / f"session_{sid}_rich16d.npz")
    X, y, ht, ts = d["X"], d["y_norm"], d["has_target"], d["time_s"]
    m = ht.copy()
    X, y, ts = X[m], y[m], ts[m]
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    keep = np.zeros(len(X), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        keep[sel[ts[sel] - ts[sel].min() >= CALIB_DISCARD]] = True
    return X[keep], y[keep], ids[keep], uniq


def euc(P, G):
    return np.sqrt(((P - G)[:, 0] * SCREEN[0]) ** 2 + ((P - G)[:, 1] * SCREEN[1]) ** 2)


def loo_eval(X, y, ids, make_model, aggregate=None, robust_drop=None):
    """点ごとLOO。make_model()->(fit(Xtr,ytr)->m, m.predict(Xte)->(n,2))。
    aggregate: 'median' なら訓練を点内中央値に集約。robust_drop: 外れフレーム除去関数。"""
    P, G, PM, GM = [], [], [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        Xtr, ytr = X[tr], y[tr]
        if robust_drop is not None:
            keep = robust_drop(Xtr, ytr, ids[tr])
            Xtr, ytr = Xtr[keep], ytr[keep]
        if aggregate == 'median':
            tids = ids[tr]
            Xa, ya = [], []
            for q in np.unique(tids):
                Xa.append(np.median(Xtr[tids == q], axis=0))
                ya.append(np.median(ytr[tids == q], axis=0))
            Xtr, ytr = np.array(Xa), np.array(ya)
        m = make_model()
        m.fit(Xtr, ytr)
        pr = m.predict(X[te])
        P.append(pr); G.append(y[te])
        PM.append(np.median(pr, axis=0)); GM.append(np.median(y[te], axis=0))
    P, G = np.vstack(P), np.vstack(G)
    e = euc(P, G)
    em = euc(np.array(PM), np.array(GM))   # 点内中央値後(実利用の時間平滑相当)
    return float(np.median(e)), float(np.mean(e)), float(np.median(em))


# ─── モデル工場（fit(X,y2d) / predict → (n,2)）────────────────────────────────
class Scaled:
    def __init__(self, mk): self.mk, self.sc, self.mx, self.my = mk, StandardScaler(), None, None
    def fit(self, X, y):
        Xs = self.sc.fit_transform(X)
        self.mx = self.mk(); self.my = self.mk()
        self.mx.fit(Xs, y[:, 0]); self.my.fit(Xs, y[:, 1]); return self
    def predict(self, X):
        Xs = self.sc.transform(X)
        return np.column_stack([self.mx.predict(Xs), self.my.predict(Xs)])


def model_zoo():
    z = {}
    for a in [0.1, 0.3, 1.0, 3.0, 10.0]:
        z[f"Ridge(a={a})"] = lambda a=a: Scaled(lambda: Ridge(alpha=a))
    z["Huber"] = lambda: Scaled(lambda: HuberRegressor(max_iter=500))
    for a in [0.01, 0.1]:
        z[f"ElasticNet(a={a})"] = lambda a=a: Scaled(lambda: ElasticNet(alpha=a))
    for k in [5, 10, 20, 40]:
        z[f"KNN(k={k})"] = lambda k=k: Scaled(lambda: KNeighborsRegressor(n_neighbors=k, weights='distance'))
    z["RF(200)"] = lambda: Scaled(lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0))
    z["GBM"] = lambda: Scaled(lambda: GradientBoostingRegressor(random_state=0))
    for c in [1.0, 10.0]:
        for g in ['scale', 0.1]:
            z[f"SVR(C={c},g={g})"] = lambda c=c, g=g: Scaled(lambda: SVR(C=c, gamma=g))
    for a in [0.1, 1.0]:
        z[f"KRR(rbf,a={a})"] = lambda a=a: Scaled(lambda: KernelRidge(alpha=a, kernel='rbf'))
    z["MLP(32,16)"] = lambda: Scaled(lambda: MLPRegressor((32, 16), max_iter=2000, random_state=0))
    def gp():
        k = ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel()
        return GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-3)
    z["GP(rbf)"] = lambda: Scaled(gp)
    return z


FEATURE_SETS = {
    "7D":            lambda X: X[:, :7],
    "4D_iris":       lambda X: X[:, :4],
    "6D_iris+py":    lambda X: X[:, [0, 1, 2, 3, 4, 5]],
    "5D_iris+pitch": lambda X: X[:, [0, 1, 2, 3, 4]],
    "7D+sq":         lambda X: np.hstack([X[:, :7], X[:, :7] ** 2]),
    "7D+iris_xy":    lambda X: np.hstack([X[:, :7], (X[:, 0] * X[:, 1])[:, None],
                                          (X[:, 2] * X[:, 3])[:, None]]),
}


def logline(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def drop_blink(Xtr, ytr, tids):
    # 16D の EAR は [8],[9]。ここは7Dのみ渡るので使えない → 全採用
    return np.ones(len(Xtr), bool)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    stage = sys.argv[2] if len(sys.argv) > 2 else "all"
    X, y, ids, uniq = load(sid)
    logline(f"\n### 探索 session_{sid}  ({len(X)}フレーム/{len(uniq)}点)  stage={stage}")
    logline("指標: median Euc(フレーム) / mean / **median Euc(点内中央値後=実効)** [cm]")

    results = []
    if stage in ("models", "all"):
        logline("\n#### 段階1: モデル×特徴 (7D基準)")
        zoo = model_zoo()
        for fname, fsel in FEATURE_SETS.items():
            Xf = fsel(X)
            best = None
            for mname, mk in zoo.items():
                try:
                    med, mean, medm = loo_eval(Xf, y, ids, mk)
                except Exception as ex:
                    continue
                results.append((fname, mname, med, mean, medm))
                if best is None or medm < best[2]:
                    best = (mname, med, medm)
            if best:
                logline(f"  [{fname:14s}] best: {best[0]:18s} 実効={best[2]:.3f}cm (frame med={best[1]:.3f})")
        results.sort(key=lambda r: r[4])
        logline("\n  --- 全体 TOP10 (実効median昇順) ---")
        for r in results[:10]:
            logline(f"    {r[4]:.3f}cm  {r[0]:14s} {r[1]:20s} (frame {r[2]:.3f})")
        with open(ROOT / "results" / f"explore_{sid}_models.json", "w") as f:
            json.dump(results, f, indent=1)

    if stage in ("ncalib", "all"):
        logline("\n#### 段階2: キャリブ点数カーブ (Ridge a=1.0, 7D, 点内中央値後)")
        rng = np.random.RandomState(0)
        allp = np.unique(ids)
        for npt in [3, 4, 5, 6, 7, 8, 9]:
            errs = []
            for _ in range(20):
                if npt >= len(allp):
                    sub = allp
                else:
                    sub = rng.choice(allp, npt, replace=False)
                mask = np.isin(ids, sub)
                Xs, ys, is_ = X[mask][:, :7], y[mask], ids[mask]
                if len(np.unique(is_)) < 3:
                    continue
                try:
                    _, _, medm = loo_eval(Xs, ys, is_, lambda: Scaled(lambda: Ridge(1.0)))
                    errs.append(medm)
                except Exception:
                    pass
                if npt >= len(allp):
                    break
            if errs:
                logline(f"  {npt}点: 実効 median={np.median(errs):.3f}cm  (n_trial={len(errs)})")

    logline(f"\n(探索完了 stage={stage})")


if __name__ == "__main__":
    main()
