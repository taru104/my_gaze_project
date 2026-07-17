"""1cm探索 段階4: 視線キャリブの古典手法。Huber 1.87cm を超えられるか。

視線推定の伝統的キャリブは「PoG候補(虹彩2D) → 画面2D」の多項式/スプライン。
9点を代表点に集約してマッピングを張る系も試す(過補間に注意)。次元は増やさない。

手法:
  - 9点集約 → 2次多項式 (虹彩2D / 4D / 7D)
  - 9点集約 → 薄板スプライン TPS (虹彩2D→画面)
  - 全frame → RBFInterpolator (thin_plate_spline / multiquadric)
  - 全frame → 2次poly full + 強Ridge
基準: 7D Huber 実効1.874cm。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import make_pipeline
try:
    from scipy.interpolate import RBFInterpolator
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

SCREEN = np.array([30.9, 17.4])
CALIB_DISCARD = 1.0
ROOT = Path(__file__).parent.parent
LOG = ROOT / "results" / "exploration_log.md"


def load(sid):
    d = np.load(ROOT / "logs" / f"session_{sid}_rich16d.npz")
    X, y, ht, ts = d["X"], d["y_norm"], d["has_target"], d["time_s"]
    m = ht.copy(); X, y, ts = X[m], y[m], ts[m]
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    keep = np.zeros(len(X), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        keep[sel[ts[sel] - ts[sel].min() >= CALIB_DISCARD]] = True
    return X[keep], y[keep], ids[keep], uniq


def euc(P, G):
    return np.sqrt(((P - G)[:, 0] * SCREEN[0]) ** 2 + ((P - G)[:, 1] * SCREEN[1]) ** 2)


def logline(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def loo_generic(X, y, ids, cols, fit_fn, agg_train=False):
    """fit_fn(Xtr_cols, ytr) -> predict(Xte_cols)->(n,2)。agg_train: 訓練を点内中央値に集約。"""
    PM, GM = [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        Xtr, ytr, itr = X[tr][:, cols], y[tr], ids[tr]
        if agg_train:
            Xa, ya = [], []
            for q in np.unique(itr):
                Xa.append(np.median(Xtr[itr == q], axis=0)); ya.append(np.median(ytr[itr == q], axis=0))
            Xtr, ytr = np.array(Xa), np.array(ya)
        pred = fit_fn(Xtr, ytr, X[te][:, cols])
        PM.append(np.median(pred, axis=0)); GM.append(np.median(y[te], axis=0))
    return float(np.median(euc(np.array(PM), np.array(GM))))


def poly_ridge(deg, alpha):
    def f(Xtr, ytr, Xte):
        mx = make_pipeline(PolynomialFeatures(deg), StandardScaler(), Ridge(alpha))
        my = make_pipeline(PolynomialFeatures(deg), StandardScaler(), Ridge(alpha))
        mx.fit(Xtr, ytr[:, 0]); my.fit(Xtr, ytr[:, 1])
        return np.column_stack([mx.predict(Xte), my.predict(Xte)])
    return f


def rbf(kernel, smoothing):
    def f(Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        m = RBFInterpolator(sc.transform(Xtr), ytr, kernel=kernel, smoothing=smoothing)
        return m(sc.transform(Xte))
    return f


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, uniq = load(sid)
    logline(f"\n### 探索段階4 古典キャリブ session_{sid} ({len(X)}フレーム/{len(uniq)}点) 基準=Huber 1.874cm")

    IRIS2 = [0, 1]        # 左目虹彩
    IRIS4 = [0, 1, 2, 3]
    D7 = list(range(7))

    logline("\n#### 9点集約 → 多項式")
    for cols, cn in [(IRIS2, "虹彩2D"), (IRIS4, "虹彩4D"), (D7, "7D")]:
        for deg in [1, 2, 3]:
            try:
                e = loo_generic(X, y, ids, cols, poly_ridge(deg, 0.1), agg_train=True)
                logline(f"  {cn:6s} deg{deg} (9点集約): 実効={e:.3f}cm")
            except Exception as ex:
                logline(f"  {cn} deg{deg}: err {ex}")

    logline("\n#### 全frame → 多項式 + 強Ridge")
    for cols, cn in [(IRIS4, "虹彩4D"), (D7, "7D")]:
        for deg, a in [(2, 1.0), (2, 10.0), (3, 10.0)]:
            e = loo_generic(X, y, ids, cols, poly_ridge(deg, a))
            logline(f"  {cn:6s} deg{deg} a={a}: 実効={e:.3f}cm")

    if HAVE_SCIPY:
        logline("\n#### RBF補間 (全frame)")
        for cols, cn in [(IRIS4, "虹彩4D"), (D7, "7D")]:
            for kern in ["thin_plate_spline", "multiquadric", "linear"]:
                for sm in [0.1, 1.0, 10.0]:
                    try:
                        e = loo_generic(X, y, ids, cols, rbf(kern, sm))
                        logline(f"  {cn:6s} {kern:18s} sm={sm}: 実効={e:.3f}cm")
                    except Exception as ex:
                        pass
        logline("\n#### RBF補間 (9点集約)")
        for cols, cn in [(IRIS2, "虹彩2D"), (D7, "7D")]:
            for kern in ["thin_plate_spline", "linear"]:
                try:
                    e = loo_generic(X, y, ids, cols, rbf(kern, 0.0), agg_train=True)
                    logline(f"  {cn:6s} {kern:18s} (9点集約): 実効={e:.3f}cm")
                except Exception:
                    pass
    else:
        logline("\n(scipy無し → RBF スキップ)")
    logline("\n(段階4完了)")


if __name__ == "__main__":
    main()
