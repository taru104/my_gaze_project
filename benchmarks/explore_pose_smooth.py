"""段階8: 姿勢(pitch/yaw/roll/dist)の時間平滑が横向きの時間分割崩壊を緩和するか。

段階7で「横向きの壁=虹彩/姿勢推定の時間的不安定性」と判明。虹彩4D(視線信号)は点で切替わるが、
頭部姿勢は時間的に滑らか。→ 姿勢だけ時系列で移動中央値平滑し、時間分割ホールドアウトで横向きが
改善するか見る。改善すれば features.py の feat7d 姿勢を「生solvePnP」から「時間平滑」に変える価値。

平滑対象: X[:,4]=pitch, [:,5]=yaw, [:,6]=dist, [:,7]=roll(参考)。虹彩[:4]は平滑しない。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import make_pipeline

SCREEN = np.array([30.9, 17.4])
CALIB_DISCARD = 1.0
ROOT = Path(__file__).parent.parent
LOG = ROOT / "results" / "exploration_log.md"
BINS = [(0, 10), (10, 20), (20, 30), (30, 90)]


def load(sid):
    d = np.load(ROOT / "logs" / f"session_{sid}_rich16d.npz")
    X, y, ht, ts = d["X"].copy(), d["y_norm"], d["has_target"], d["time_s"]
    m = ht.copy(); X, y, ts = X[m], y[m], ts[m]
    # time_s昇順に整列(時系列平滑のため)
    order = np.argsort(ts)
    X, y, ts = X[order], y[order], ts[order]
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    keep = np.zeros(len(X), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        keep[sel[ts[sel] - ts[sel].min() >= CALIB_DISCARD]] = True
    return X[keep], y[keep], ids[keep], ts[keep], uniq


def smooth_pose(X, win):
    """姿勢列 [4,5,6,7] を時系列移動中央値で平滑(虹彩[:4]はそのまま)。"""
    Xs = X.copy()
    half = win // 2
    for col in [4, 5, 6, 7]:
        v = X[:, col]
        out = v.copy()
        for i in range(len(v)):
            lo, hi = max(0, i - half), min(len(v), i + half + 1)
            out[i] = np.median(v[lo:hi])
        Xs[:, col] = out
    return Xs


def euc(P, G):
    return np.sqrt(((P - G)[:, 0] * SCREEN[0]) ** 2 + ((P - G)[:, 1] * SCREEN[1]) ** 2)


def logline(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def fit_m2(Xtr, ytr):
    sc = StandardScaler().fit(Xtr[:, :7])
    hx = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 0])
    hy = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 1])
    return lambda Xte: np.column_stack([hx.predict(sc.transform(Xte[:, :7])),
                                        hy.predict(sc.transform(Xte[:, :7]))])


def bin_report(e, yd):
    return [np.median(e[(yd >= lo) & (yd < hi)]) if ((yd >= lo) & (yd < hi)).sum() else np.nan
            for lo, hi in BINS]


def time_split(X, y, ids, ts):
    tr = np.zeros(len(X), bool)
    for p in np.unique(ids):
        s = np.where(ids == p)[0]
        order = s[np.argsort(ts[s])]
        tr[order[:len(order)//2]] = True
    return tr, ~tr


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, ts, uniq = load(sid)
    logline(f"\n### 段階8 姿勢時間平滑 session_{sid} ({len(X)}フレーム/{len(uniq)}点)")
    logline("  時間分割ホールドアウト(前半学習→後半評価) M2 Huber, frame median Euc")
    logline(f"  {'平滑window':12s} {'frame':>6s}  " + " ".join(f"|y|{lo}-{hi}" for lo, hi in BINS))

    trm, tem = time_split(X, y, ids, ts)
    yd = np.abs(np.degrees(X[tem][:, 5]))
    for win in [1, 5, 15, 31, 61]:
        Xs = X if win == 1 else smooth_pose(X, win)
        pred = fit_m2(Xs[trm], y[trm])
        e = euc(pred(Xs[tem]), y[tem])
        logline(f"  win={win:3d}      {np.median(e):6.3f}  " + " ".join(f"{v:7.2f}" for v in bin_report(e, yd)))

    # 点ごとLOO(空間)でも副作用がないか
    logline("\n  参考: 点ごとLOO 実効median (平滑の空間精度への影響)")
    for win in [1, 15, 61]:
        Xs = X if win == 1 else smooth_pose(X, win)
        PM, GM = [], []
        for p in np.unique(ids):
            te, tr = ids == p, ids != p
            pred = fit_m2(Xs[tr], y[tr])
            pr = pred(Xs[te])
            PM.append(np.median(pr, axis=0)); GM.append(np.median(y[te], axis=0))
        logline(f"  win={win:3d}: 実効={np.median(euc(np.array(PM), np.array(GM))):.3f}cm")
    logline("\n(段階8完了)")


if __name__ == "__main__":
    main()
