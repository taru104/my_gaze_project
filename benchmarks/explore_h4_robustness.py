"""段階7: H4(yawゲートブレンド)が本物か過適合かを1セッション内で最大限検証。

(A) ゲート tau 感度: tau=4,6,8,10,12,16° で H4 の精度がどう動くか。
    ピーキー(特定tauだけ良い)=過適合、なだらか=頑健。
(B) 時間分割ホールドアウト: 各点のフレームを時間順で前半/後半に分割。前半で学習・後半で評価。
    点ごとLOO(空間外挿)とは別の汎化軸=「同じ点の別時刻」＝実利用に近い。
    ここでもH4がM2に勝てば、過適合でなく実質的改善の可能性が上がる。
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
    X, y, ht, ts = d["X"], d["y_norm"], d["has_target"], d["time_s"]
    m = ht.copy(); X, y, ts = X[m], y[m], ts[m]
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    keep = np.zeros(len(X), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        keep[sel[ts[sel] - ts[sel].min() >= CALIB_DISCARD]] = True
    return X[keep], y[keep], ids[keep], ts[keep], uniq


def euc(P, G):
    return np.sqrt(((P - G)[:, 0] * SCREEN[0]) ** 2 + ((P - G)[:, 1] * SCREEN[1]) ** 2)


def logline(s):
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def agg(X, y, ids):
    Xa, ya = [], []
    for q in np.unique(ids):
        Xa.append(np.median(X[ids == q], axis=0)); ya.append(np.median(y[ids == q], axis=0))
    return np.array(Xa), np.array(ya)


def fit_h4(Xtr, ytr, itr, tau_deg):
    Xa, ya = agg(Xtr[:, :4], ytr, itr)
    bx = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 0])
    by = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 1])
    sc = StandardScaler().fit(Xtr[:, :7])
    hx = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 0])
    hy = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 1])
    ref = np.mean(np.abs(Xtr[:, 5]))
    def pred(Xte):
        poly = np.column_stack([bx.predict(Xte[:, :4]), by.predict(Xte[:, :4])])
        hub = np.column_stack([hx.predict(sc.transform(Xte[:, :7])), hy.predict(sc.transform(Xte[:, :7]))])
        w = np.exp(-np.abs(np.abs(Xte[:, 5]) - ref) / np.radians(tau_deg))[:, None]
        return w * poly + (1 - w) * hub
    return pred


def fit_m2(Xtr, ytr, itr):
    sc = StandardScaler().fit(Xtr[:, :7])
    hx = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 0])
    hy = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 1])
    return lambda Xte: np.column_stack([hx.predict(sc.transform(Xte[:, :7])),
                                        hy.predict(sc.transform(Xte[:, :7]))])


def loo_eff(X, y, ids, fit, **kw):
    PM, GM = [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        pred = fit(X[tr], y[tr], ids[tr], **kw)
        pr = pred(X[te])
        PM.append(np.median(pr, axis=0)); GM.append(np.median(y[te], axis=0))
    return float(np.median(euc(np.array(PM), np.array(GM))))


def bin_report(e, yd):
    return [np.median(e[(yd >= lo) & (yd < hi)]) if ((yd >= lo) & (yd < hi)).sum() else np.nan
            for lo, hi in BINS]


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, ts, uniq = load(sid)
    logline(f"\n### 段階7 H4頑健性検証 session_{sid} ({len(X)}フレーム/{len(uniq)}点)")

    logline("\n#### (A) ゲート tau 感度 (点ごとLOO 実効median)")
    for tau in [4, 6, 8, 10, 12, 16, 24]:
        e = loo_eff(X, y, ids, fit_h4, tau_deg=tau)
        logline(f"  tau={tau:2d}°: 実効={e:.3f}cm")
    logline("  → 広い範囲で低ければ頑健。特定tauだけ低ければ過適合。")

    logline("\n#### (B) 時間分割ホールドアウト (各点 前半学習→後半評価)")
    tr_mask = np.zeros(len(X), bool)
    for p in np.unique(ids):
        s = np.where(ids == p)[0]
        order = s[np.argsort(ts[s])]
        tr_mask[order[:len(order)//2]] = True
    te_mask = ~tr_mask
    yd_te = np.abs(np.degrees(X[te_mask][:, 5]))
    logline(f"  学習{tr_mask.sum()} / 評価{te_mask.sum()}フレーム")
    logline(f"  {'手法':16s} {'frame':>6s}  " + " ".join(f"|y|{lo}-{hi}" for lo, hi in BINS))
    # M2
    pred = fit_m2(X[tr_mask], y[tr_mask], ids[tr_mask])
    e = euc(pred(X[te_mask]), y[te_mask])
    logline(f"  {'M2 Huber':16s} {np.median(e):6.3f}  " + " ".join(f"{v:7.2f}" for v in bin_report(e, yd_te)))
    # H4 (tau=8)
    pred = fit_h4(X[tr_mask], y[tr_mask], ids[tr_mask], tau_deg=8)
    e = euc(pred(X[te_mask]), y[te_mask])
    logline(f"  {'H4 tau=8':16s} {np.median(e):6.3f}  " + " ".join(f"{v:7.2f}" for v in bin_report(e, yd_te)))
    logline("  → 時間外挿でもH4がM2並み/以上なら過適合でない可能性↑")
    logline("\n(段階7完了)")


if __name__ == "__main__":
    main()
