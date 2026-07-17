"""段階6: M3(虹彩4D 2次集約 + 姿勢補正)を強化し「正面1cm + 横向きロバスト」両立を狙う。
基準: M2 Huber(全姿勢安定, 30+ 4.26cm) と M3(正面1.09cm, 30+ 7.10cm)。

バリアント:
  H1 姿勢補正を2次(pitch,yaw,dist の2次多項式)
  H2 姿勢補正を Huber (残差の外れ値に強く)
  H3 姿勢補正を全16D特徴の一部(roll,aspect も) ※次元は最終出力でなく補正内部なので可
  H4 虹彩4D 2次集約 と 7D Huber を姿勢ゲートでブレンド
     (正面=虹彩多項式[高精度], 横向き=Huber[ロバスト], hybrid_calibration の発想)
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
    return X[keep], y[keep], ids[keep], uniq


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


def iris_poly(Xtr, ytr, itr):
    Xa, ya = agg(Xtr[:, :4], ytr, itr)
    bx = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 0])
    by = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 1])
    return bx, by


def huber7(Xtr, ytr):
    sc = StandardScaler().fit(Xtr[:, :7])
    mx = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 0])
    my = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 1])
    return sc, mx, my


def pose_corr(Xtr, ytr, base_tr, cols, deg, huber):
    resid = ytr - base_tr
    P = Xtr[:, cols]
    if deg == 2:
        pf = PolynomialFeatures(2)
        P = pf.fit_transform(P)
    else:
        pf = None
    sc = StandardScaler().fit(P)
    Reg = (lambda: HuberRegressor(max_iter=800)) if huber else (lambda: Ridge(1.0))
    rx = Reg().fit(sc.transform(P), resid[:, 0])
    ry = Reg().fit(sc.transform(P), resid[:, 1])
    return pf, sc, rx, ry


def apply_corr(Xte, cols, pf, sc, rx, ry):
    P = Xte[:, cols]
    if pf is not None:
        P = pf.transform(P)
    Ps = sc.transform(P)
    return np.column_stack([rx.predict(Ps), ry.predict(Ps)])


def H_variant(cols, deg, huber):
    def fn(Xtr, ytr, itr, Xte):
        bx, by = iris_poly(Xtr, ytr, itr)
        base_tr = np.column_stack([bx.predict(Xtr[:, :4]), by.predict(Xtr[:, :4])])
        pf, sc, rx, ry = pose_corr(Xtr, ytr, base_tr, cols, deg, huber)
        base_te = np.column_stack([bx.predict(Xte[:, :4]), by.predict(Xte[:, :4])])
        return base_te + apply_corr(Xte, cols, pf, sc, rx, ry)
    return fn


def H4_gate(Xtr, ytr, itr, Xte):
    # 正面=虹彩多項式, 横向き=Huber を yaw でゲートブレンド
    bx, by = iris_poly(Xtr, ytr, itr)
    sc, mx, my = huber7(Xtr, ytr)
    poly = np.column_stack([bx.predict(Xte[:, :4]), by.predict(Xte[:, :4])])
    hub = np.column_stack([mx.predict(sc.transform(Xte[:, :7])), my.predict(sc.transform(Xte[:, :7]))])
    # キャリブ時の平均|yaw|
    ref_yaw = np.mean(np.abs(Xtr[:, 5]))
    dev = np.abs(np.abs(Xte[:, 5]) - ref_yaw)
    w = np.exp(-dev / np.radians(8.0))[:, None]  # 正面近い=polyを信頼
    return w * poly + (1 - w) * hub


def evaluate(X, y, ids, fn):
    P, G, Yaw = [], [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        P.append(fn(X[tr], y[tr], ids[tr], X[te])); G.append(y[te]); Yaw.append(X[te][:, 5])
    P, G, Yaw = np.vstack(P), np.vstack(G), np.concatenate(Yaw)
    e = euc(P, G)
    PM, GM, idx = [], [], 0
    for p in np.unique(ids):
        n = int((ids == p).sum())
        PM.append(np.median(P[idx:idx+n], axis=0)); GM.append(np.median(G[idx:idx+n], axis=0)); idx += n
    eff = float(np.median(euc(np.array(PM), np.array(GM))))
    yd = np.abs(np.degrees(Yaw))
    bv = [np.median(e[(yd >= lo) & (yd < hi)]) if ((yd >= lo) & (yd < hi)).sum() else np.nan
          for lo, hi in BINS]
    return eff, float(np.median(e)), bv


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, uniq = load(sid)
    logline(f"\n### 段階6 姿勢補正ハイブリッド session_{sid} ({len(X)}フレーム/{len(uniq)}点)")
    logline(f"  目標: 正面1cm台 & 30+ を M2 Huber(4.26)並みに")
    logline(f"  {'手法':34s} {'実効':>6s} {'frame':>6s}  " +
            " ".join(f"|y|{lo}-{hi}" for lo, hi in BINS))
    variants = [
        ("H1 補正=pyd 2次", H_variant([4, 5, 6], 2, False)),
        ("H2 補正=pyd Huber", H_variant([4, 5, 6], 1, True)),
        ("H3 補正=py+roll+aspect", H_variant([4, 5, 7, 14, 15], 1, False)),
        ("H3b 補正=py+roll+aspect 2次", H_variant([4, 5, 7, 14, 15], 2, False)),
        ("H4 yawゲートブレンド", H4_gate),
    ]
    for name, fn in variants:
        try:
            eff, fr, bv = evaluate(X, y, ids, fn)
            logline(f"  {name:34s} {eff:6.3f} {fr:6.3f}  " + " ".join(f"{v:7.2f}" for v in bv))
        except Exception as ex:
            logline(f"  {name:34s} ERR {ex}")
    logline("\n(段階6完了)")


if __name__ == "__main__":
    main()
