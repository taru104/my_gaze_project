"""段階5: 段階4ベスト(虹彩4D 2次多項式 9点集約 1.082cm)の姿勢ロバスト性検証 +
姿勢補正ハイブリッド探索。横向き(ユーザ最重要要件)で崩れないかを確かめる。

比較:
  M1 虹彩4D deg2 9点集約           (段階4ベスト。姿勢項なし)
  M2 7D Huber 全frame              (段階2ベスト。姿勢項あり・ロバスト)
  M3 虹彩4D deg2 集約 + 姿勢線形補正 (基本は虹彩2次、pitch/yaw/distで線形オフセット)
  M4 (虹彩4D+pitch,yaw) deg2 集約   (姿勢も2次に入れる)
姿勢bin別 median Euclidean で評価。フレーム単位の実yawで層別。
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


def m1(Xtr, ytr, itr, Xte):  # 虹彩4D deg2 集約
    Xa, ya = agg(Xtr[:, :4], ytr, itr)
    mx = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 0])
    my = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 1])
    return np.column_stack([mx.predict(Xte[:, :4]), my.predict(Xte[:, :4])])


def m2(Xtr, ytr, itr, Xte):  # 7D Huber
    sc = StandardScaler().fit(Xtr[:, :7])
    mx = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 0])
    my = HuberRegressor(max_iter=800).fit(sc.transform(Xtr[:, :7]), ytr[:, 1])
    return np.column_stack([mx.predict(sc.transform(Xte[:, :7])), my.predict(sc.transform(Xte[:, :7]))])


def m3(Xtr, ytr, itr, Xte):  # 虹彩4D deg2 集約 + 姿勢線形補正(全frameで残差回帰)
    Xa, ya = agg(Xtr[:, :4], ytr, itr)
    bx = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 0])
    by = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.1)).fit(Xa, ya[:, 1])
    base_tr = np.column_stack([bx.predict(Xtr[:, :4]), by.predict(Xtr[:, :4])])
    resid = ytr - base_tr
    pose = Xtr[:, [4, 5, 6]]
    scp = StandardScaler().fit(pose)
    rx = Ridge(1.0).fit(scp.transform(pose), resid[:, 0])
    ry = Ridge(1.0).fit(scp.transform(pose), resid[:, 1])
    base_te = np.column_stack([bx.predict(Xte[:, :4]), by.predict(Xte[:, :4])])
    corr = np.column_stack([rx.predict(scp.transform(Xte[:, [4, 5, 6]])),
                            ry.predict(scp.transform(Xte[:, [4, 5, 6]]))])
    return base_te + corr


def m4(Xtr, ytr, itr, Xte):  # (虹彩4D+pitch,yaw) deg2 集約
    cols = [0, 1, 2, 3, 4, 5]
    Xa, ya = agg(Xtr[:, cols], ytr, itr)
    mx = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.3)).fit(Xa, ya[:, 0])
    my = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(0.3)).fit(Xa, ya[:, 1])
    return np.column_stack([mx.predict(Xte[:, cols]), my.predict(Xte[:, cols])])


def evaluate(X, y, ids, fn):
    P, G, Yaw = [], [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        pr = fn(X[tr], y[tr], ids[tr], X[te])
        P.append(pr); G.append(y[te]); Yaw.append(X[te][:, 5])
    P, G, Yaw = np.vstack(P), np.vstack(G), np.concatenate(Yaw)
    e = euc(P, G)
    # 実効(点内中央値)
    PM, GM = [], []
    idx = 0
    for p in np.unique(ids):
        n = int((ids == p).sum())
        PM.append(np.median(P[idx:idx+n], axis=0)); GM.append(np.median(G[idx:idx+n], axis=0)); idx += n
    eff = float(np.median(euc(np.array(PM), np.array(GM))))
    yd = np.abs(np.degrees(Yaw))
    binvals = []
    for lo, hi in BINS:
        k = (yd >= lo) & (yd < hi)
        binvals.append(np.median(e[k]) if k.sum() else np.nan)
    return eff, float(np.median(e)), binvals


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, uniq = load(sid)
    logline(f"\n### 段階5 姿勢ロバスト検証 session_{sid} ({len(X)}フレーム/{len(uniq)}点)")
    logline(f"  {'手法':36s} {'実効':>6s} {'frame':>6s}  " +
            " ".join(f"|y|{lo}-{hi}" for lo, hi in BINS))
    for name, fn in [("M1 虹彩4D deg2 集約(段階4ベスト)", m1),
                     ("M2 7D Huber(段階2ベスト)", m2),
                     ("M3 虹彩4D deg2 + 姿勢線形補正", m3),
                     ("M4 (虹彩4D+py) deg2 集約", m4)]:
        try:
            eff, fr, bv = evaluate(X, y, ids, fn)
            logline(f"  {name:36s} {eff:6.3f} {fr:6.3f}  " +
                    " ".join(f"{v:7.2f}" for v in bv))
        except Exception as ex:
            logline(f"  {name:36s} ERR {ex}")
    logline("\n(段階5完了)")


if __name__ == "__main__":
    main()
