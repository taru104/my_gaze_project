"""1cm探索 段階3: 7Dのアブレーション（どの次元が効くか）+ 姿勢bin別 + Huber係数解釈。

次元は増やさない方針なので、7Dの「中身」を理解して将来の特徴設計指針にする。
- 各次元を1つ抜く(leave-one-feature-out)と実効がどう悪化するか＝寄与度
- 各次元 単独/累積 の予測力
- Huber vs Ridge の姿勢bin別（横向きで差が出るか）

7D = [Lx,Ly,Rx,Ry,pitch,yaw,dist]  (index 0..6)
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

SCREEN = np.array([30.9, 17.4])
CALIB_DISCARD = 1.0
ROOT = Path(__file__).parent.parent
LOG = ROOT / "results" / "exploration_log.md"
NAMES = ["Lx", "Ly", "Rx", "Ry", "pitch", "yaw", "dist"]


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


def huber():
    return _XY(lambda: HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=800))


def ridge():
    return _XY(lambda: Ridge(1.0))


class _XY:
    def __init__(self, mk): self.mk, self.sc = mk, StandardScaler()
    def fit(self, X, y):
        Xs = self.sc.fit_transform(X)
        self.mx = self.mk(); self.my = self.mk()
        self.mx.fit(Xs, y[:, 0]); self.my.fit(Xs, y[:, 1]); return self
    def predict(self, X):
        Xs = self.sc.transform(X)
        return np.column_stack([self.mx.predict(Xs), self.my.predict(Xs)])


def loo(X, y, ids, make, cols):
    PM, GM, Pf, Gf, YAW = [], [], [], [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        m = make().fit(X[tr][:, cols], y[tr])
        pr = m.predict(X[te][:, cols])
        Pf.append(pr); Gf.append(y[te]); YAW.append(X[te][:, 5])
        PM.append(np.median(pr, axis=0)); GM.append(np.median(y[te], axis=0))
    ef = euc(np.vstack(Pf), np.vstack(Gf))
    return float(np.median(euc(np.array(PM), np.array(GM)))), ef, np.concatenate(YAW)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, uniq = load(sid)
    logline(f"\n### 探索段階3 アブレーション session_{sid} ({len(X)}フレーム/{len(uniq)}点)")

    full, ff, _ = loo(X, y, ids, huber, list(range(7)))
    logline(f"\n#### 全7D Huber: 実効={full:.3f}cm (frame {np.median(ff):.3f})")

    logline("\n#### leave-one-feature-out (1つ抜くと実効がどう悪化=寄与度)")
    rows = []
    for i in range(7):
        cols = [c for c in range(7) if c != i]
        e, _, _ = loo(X, y, ids, huber, cols)
        rows.append((NAMES[i], e, e - full))
    for n, e, d in sorted(rows, key=lambda r: -r[2]):
        logline(f"  -{n:6s} → 実効={e:.3f}cm  (悪化 {d:+.3f})")

    logline("\n#### 各次元 単独の予測力")
    for i in range(7):
        e, _, _ = loo(X, y, ids, huber, [i])
        logline(f"  {NAMES[i]:6s}単独: 実効={e:.3f}cm")

    logline("\n#### 累積 (虹彩4D → +pitch → +yaw → +dist)")
    for cols, name in [([0,1,2,3], "虹彩4D"), ([0,1,2,3,4], "+pitch"),
                       ([0,1,2,3,4,5], "+yaw"), ([0,1,2,3,4,5,6], "+dist=7D")]:
        e, _, _ = loo(X, y, ids, huber, cols)
        logline(f"  {name:10s}: 実効={e:.3f}cm")

    logline("\n#### Huber vs Ridge 姿勢|yaw|bin別 (frame median)")
    _, efh, yawh = loo(X, y, ids, huber, list(range(7)))
    _, efr, yawr = loo(X, y, ids, ridge, list(range(7)))
    yd = np.abs(np.degrees(yawh))
    logline(f"  {'bin':10s} {'Huber':>8s} {'Ridge':>8s}  n")
    for lo, hi in [(0,10),(10,20),(20,30),(30,90)]:
        k = (yd >= lo) & (yd < hi)
        if k.sum():
            logline(f"  |yaw|{lo:2d}-{hi:2d}  {np.median(efh[k]):8.3f} {np.median(efr[k]):8.3f}  {int(k.sum())}")
    logline("\n(段階3完了)")


if __name__ == "__main__":
    main()
