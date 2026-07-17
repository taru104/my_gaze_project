"""1cm探索 段階2: Huber(段階1ベスト)を軸に前処理/後処理/アンサンブルで詰める。

モデル入力は7Dのまま(次元は増やさない方針)。ただし訓練データのクレンジングには
16D特徴(EAR等)を「フレームの質の判定」に使う(=特徴追加ではなくデータ選別)。

軸:
  A. 外れフレーム除去 (まばたきEAR低 / 姿勢MAD大 / 時間ジャンプ)
  B. per-eye アンサンブル (左目/右目 別モデル→平均)
  C. モデルアンサンブル (Huber + KNN + GBM ブレンド)
  D. Huber ハイパラ sweep (epsilon, alpha)
  E. 後処理 (点内 trimmed-mean, 予測クリップ)

指標: median Euc(点内中央値後=実効) [cm]。段階1ベスト 7D Huber 実効1.874 が基準。
結果は results/exploration_log.md に追記。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

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


class HuberXY:
    def __init__(self, eps=1.35, alpha=1e-4):
        self.eps, self.alpha, self.sc = eps, alpha, StandardScaler()
    def fit(self, X, y):
        Xs = self.sc.fit_transform(X)
        self.mx = HuberRegressor(epsilon=self.eps, alpha=self.alpha, max_iter=800).fit(Xs, y[:, 0])
        self.my = HuberRegressor(epsilon=self.eps, alpha=self.alpha, max_iter=800).fit(Xs, y[:, 1])
        return self
    def predict(self, X):
        Xs = self.sc.transform(X)
        return np.column_stack([self.mx.predict(Xs), self.my.predict(Xs)])


def loo(Xful, y, ids, make, feat=lambda X: X[:, :7], clean=None, post='median'):
    """点ごとLOO。feat: 16D→モデル入力。clean(X16,y,ids)->bool mask(訓練のみ)。
    post: 'median'|'trim' 点内集約。返り: 実効median, frame median。"""
    PM, GM, Pf, Gf = [], [], [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        X16tr, ytr, itr = Xful[tr], y[tr], ids[tr]
        if clean is not None:
            k = clean(X16tr, ytr, itr)
            X16tr, ytr = X16tr[k], ytr[k]
        m = make().fit(feat(X16tr), ytr)
        pr = m.predict(feat(Xful[te]))
        Pf.append(pr); Gf.append(y[te])
        if post == 'trim':
            lo, hi = np.percentile(pr, [20, 80], axis=0)
            sel = np.all((pr >= lo) & (pr <= hi), axis=1)
            pm = pr[sel].mean(0) if sel.sum() else pr.mean(0)
        else:
            pm = np.median(pr, axis=0)
        PM.append(pm); GM.append(np.median(y[te], axis=0))
    return float(np.median(euc(np.array(PM), np.array(GM)))), \
           float(np.median(euc(np.vstack(Pf), np.vstack(Gf))))


# ─── クレンジング関数 ──────────────────────────────────────────────────────
def clean_blink(frac):
    def f(X16, y, ids):
        keep = np.ones(len(X16), bool)
        ear = (X16[:, 8] + X16[:, 9]) / 2.0
        for q in np.unique(ids):
            s = np.where(ids == q)[0]
            thr = np.percentile(ear[s], frac * 100)
            keep[s[ear[s] < thr]] = False
        return keep
    return f


def clean_pose_mad(k):
    def f(X16, y, ids):
        keep = np.ones(len(X16), bool)
        for q in np.unique(ids):
            s = np.where(ids == q)[0]
            for col in [4, 5, 7]:  # pitch,yaw,roll
                v = X16[s, col]
                med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-8
                keep[s[np.abs(v - med) > k * mad]] = False
        return keep
    return f


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    X, y, ids, uniq = load(sid)
    logline(f"\n### 探索段階2 session_{sid} ({len(X)}フレーム/{len(uniq)}点)  基準=7D Huber 実効1.874cm")

    base, basef = loo(X, y, ids, HuberXY)
    logline(f"\n#### 基準再現: 7D Huber  実効={base:.3f}cm (frame {basef:.3f})")

    logline("\n#### A. 外れフレーム除去")
    for frac in [0.1, 0.2, 0.3]:
        e, ef = loo(X, y, ids, HuberXY, clean=clean_blink(frac))
        logline(f"  まばたき下位{int(frac*100)}%除去: 実効={e:.3f}cm (frame {ef:.3f})")
    for k in [2.5, 3.5, 5.0]:
        e, ef = loo(X, y, ids, HuberXY, clean=clean_pose_mad(k))
        logline(f"  姿勢MAD>{k}除去: 実効={e:.3f}cm (frame {ef:.3f})")

    logline("\n#### B. per-eye アンサンブル")
    class PerEye:
        def fit(self, X, y):
            # X は7D。左目[0,1]+[4,5,6], 右目[2,3]+[4,5,6]
            self.l = HuberXY().fit(X[:, [0, 1, 4, 5, 6]], y)
            self.r = HuberXY().fit(X[:, [2, 3, 4, 5, 6]], y)
            return self
        def predict(self, X):
            return (self.l.predict(X[:, [0, 1, 4, 5, 6]]) + self.r.predict(X[:, [2, 3, 4, 5, 6]])) / 2.0
    e, ef = loo(X, y, ids, PerEye)
    logline(f"  per-eye 平均: 実効={e:.3f}cm (frame {ef:.3f})")

    logline("\n#### C. モデルアンサンブル (Huber+KNN+GBM 平均)")
    class Ens:
        def fit(self, X, y):
            self.h = HuberXY().fit(X, y)
            self.k = _ScaledKNN().fit(X, y)
            self.g = _ScaledGBM().fit(X, y)
            return self
        def predict(self, X):
            return (self.h.predict(X) + self.k.predict(X) + self.g.predict(X)) / 3.0
    e, ef = loo(X, y, ids, Ens)
    logline(f"  Huber+KNN+GBM: 実効={e:.3f}cm (frame {ef:.3f})")

    logline("\n#### D. Huber ハイパラ sweep")
    besthp = (base, 1.35, 1e-4)
    for eps in [1.1, 1.35, 2.0, 3.0]:
        for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
            e, ef = loo(X, y, ids, lambda eps=eps, alpha=alpha: HuberXY(eps, alpha))
            if e < besthp[0]:
                besthp = (e, eps, alpha)
    logline(f"  best: eps={besthp[1]} alpha={besthp[2]} → 実効={besthp[0]:.3f}cm")

    logline("\n#### E. 後処理 trimmed-mean")
    e, ef = loo(X, y, ids, HuberXY, post='trim')
    logline(f"  点内trimmed-mean(20-80%): 実効={e:.3f}cm (frame {ef:.3f})")

    logline("\n#### A+D 合わせ技: まばたき除去 + best Huber")
    e, ef = loo(X, y, ids, lambda: HuberXY(besthp[1], besthp[2]), clean=clean_blink(0.2))
    logline(f"  実効={e:.3f}cm (frame {ef:.3f})")
    logline("\n(段階2完了)")


class _ScaledKNN:
    def __init__(self): self.sc = StandardScaler()
    def fit(self, X, y):
        Xs = self.sc.fit_transform(X)
        self.m = KNeighborsRegressor(20, weights='distance').fit(Xs, y); return self
    def predict(self, X): return self.m.predict(self.sc.transform(X))


class _ScaledGBM:
    def __init__(self): self.sc = StandardScaler()
    def fit(self, X, y):
        Xs = self.sc.fit_transform(X)
        self.mx = GradientBoostingRegressor(random_state=0).fit(Xs, y[:, 0])
        self.my = GradientBoostingRegressor(random_state=0).fit(Xs, y[:, 1]); return self
    def predict(self, X):
        Xs = self.sc.transform(X)
        return np.column_stack([self.mx.predict(Xs), self.my.predict(Xs)])


if __name__ == "__main__":
    main()
