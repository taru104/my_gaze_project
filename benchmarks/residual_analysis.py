"""
グローバルMLPの残差分析: どこで誤るかを特定し次の改善の的を絞る。
保存済みモデルを使うので軽量(学習なし)。

観点:
  - 被験者別の誤差ばらつき (どの被験者で悪いか)
  - 誤差 vs 各特徴 の相関 (何が誤差を生むか)
  - 大誤差フレームの特徴傾向

Usage:
    .venv/Scripts/python.exe benchmarks/residual_analysis.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
import joblib

class Ensemble:
    def __init__(self, models): self.models = models
    def predict(self, X): return np.mean([m.predict(X) for m in self.models], axis=0)

ROOT = Path(__file__).parent.parent
CACHE_7D = ROOT / "cache" / "sota_7d_cache.npz"
MODEL = ROOT / "cache" / "global_mlp_v2.joblib"


def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))


def main():
    gm = joblib.load(MODEL)
    d = np.load(str(CACHE_7D))
    X, ycm, subj = d["X"], d["y_cm"], d["subj_id"]
    pred = gm.predict(X)
    euc = euclidean_cm(pred, ycm)
    names = ['Lx','Ly','Rx','Ry','Pitch','Yaw','dist']

    print(f"全体: median={np.median(euc):.3f} mean={euc.mean():.3f} p90={np.percentile(euc,90):.3f} cm\n")

    # 被験者別
    print("被験者別 median誤差 (悪い順 top8):")
    rows = []
    for sid in np.unique(subj):
        m = subj == sid
        rows.append((sid, np.median(euc[m]), m.sum()))
    rows.sort(key=lambda r: -r[1])
    for sid, med, n in rows[:8]:
        print(f"  subj {sid}: median={med:.2f}cm  n={n}")
    print(f"  ... 最良: subj {rows[-1][0]} median={rows[-1][1]:.2f}cm")

    # 誤差と特徴の相関 (絶対誤差 vs 各特徴の絶対値)
    print("\n誤差(euc) と 各特徴 の相関:")
    for i, nm in enumerate(names):
        c = np.corrcoef(np.abs(X[:, i]), euc)[0, 1]
        print(f"  |{nm}|: r={c:+.3f}")
    # pose magnitudeとの相関
    mag = np.sqrt(np.degrees(X[:,4])**2 + np.degrees(X[:,5])**2)
    print(f"  pose_mag: r={np.corrcoef(mag, euc)[0,1]:+.3f}")
    # EAR代替: Ly-Ryの垂直差など無いので dist
    print(f"  dist: r={np.corrcoef(X[:,6], euc)[0,1]:+.3f}")

    # 大誤差フレーム(top10%)の特徴傾向
    thr = np.percentile(euc, 90)
    hi = euc >= thr
    print(f"\n大誤差フレーム(top10%, euc>={thr:.1f}cm) の特徴平均 vs 全体:")
    for i, nm in enumerate(names):
        print(f"  {nm}: 大誤差={X[hi,i].mean():+.3f}  全体={X[:,i].mean():+.3f}")
    print(f"  pose_mag: 大誤差={mag[hi].mean():.1f}deg  全体={mag.mean():.1f}deg")

    # 方向別誤差 (x/y どちらが悪いか)
    ex = np.abs(pred[:,0]-ycm[:,0]); ey = np.abs(pred[:,1]-ycm[:,1])
    print(f"\n方向別 median絶対誤差: X={np.median(ex):.3f}cm  Y={np.median(ey):.3f}cm")
    print(f"  → 大きい方が改善余地。")


if __name__ == "__main__":
    main()
