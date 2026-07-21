"""MPII汎用モデル(他人15人)を「あなたのデータ」に適用し、転移するか検証。
転移すれば = 他人で学んだモデルがあなたで動く = 真の汎用性。
GazeCapture(モバイル)は転移せず10.88cmだった。MPIIはノートPC webカメラなので近いはず。

3条件で各ユーザセッションを評価:
  A. MPII汎用そのまま (キャリブ無し)
  B. MPII汎用 + あなたの少数点でアフィン適応
  C. あなた個人H1 (参考: 個人特化の実力, 点ごとLOO)
"""
import sys, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4])  # あなたの画面(cm)

def euc_cm(P, G): return np.linalg.norm((P - G) * CM, axis=1)

# --- MPII汎用モデル(全15人で学習) ---
d = np.load(ROOT / "cache" / "mpii_7d.npz")
Xm, ym = d["X"], d["y"]
sc = StandardScaler().fit(Xm)
gmx = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:, 0])
gmy = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:, 1])
def gen(X): return np.column_stack([gmx.predict(sc.transform(X)), gmy.predict(sc.transform(X))])
print(f"MPII汎用モデル学習: {len(Xm)}フレーム/15人\n")

def affine(base_tr, ytr, base_te):
    A = np.hstack([base_tr, np.ones((len(base_tr), 1))])
    W, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.hstack([base_te, np.ones((len(base_te), 1))]) @ W

def h1_loo(X, y):
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    PM, GM = [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        if tr.sum() < 20: continue
        m = H1Calibration()
        for f_, t in zip(X[tr], y[tr]): m.add(f_, t[0], t[1], 1.0)
        try: m.fit()
        except Exception: continue
        pr = np.array([m.predict(f_) for f_ in X[te]])
        PM.append(np.median(pr, axis=0)); GM.append(np.median(y[te], axis=0))
    return np.median(euc_cm(np.array(PM), np.array(GM)))

print(f"{'session':22s} {'n':>5s}  {'A:汎用ｷｬﾘﾌﾞ無':>12s} {'B:汎用+適応':>11s} {'C:個人H1':>9s}")
rng = np.random.RandomState(0)
for f in sorted(glob.glob(str(ROOT / "logs" / "session_*_rich16d.npz"))):
    d = np.load(f); m = d["has_target"].astype(bool)
    Xu, yu = d["X"][m][:, :7], d["y_norm"][m]
    if len(Xu) < 100: continue
    # A: 汎用そのまま
    a = np.median(euc_cm(gen(Xu), yu))
    # B: 汎用 + 少数点アフィン適応(先頭50点で適応, 残りで評価)
    base = gen(Xu)
    idx = rng.permutation(len(Xu)); cal, ev = idx[:50], idx[50:]
    b = np.median(euc_cm(affine(base[cal], yu[cal], base[ev]), yu[ev]))
    # C: 個人H1
    c = h1_loo(Xu, yu)
    name = Path(f).name.replace("session_", "").replace("_rich16d.npz", "")
    print(f"{name:22s} {len(Xu):5d}  {a:10.2f}cm {b:9.2f}cm {c:7.2f}cm")
print("\n→ A(汎用キャリブ無し)が実用圏(〜数cm)なら、他人モデルがあなたで動く=転移成功=真の汎用性。")
print("  B(汎用+少数適応)がC(個人H1)に近ければ、汎用ベース+軽い適応で個人特化に迫れる。")
