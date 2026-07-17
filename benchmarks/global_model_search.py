"""
グローバルモデルの表現力探索。
264k フレームあるので線形Ridgeより表現力の高いモデルを試す。
各モデルで (1)無キャリブ精度 (2)頭部姿勢ロバスト性 を測る。

7D: X=[Lx,Ly,Rx,Ry,Pitch,Yaw,dist]

Usage:
    .venv/Scripts/python.exe benchmarks/global_model_search.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

PROJECT_DIR = Path(__file__).parent.parent
CACHE_BIG   = PROJECT_DIR / "cache" / "gazeCapture_features_cache.npz"
CACHE_TEST  = PROJECT_DIR / "cache" / "sota_7d_cache.npz"


def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))


def robustness_eval(model, Xt, yct, subj, turn_thr=20.0):
    """無キャリブでのfrontal/turned層別誤差 (グローバル素の性能)"""
    pitch = np.degrees(Xt[:, 4]); yaw = np.degrees(Xt[:, 5])
    mag = np.sqrt(pitch**2 + yaw**2)
    pred = model.predict(Xt)
    euc = euclidean_cm(pred, yct)
    ef, et = [], []
    for sid in np.unique(subj):
        m = subj == sid
        e, ms = euc[m], mag[m]
        fm = ms < turn_thr
        if fm.sum() >= 5: ef.append(np.median(e[fm]))
        if (~fm).sum() >= 5: et.append(np.median(e[~fm]))
    return (float(np.median(ef)), float(np.median(et)),
            float(np.median(euc)), float(np.mean(euc)))


def build_models():
    return [
        ("Ridge(a=10) linear",       make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("Ridge(a=1) linear",        make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Ridge poly2",              make_pipeline(StandardScaler(),
                                       PolynomialFeatures(2, include_bias=False),
                                       Ridge(alpha=10.0))),
        ("Ridge poly3",              make_pipeline(StandardScaler(),
                                       PolynomialFeatures(3, include_bias=False),
                                       Ridge(alpha=20.0))),
        ("MLP(64,32)",               make_pipeline(StandardScaler(),
                                       MLPRegressor(hidden_layer_sizes=(64, 32),
                                                    activation="relu", alpha=1e-3,
                                                    max_iter=60, early_stopping=True,
                                                    random_state=0))),
        ("MLP(128,64,32)",           make_pipeline(StandardScaler(),
                                       MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                                    activation="relu", alpha=1e-3,
                                                    max_iter=80, early_stopping=True,
                                                    random_state=0))),
    ]


def main():
    t0 = time.time()
    db = np.load(str(CACHE_BIG))
    Xb, ycb, scb = db["X"], db["y_cm"], db["split_code"]
    tr = scb == 0
    Xtr, ytr = Xb[tr], ycb[tr]
    print(f"[Train] {len(Xtr)} frames (split=0)")

    dt = np.load(str(CACHE_TEST))
    Xt, yct, subj = dt["X"], dt["y_cm"], dt["subj_id"]
    print(f"[Test]  {len(Xt)} frames, {len(np.unique(subj))} subjects\n")

    print(f"{'='*84}")
    print(f"  グローバルモデル探索 (無キャリブ性能 + 頭部姿勢ロバスト性)")
    print(f"{'='*84}")
    print(f"  {'Model':<22}  {'Euc_med':>8}  {'Euc_mean':>8}  {'front':>7}  {'turn':>7}  {'劣化':>6}  {'fit_s':>6}")
    print(f"  {'-'*82}")

    results = []
    for name, model in build_models():
        ts = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - ts
        ef, et, med, mean = robustness_eval(model, Xt, yct, subj)
        print(f"  {name:<22}  {med:>8.3f}  {mean:>8.3f}  {ef:>7.3f}  {et:>7.3f}  "
              f"{et-ef:>+6.3f}  {fit_s:>6.1f}")
        results.append((name, med, mean, ef, et, et-ef))
    print(f"  {'-'*82}")
    print(f"\n[{time.time()-t0:.1f}s] Euc=cm median, 低いほど良い / 劣化=turn-front")

    best = min(results, key=lambda r: r[4])  # turn誤差最小
    print(f"\n★ 横向き最良: {best[0]}  (turn={best[4]:.3f}cm, 劣化={best[5]:+.3f})")


if __name__ == "__main__":
    main()
