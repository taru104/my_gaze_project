"""
極端な頭部姿勢でのロバスト性確認。
グローバルMLPが >25°,>30°,>35° でも破綻しないかをbin別に検証。
現行(純ローカル2D iris)と比較。

Usage:
    .venv/Scripts/python.exe benchmarks/extreme_pose_check.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
import joblib

PROJECT_DIR = Path(__file__).parent.parent
CACHE_TEST  = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
MODEL_IN    = PROJECT_DIR / "cache" / "global_mlp.joblib"


def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))

def feat_2d(X):
    return np.column_stack([(X[:, 0] + X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2])


def main():
    model = joblib.load(MODEL_IN)
    d = np.load(str(CACHE_TEST))
    Xt, yct, subj = d["X"], d["y_cm"], d["subj_id"]
    pitch = np.degrees(Xt[:, 4]); yaw = np.degrees(Xt[:, 5])
    mag = np.sqrt(pitch**2 + yaw**2)

    gp = model.predict(Xt)
    euc_global = euclidean_cm(gp, yct)

    # 現行相当: 各被験者 正面10%キャリブ→ローカル2D affine
    euc_local = np.full(len(Xt), np.nan)
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = Xt[m], yct[m], mag[m]
        F = feat_2d(Xs)
        order = np.argsort(ms)
        n_cal = max(6, int(np.ceil(0.10 * len(Xs))))
        idx_cal = order[:n_cal]
        D = np.column_stack([F[idx_cal], np.ones(n_cal)])
        A, *_ = np.linalg.lstsq(D, yc[idx_cal], rcond=None)
        pred = np.column_stack([F, np.ones(len(F))]) @ A
        euc_local[np.where(m)[0]] = euclidean_cm(pred, yc)

    bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 90)]
    print(f"{'='*72}")
    print(f"  頭部姿勢bin別 誤差 (cm median) — グローバルMLP vs 現行ローカル")
    print(f"{'='*72}")
    print(f"  {'pose bin(deg)':<14}  {'frames':>7}  {'Global':>8}  {'Local(現行)':>11}  {'改善':>7}")
    print(f"  {'-'*70}")
    for lo, hi in bins:
        bm = (mag >= lo) & (mag < hi)
        if bm.sum() < 20:
            continue
        g = np.median(euc_global[bm])
        l = np.median(euc_local[bm])
        print(f"  {f'[{lo:>2},{hi:>2})':<14}  {bm.sum():>7}  {g:>8.3f}  {l:>11.3f}  "
              f"{(l-g)/l*100:>6.1f}%")
    print(f"  {'-'*70}")
    # 全体
    print(f"  {'全体':<14}  {len(Xt):>7}  {np.median(euc_global):>8.3f}  "
          f"{np.median(euc_local):>11.3f}  {(np.median(euc_local)-np.median(euc_global))/np.median(euc_local)*100:>6.1f}%")
    print(f"\n  Global=無キャリブMLP, Local=正面10%キャリブ+2D affine(現行相当)")


if __name__ == "__main__":
    main()
