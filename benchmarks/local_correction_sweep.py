"""
保存済みグローバルMLPに対する個人補正の設計探索。
「なぜ正面補正が横向きロバスト性を壊すか」を踏まえ、壊さない補正を探す。

仮説: グローバルは既に姿勢ロバスト。個人差はほぼ一定のカッパ角オフセット。
  → 補正は「姿勢非依存の小さな平行移動」に留めるべき。
  → 過度な補正(残差アフィンや大きなオフセット)は正面ノイズに過適合し横向きで害。

対策: オフセットの収縮 k、残差アフィンのRidge収縮を掃引。

Usage:
    .venv/Scripts/python.exe benchmarks/local_correction_sweep.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
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


# ─── 補正関数群 (すべて global予測 gp に対する後処理) ───────────────────────
def corr_none(gp_cal, y_cal, F_cal, gp_ev, F_ev, **kw):
    return gp_ev

def corr_offset_shrunk(gp_cal, y_cal, F_cal, gp_ev, F_ev, k=0.5, **kw):
    off = np.mean(y_cal - gp_cal, axis=0)
    return gp_ev + k * off

def corr_offset_median(gp_cal, y_cal, F_cal, gp_ev, F_ev, k=1.0, **kw):
    off = np.median(y_cal - gp_cal, axis=0)  # 外れ値に強い
    return gp_ev + k * off

def corr_resid_affine_reg(gp_cal, y_cal, F_cal, gp_ev, F_ev, lam=50.0, **kw):
    """残差 = A[iris,1] をRidgeで(強収縮)。lam大→ほぼオフセットのみ。"""
    resid = y_cal - gp_cal
    D = np.column_stack([F_cal, np.ones(len(F_cal))])
    # Ridge closed form: A=(D'D+lam I)^-1 D'resid  (biasは正則化しない)
    n_feat = D.shape[1]
    R = lam * np.eye(n_feat); R[-1, -1] = 0.0
    A = np.linalg.solve(D.T @ D + R, D.T @ resid)
    De = np.column_stack([F_ev, np.ones(len(F_ev))])
    return gp_ev + De @ A


def eval_corr(model, Xt, yct, subj, corr_fn, turn_thr=20.0, cal_ratio=0.10, **kw):
    pitch = np.degrees(Xt[:, 4]); yaw = np.degrees(Xt[:, 5])
    mag = np.sqrt(pitch**2 + yaw**2)
    gp_all_full = model.predict(Xt)
    ef, et = [], []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = Xt[m], yct[m], mag[m]
        gp = gp_all_full[m]
        F = feat_2d(Xs)
        n = len(Xs); order = np.argsort(ms)
        n_cal = max(6, int(np.ceil(cal_ratio * n)))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10:
            continue
        pred = corr_fn(gp[idx_cal], yc[idx_cal], F[idx_cal],
                       gp[idx_rest], F[idx_rest], **kw)
        euc = euclidean_cm(pred, yc[idx_rest])
        rm = ms[idx_rest]; fm = rm < turn_thr
        if fm.sum() >= 5: ef.append(np.median(euc[fm]))
        if (~fm).sum() >= 5: et.append(np.median(euc[~fm]))
    return (float(np.median(ef)), float(np.median(et)))


def main():
    model = joblib.load(MODEL_IN)
    dt = np.load(str(CACHE_TEST))
    Xt, yct, subj = dt["X"], dt["y_cm"], dt["subj_id"]
    print(f"[Load] MLP + test {len(Xt)} frames, {len(np.unique(subj))} subjects\n")

    configs = [
        ("補正なし(raw)",              corr_none, {}),
        ("offset k=0.25",             corr_offset_shrunk, {"k": 0.25}),
        ("offset k=0.5",              corr_offset_shrunk, {"k": 0.5}),
        ("offset k=0.75",             corr_offset_shrunk, {"k": 0.75}),
        ("offset k=1.0",              corr_offset_shrunk, {"k": 1.0}),
        ("offset median k=0.5",       corr_offset_median, {"k": 0.5}),
        ("offset median k=1.0",       corr_offset_median, {"k": 1.0}),
        ("残差affine lam=200(強収縮)", corr_resid_affine_reg, {"lam": 200.0}),
        ("残差affine lam=50",         corr_resid_affine_reg, {"lam": 50.0}),
        ("残差affine lam=10",         corr_resid_affine_reg, {"lam": 10.0}),
    ]

    for thr in (15.0, 20.0):
        print(f"{'='*66}")
        print(f"  個人補正掃引  (turn>={thr:.0f}deg, cal=正面10%)")
        print(f"{'='*66}")
        print(f"  {'補正':<26}  {'front':>7}  {'turn':>7}  {'劣化':>7}")
        print(f"  {'-'*64}")
        rows = []
        for label, fn, kw in configs:
            ef, et = eval_corr(model, Xt, yct, subj, fn, turn_thr=thr, **kw)
            print(f"  {label:<26}  {ef:>7.3f}  {et:>7.3f}  {et-ef:>+7.3f}")
            rows.append((label, ef, et))
        best_turn = min(rows, key=lambda r: r[2])
        best_front = min(rows, key=lambda r: r[1])
        print(f"  {'-'*64}")
        print(f"  横向き最良: {best_turn[0]} ({best_turn[2]:.3f}) / "
              f"正面最良: {best_front[0]} ({best_front[1]:.3f})\n")


if __name__ == "__main__":
    main()
