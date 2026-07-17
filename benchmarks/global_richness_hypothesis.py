"""
仮説検証: 豊富特徴(486D)は「グローバル学習」なら横向き精度を上げるか?
(ローカルキャリブでは過学習して逆効果だった。グローバルなら別かを確認。)

方法: 26被験者を subject-held-out で train/test 分割。
  7D-global と 486D-global を同一分割で学習・評価し、bin別に比較。
  486Dが横向きbinで7Dより低ければ → 豊富特徴のフル再抽出に投資する価値あり。

Usage:
    .venv/Scripts/python.exe benchmarks/global_richness_hypothesis.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import euclidean_cm, POSE_BINS

PROJECT_DIR = Path(__file__).parent.parent
CACHE_7D    = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
CACHE_486   = PROJECT_DIR / "cache" / "sota_486d_cache.npz"


def make_mlp(seed=0):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                     alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                     max_iter=120, early_stopping=True, n_iter_no_change=12,
                     random_state=seed),
    )


def eval_global_cv(X, y_cm, subj, mag, n_folds=3):
    """subject-held-out CV。各foldでheld-out被験者のbin別誤差を集める。"""
    sids = np.unique(subj)
    rng = np.random.RandomState(0)
    sids = rng.permutation(sids)
    folds = np.array_split(sids, n_folds)
    per_bin = {b: [] for b in POSE_BINS}
    all_euc = []
    for fold in folds:
        test_mask = np.isin(subj, fold)
        tr_mask = ~test_mask
        model = make_mlp()
        model.fit(X[tr_mask], y_cm[tr_mask])
        pred = model.predict(X[test_mask])
        euc = euclidean_cm(pred, y_cm[test_mask])
        me = mag[test_mask]
        all_euc.append(euc)
        for lo, hi in POSE_BINS:
            bm = (me >= lo) & (me < hi)
            if bm.sum() >= 20:
                per_bin[(lo, hi)].append(np.median(euc[bm]))
    all_euc = np.concatenate(all_euc)
    rep = {b: (float(np.mean(per_bin[b])) if per_bin[b] else None) for b in POSE_BINS}
    rep["all"] = float(np.median(all_euc))
    return rep


def print_row(label, rep):
    s = f"  {label:<14}"
    for b in POSE_BINS:
        v = rep[b]; s += f" {v:>6.2f}" if v is not None else "   -  "
    s += f"  | {rep['all']:.3f}"
    print(s)


def main():
    t0 = time.time()
    d7 = np.load(str(CACHE_7D))
    X7, yc7, sid7 = d7["X"], d7["y_cm"], d7["subj_id"]
    mag7 = np.sqrt(np.degrees(X7[:,4])**2 + np.degrees(X7[:,5])**2)

    d4 = np.load(str(CACHE_486))
    X4, yc4, sid4 = d4["X"], d4["y_cm"], d4["subj_id"]
    yaw4 = np.degrees(X4[:, -3]); pit4 = np.degrees(X4[:, -2])
    mag4 = np.sqrt(pit4**2 + yaw4**2)

    print(f"[Data] 7D:{X7.shape}  486D:{X4.shape}")
    print(f"\n{'='*84}")
    print(f"  グローバル学習の特徴豊富度 (subject-held-out 3-fold CV, bin別 cm)")
    print(f"{'='*84}")
    hdr = f"  {'model':<14}"
    for lo, hi in POSE_BINS: hdr += f" {lo:>2}-{hi:<3}"
    hdr += "  |  all"
    print(hdr); print("  " + "-"*80)

    rep7 = eval_global_cv(X7, yc7, sid7, mag7)
    print_row("7D global", rep7)
    rep4 = eval_global_cv(X4, yc4, sid4, mag4)
    print_row("486D global", rep4)
    print("  " + "-"*80)

    # 横向き(>=20)平均で判定
    turn7 = np.mean([rep7[b] for b in [(20,25),(25,30),(30,40),(40,90)] if rep7[b]])
    turn4 = np.mean([rep4[b] for b in [(20,25),(25,30),(30,40),(40,90)] if rep4[b]])
    print(f"\n  横向き(>=20°)平均: 7D={turn7:.2f}cm  486D={turn4:.2f}cm")
    if turn4 < turn7 - 0.2:
        print(f"  → 豊富特徴がグローバルで有効。フル再抽出に投資する価値あり(タスク#7)")
    else:
        print(f"  → 486Dはグローバルでも横向きを改善せず。7Dで十分。別方向を探る。")
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
