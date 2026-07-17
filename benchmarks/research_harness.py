"""
高速研究ハーネス — MediaPipe再実行なしで cache/sota_7d_cache.npz を使い、
特徴量設計・キャリブレーションモデルを数秒で反復実験する。

7D特徴: X = [Lx, Ly, Rx, Ry, Pitch, Yaw, dist]
  Lx,Ly : 左虹彩の正規化位置 (距離不変)
  Rx,Ry : 右虹彩の正規化位置
  Pitch,Yaw : solvePnP頭部姿勢 (rad)
  dist  : 両目間距離 (正規化, 深度プロキシ)

現行ベスト手法 (IrisDepth+AffineCalib) の2D特徴は:
  X_feat = (Lx + Rx)/2,  Y_feat = (Ly + Ry)/2

評価プロトコル (evaluate_eyetrax.py と同一):
  被験者ごと 最初(or ランダム) 5% でキャリブ → 残りで評価
  指標: MGAE_3D (固定Z=50cm), Euc(cm) median

Usage:
    .venv/Scripts/python.exe benchmarks/research_harness.py
    .venv/Scripts/python.exe benchmarks/research_harness.py --split random
"""
import sys
import io
import time
import argparse
from pathlib import Path

# Windows cp932 回避: stdout を UTF-8 に固定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

PROJECT_DIR = Path(__file__).parent.parent
CACHE_7D    = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
Z_FACE_CM   = 50.0
SEED        = 42


# ═══════════════════════════════════════════════════════════════════════════
#  指標
# ═══════════════════════════════════════════════════════════════════════════

def euclidean_cm(pred_cm, gt_cm):
    return np.sqrt(np.sum((pred_cm - gt_cm) ** 2, axis=-1))


def compute_3d_mgae_fixed(pred_cm, gt_cm, z_face=Z_FACE_CM):
    g_pred = np.column_stack([pred_cm[:, 0], pred_cm[:, 1], np.full(len(pred_cm), -z_face)])
    g_gt   = np.column_stack([gt_cm[:, 0],   gt_cm[:, 1],   np.full(len(gt_cm),   -z_face)])
    n_pred = np.linalg.norm(g_pred, axis=-1, keepdims=True) + 1e-8
    n_gt   = np.linalg.norm(g_gt,   axis=-1, keepdims=True) + 1e-8
    cos = np.sum((g_pred / n_pred) * (g_gt / n_gt), axis=-1).clip(-1 + 1e-7, 1 - 1e-7)
    return np.degrees(np.arccos(cos))


# ═══════════════════════════════════════════════════════════════════════════
#  特徴量トランスフォーム
# ═══════════════════════════════════════════════════════════════════════════

def feat_2d_irisdepth(X):
    """現行ベスト: X_feat=(Lx+Rx)/2, Y_feat=(Ly+Ry)/2  → (N,2)"""
    return np.column_stack([(X[:, 0] + X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2])


def feat_2d_plus_pitch(X):
    """2D虹彩 + pitch → (N,3)  (Y補正用)。X[4]=Pitch"""
    xf = (X[:, 0] + X[:, 2]) / 2
    yf = (X[:, 1] + X[:, 3]) / 2
    return np.column_stack([xf, yf, X[:, 4]])


def feat_full7(X):
    return X.copy()


# ═══════════════════════════════════════════════════════════════════════════
#  キャリブレーションモデル (fit(Xcal,ycal)->predict(Xev)->pred_cm)
# ═══════════════════════════════════════════════════════════════════════════

def cal_affine(Xf_cal, y_cal, Xf_ev):
    """アフィン (lstsq): 現行手法相当。 [f..., 1] @ A"""
    D = np.column_stack([Xf_cal, np.ones(len(Xf_cal))])
    A, *_ = np.linalg.lstsq(D, y_cal, rcond=None)
    De = np.column_stack([Xf_ev, np.ones(len(Xf_ev))])
    return De @ A


def make_ridge_poly(degree, alpha_base):
    def _fn(Xf_cal, y_cal, Xf_ev):
        n_cal = len(Xf_cal)
        alpha = max(alpha_base, 5000.0 / n_cal)
        sc = StandardScaler()
        Xc = sc.fit_transform(Xf_cal)
        Xe = sc.transform(Xf_ev)
        if degree > 1:
            pf = PolynomialFeatures(degree=degree, include_bias=False)
            Xc = pf.fit_transform(Xc)
            Xe = pf.transform(Xe)
        r = Ridge(alpha=alpha)
        r.fit(Xc, y_cal)
        return r.predict(Xe)
    return _fn


# ═══════════════════════════════════════════════════════════════════════════
#  評価ループ
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(X, y_cm, subj_ids, feat_fn, cal_fn, split="first", cal_ratio=0.05):
    """被験者ごとにキャリブ/評価を分けて誤差を集計。pred_cm を直接フィット。"""
    rng = np.random.RandomState(SEED)
    euc_med_per_subj, mgae_mean_per_subj = [], []

    for sid in np.unique(subj_ids):
        mask = subj_ids == sid
        Xs, yc = X[mask], y_cm[mask]
        n = len(Xs)
        n_cal = max(5, int(np.ceil(cal_ratio * n)))
        if n - n_cal < 10:
            continue

        if split == "first":
            idx_cal = np.arange(n_cal)
            idx_ev  = np.arange(n_cal, n)
        else:  # random
            perm = rng.permutation(n)
            idx_cal, idx_ev = perm[:n_cal], perm[n_cal:]

        Ff = feat_fn(Xs)
        pred_cm = cal_fn(Ff[idx_cal], yc[idx_cal], Ff[idx_ev])
        gt_cm   = yc[idx_ev]

        euc  = euclidean_cm(pred_cm, gt_cm)
        mgae = compute_3d_mgae_fixed(pred_cm, gt_cm)
        euc_med_per_subj.append(np.median(euc))
        mgae_mean_per_subj.append(np.mean(mgae))

    return {
        "n_subj":      len(euc_med_per_subj),
        "euc_med":     float(np.median(euc_med_per_subj)),
        "euc_mean":    float(np.mean(euc_med_per_subj)),
        "mgae3d_mean": float(np.mean(mgae_mean_per_subj)),
        "mgae3d_med":  float(np.median(mgae_mean_per_subj)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  実験定義
# ═══════════════════════════════════════════════════════════════════════════

def build_experiments():
    return [
        # (ラベル, feat_fn, cal_fn)
        ("2D iris + Affine (現行ベスト相当)", feat_2d_irisdepth, cal_affine),
        ("2D iris + Ridge(deg1)",           feat_2d_irisdepth, make_ridge_poly(1, 0.473)),
        ("2D iris + Ridge(deg2)",           feat_2d_irisdepth, make_ridge_poly(2, 0.473)),
        ("2D+pitch + Ridge(deg2)",          feat_2d_plus_pitch, make_ridge_poly(2, 0.473)),
        ("7D full + Ridge(deg1)",           feat_full7,        make_ridge_poly(1, 0.473)),
        ("7D full + Ridge(deg2)",           feat_full7,        make_ridge_poly(2, 0.473)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["first", "random"], default="first")
    args = ap.parse_args()

    d = np.load(str(CACHE_7D))
    X, y_cm, subj = d["X"], d["y_cm"], d["subj_id"]
    print(f"[Load] {CACHE_7D.name}: {len(X)} frames, {len(np.unique(subj))} subjects")
    print(f"[Feature index check] mean|std per dim:")
    names = ["Lx", "Ly", "Rx", "Ry", "Pitch", "Yaw", "dist"]
    for i, nm in enumerate(names):
        print(f"    X[{i}] {nm}: mean={X[:,i].mean():+.3f} std={X[:,i].std():.3f} "
              f"range=[{X[:,i].min():+.2f},{X[:,i].max():+.2f}]")

    print(f"\n{'='*78}")
    print(f"  Research Harness — split={args.split}, protocol=perSubj {int(0.05*100)}% cal")
    print(f"{'='*78}")
    print(f"  {'Experiment':<38}  {'n':>3}  {'Euc_med':>8}  {'Euc_mean':>8}  {'MGAE3d':>7}")
    print(f"  {'-'*76}")

    t0 = time.time()
    for label, feat_fn, cal_fn in build_experiments():
        r = evaluate(X, y_cm, subj, feat_fn, cal_fn, split=args.split)
        print(f"  {label:<38}  {r['n_subj']:>3}  {r['euc_med']:>8.3f}  "
              f"{r['euc_mean']:>8.3f}  {r['mgae3d_mean']:>7.2f}")
    print(f"  {'-'*76}")
    print(f"  [{time.time()-t0:.1f}s]  (Euc_med = lower is better, cm)")


if __name__ == "__main__":
    main()
