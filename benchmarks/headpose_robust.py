"""
頭部姿勢ロバスト性の層別評価。

ユーザ最重要要件:
  「顔が正面のときは良いが、顔を横にずらしても視線がズレないようにしたい」

方法:
  1. 各被験者の頭部姿勢分布を確認 (pitch/yaw)
  2. frontal(正面) フレームでキャリブ → turned(横向き) フレームで評価
     これは実アプリの状況を模す: キャリブは正面で行い、その後首を振る。
  3. 頭部姿勢を無視する 2D虹彩 と、頭部姿勢を使う各モデルを比較。
     turned bin で誤差が増えないモデルが目標。

7D: X=[Lx,Ly,Rx,Ry,Pitch,Yaw,dist]  (Pitch=X[4], Yaw=X[5], rad)

Usage:
    .venv/Scripts/python.exe benchmarks/headpose_robust.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

PROJECT_DIR = Path(__file__).parent.parent
CACHE_7D    = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
Z_FACE_CM   = 50.0


def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))


# ─── 特徴量 ───────────────────────────────────────────────────────────────
def feat_2d(X):
    return np.column_stack([(X[:, 0] + X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2])

def feat_2d_pose(X):
    """2D虹彩 + pitch,yaw を線形項として追加 (N,4)"""
    xf = (X[:, 0] + X[:, 2]) / 2
    yf = (X[:, 1] + X[:, 3]) / 2
    return np.column_stack([xf, yf, X[:, 4], X[:, 5]])

def feat_2d_pose_inter(X):
    """2D虹彩 + pose + 虹彩×pose交互作用。首振り時の虹彩位置の意味変化を線形補正。"""
    xf = (X[:, 0] + X[:, 2]) / 2
    yf = (X[:, 1] + X[:, 3]) / 2
    p, y = X[:, 4], X[:, 5]
    return np.column_stack([xf, yf, p, y, xf*y, yf*p, xf*p, yf*y])

def feat_full7(X):
    return X.copy()


# ─── モデル ───────────────────────────────────────────────────────────────
def cal_affine(Fc, yc, Fe):
    D = np.column_stack([Fc, np.ones(len(Fc))])
    A, *_ = np.linalg.lstsq(D, yc, rcond=None)
    return np.column_stack([Fe, np.ones(len(Fe))]) @ A

def make_ridge(degree, alpha_base):
    def _fn(Fc, yc, Fe):
        alpha = max(alpha_base, 5000.0 / len(Fc))
        sc = StandardScaler()
        Xc = sc.fit_transform(Fc); Xe = sc.transform(Fe)
        if degree > 1:
            pf = PolynomialFeatures(degree=degree, include_bias=False)
            Xc = pf.fit_transform(Xc); Xe = pf.transform(Xe)
        r = Ridge(alpha=alpha); r.fit(Xc, yc)
        return r.predict(Xe)
    return _fn


# ─── 頭部姿勢分布の確認 ────────────────────────────────────────────────────
def analyze_pose(X, subj):
    pitch_deg = np.degrees(X[:, 4]); yaw_deg = np.degrees(X[:, 5])
    mag = np.sqrt(pitch_deg**2 + yaw_deg**2)
    print("頭部姿勢分布 (全フレーム):")
    print(f"  |pitch|: p50={np.percentile(np.abs(pitch_deg),50):.1f}deg  "
          f"p90={np.percentile(np.abs(pitch_deg),90):.1f}  max={np.abs(pitch_deg).max():.1f}")
    print(f"  |yaw|  : p50={np.percentile(np.abs(yaw_deg),50):.1f}deg  "
          f"p90={np.percentile(np.abs(yaw_deg),90):.1f}  max={np.abs(yaw_deg).max():.1f}")
    for thr in (10, 15, 20, 25):
        frac = np.mean(mag > thr)
        print(f"  |pose|>{thr}deg のフレーム割合: {frac*100:.1f}%")
    return mag


# ─── frontal→turned 層別評価 ───────────────────────────────────────────────
def eval_stratified(X, y_cm, subj, feat_fn, cal_fn, turn_thr_deg=15.0, cal_ratio=0.10):
    """
    各被験者: pose magnitude が小さい順にソート。
      下位(=最も正面) cal_ratio 割をキャリブに使用。
      残りを frontal(<thr) と turned(>=thr) に分けて別々に誤差集計。
    """
    pitch = np.degrees(X[:, 4]); yaw = np.degrees(X[:, 5])
    mag = np.sqrt(pitch**2 + yaw**2)

    euc_front, euc_turn = [], []
    n_turn_total = 0
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = X[m], y_cm[m], mag[m]
        n = len(Xs)
        order = np.argsort(ms)              # 正面ほど前
        n_cal = max(6, int(np.ceil(cal_ratio * n)))
        idx_cal = order[:n_cal]
        idx_rest = order[n_cal:]
        if len(idx_rest) < 10:
            continue

        Ff = feat_fn(Xs)
        pred = cal_fn(Ff[idx_cal], yc[idx_cal], Ff[idx_rest])
        gt   = yc[idx_rest]
        euc  = euclidean_cm(pred, gt)

        rest_mag = ms[idx_rest]
        fm = rest_mag < turn_thr_deg
        tm = ~fm
        if fm.sum() >= 5:
            euc_front.append(np.median(euc[fm]))
        if tm.sum() >= 5:
            euc_turn.append(np.median(euc[tm]))
            n_turn_total += tm.sum()

    return {
        "euc_front_med": float(np.median(euc_front)) if euc_front else float('nan'),
        "euc_turn_med":  float(np.median(euc_turn)) if euc_turn else float('nan'),
        "n_front": len(euc_front), "n_turn": len(euc_turn),
        "n_turn_frames": int(n_turn_total),
    }


def main():
    d = np.load(str(CACHE_7D))
    X, y_cm, subj = d["X"], d["y_cm"], d["subj_id"]
    print(f"[Load] {len(X)} frames, {len(np.unique(subj))} subjects\n")
    analyze_pose(X, subj)

    experiments = [
        ("2D iris (頭部姿勢無視/現行)",   feat_2d,           cal_affine),
        ("2D iris + Ridge",             feat_2d,           make_ridge(1, 0.473)),
        ("2D + pose(線形)",             feat_2d_pose,      make_ridge(1, 0.473)),
        ("2D + pose + 交互作用",         feat_2d_pose_inter, make_ridge(1, 0.473)),
        ("7D full + Ridge(deg1)",       feat_full7,        make_ridge(1, 0.473)),
        ("7D full + Ridge(deg2)",       feat_full7,        make_ridge(2, 0.473)),
    ]

    for thr in (15.0, 20.0):
        print(f"\n{'='*82}")
        print(f"  frontalキャリブ → 層別評価  (turned閾値={thr:.0f}deg, cal=正面10%)")
        print(f"{'='*82}")
        print(f"  {'Experiment':<30}  {'Euc_front':>9}  {'Euc_turn':>9}  {'劣化':>7}  {'n_turn':>6}")
        print(f"  {'-'*80}")
        for label, ff, cf in experiments:
            r = eval_stratified(X, y_cm, subj, ff, cf, turn_thr_deg=thr)
            degr = r["euc_turn_med"] - r["euc_front_med"]
            print(f"  {label:<30}  {r['euc_front_med']:>9.3f}  {r['euc_turn_med']:>9.3f}  "
                  f"{degr:>+7.3f}  {r['n_turn']:>6}")
        print(f"  {'-'*80}")
        print(f"  (Euc=cm, 低いほど良い。劣化=turn-front, 小さいほど頭部ロバスト)")


if __name__ == "__main__":
    main()
