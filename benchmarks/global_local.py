"""
グローバル事前学習 + ユーザ個別補正 (頭部姿勢ロバスト化の本命)

課題(ユーザ最重要):
  個人キャリブは正面でしか取れない → 正面のみで学習したモデルは
  横向きへ外挿すると破綻する (headpose_robust.py で実証済み)。

解決:
  「視線特徴が頭部姿勢でどう変わるか」は 264k フレーム(多姿勢)の
  train split から グローバル に事前学習する。
  ユーザ個別キャリブ(正面のみ)は、そのグローバル予測に対する
  小さな個人補正(カッパ角オフセット等)だけを担う。
  → 姿勢依存はグローバルが吸収し、個人補正は姿勢に依らないので
    正面キャリブでも横向きでズレにくい。

データ:
  train (global) : cache/gazeCapture_features_cache.npz  split_code==0 (264k)
  test  (per-sub): cache/sota_7d_cache.npz  (26被験者, subj_id有り)

7D: X=[Lx,Ly,Rx,Ry,Pitch,Yaw,dist]  Pitch=X[4],Yaw=X[5] (rad)

Usage:
    .venv/Scripts/python.exe benchmarks/global_local.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).parent.parent
CACHE_BIG   = PROJECT_DIR / "cache" / "gazeCapture_features_cache.npz"
CACHE_TEST  = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
Z_FACE_CM   = 50.0


def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))

def feat_2d(X):
    return np.column_stack([(X[:, 0] + X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2])


# ─── グローバルモデル ──────────────────────────────────────────────────────
class GlobalModel:
    """7D → y_cm を大規模train splitで学習。standardize + Ridge。"""
    def __init__(self, alpha=10.0):
        self.sc = StandardScaler()
        self.r  = Ridge(alpha=alpha)
    def fit(self, X, y_cm):
        self.r.fit(self.sc.fit_transform(X), y_cm)
        return self
    def predict(self, X):
        return self.r.predict(self.sc.transform(X))


# ─── 個人補正のバリエーション ──────────────────────────────────────────────
def correct_none(gp_cal, y_cal, F2d_cal, gp_ev, F2d_ev):
    """補正なし: グローバル予測そのまま"""
    return gp_ev

def correct_offset(gp_cal, y_cal, F2d_cal, gp_ev, F2d_ev):
    """平均オフセット補正: 個人のカッパ角=一定バイアスを除去。姿勢非依存。"""
    off = np.mean(y_cal - gp_cal, axis=0)
    return gp_ev + off

def correct_affine2d(gp_cal, y_cal, F2d_cal, gp_ev, F2d_ev):
    """
    残差を2D虹彩特徴の線形関数でフィット (残差アフィン)。
    residual = y - global_pred ~ A[iris,1]。姿勢項を含まないので外挿に安全。
    """
    resid = y_cal - gp_cal
    D = np.column_stack([F2d_cal, np.ones(len(F2d_cal))])
    A, *_ = np.linalg.lstsq(D, resid, rcond=None)
    De = np.column_stack([F2d_ev, np.ones(len(F2d_ev))])
    return gp_ev + De @ A

def make_correct_blend(alpha_base):
    """個人フルモデル(2D iris affine)とグローバルを信頼度でブレンド。"""
    def _fn(gp_cal, y_cal, F2d_cal, gp_ev, F2d_ev):
        # 個人モデル: 2D iris affine to y_cm
        D = np.column_stack([F2d_cal, np.ones(len(F2d_cal))])
        A, *_ = np.linalg.lstsq(D, y_cal, rcond=None)
        personal_ev = np.column_stack([F2d_ev, np.ones(len(F2d_ev))]) @ A
        # ブレンド重み: キャリブ点数が少ないほどグローバル寄り
        w = min(1.0, len(F2d_cal) / 200.0) * 0.5
        return w * personal_ev + (1 - w) * gp_ev
    return _fn


# ─── frontal→turned 層別評価 ───────────────────────────────────────────────
def eval_pure_local(Xt, yct, subj, turn_thr_deg=20.0, cal_ratio=0.10):
    """基準: 純ローカル 2D iris affine (グローバル無し, 現行相当)"""
    pitch = np.degrees(Xt[:, 4]); yaw = np.degrees(Xt[:, 5])
    mag = np.sqrt(pitch**2 + yaw**2)
    euc_front, euc_turn = [], []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = Xt[m], yct[m], mag[m]
        n = len(Xs)
        order = np.argsort(ms)
        n_cal = max(6, int(np.ceil(cal_ratio * n)))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10:
            continue
        F2d = feat_2d(Xs)
        D = np.column_stack([F2d[idx_cal], np.ones(n_cal)])
        A, *_ = np.linalg.lstsq(D, yc[idx_cal], rcond=None)
        pred = np.column_stack([F2d[idx_rest], np.ones(len(idx_rest))]) @ A
        euc = euclidean_cm(pred, yc[idx_rest])
        rm = ms[idx_rest]; fm = rm < turn_thr_deg
        if fm.sum() >= 5: euc_front.append(np.median(euc[fm]))
        if (~fm).sum() >= 5: euc_turn.append(np.median(euc[~fm]))
    return (float(np.median(euc_front)) if euc_front else float('nan'),
            float(np.median(euc_turn)) if euc_turn else float('nan'))


def eval_global_local(Xt, yct, subj, gm, correct_fn,
                      turn_thr_deg=20.0, cal_ratio=0.10):
    pitch = np.degrees(Xt[:, 4]); yaw = np.degrees(Xt[:, 5])
    mag = np.sqrt(pitch**2 + yaw**2)
    euc_front, euc_turn = [], []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = Xt[m], yct[m], mag[m]
        n = len(Xs)
        order = np.argsort(ms)
        n_cal = max(6, int(np.ceil(cal_ratio * n)))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10:
            continue
        F2d = feat_2d(Xs)
        gp_all = gm.predict(Xs)
        pred = correct_fn(gp_all[idx_cal], yc[idx_cal], F2d[idx_cal],
                          gp_all[idx_rest], F2d[idx_rest])
        euc = euclidean_cm(pred, yc[idx_rest])
        rm = ms[idx_rest]
        fm = rm < turn_thr_deg
        if fm.sum() >= 5: euc_front.append(np.median(euc[fm]))
        if (~fm).sum() >= 5: euc_turn.append(np.median(euc[~fm]))
    return (float(np.median(euc_front)) if euc_front else float('nan'),
            float(np.median(euc_turn)) if euc_turn else float('nan'))


def main():
    t0 = time.time()
    # ── グローバル学習データ (train split)
    db = np.load(str(CACHE_BIG))
    Xb, ycb, scb = db["X"], db["y_cm"], db["split_code"]
    tr = scb == 0
    print(f"[Global train] split=0: {tr.sum()} frames")
    gm = GlobalModel(alpha=10.0).fit(Xb[tr], ycb[tr])

    # グローバル単体の素の精度 (test全体)
    dt = np.load(str(CACHE_TEST))
    Xt, yct, subj = dt["X"], dt["y_cm"], dt["subj_id"]
    gp = gm.predict(Xt)
    print(f"[Global raw on test] Euc mean={euclidean_cm(gp,yct).mean():.3f}cm "
          f"median={np.median(euclidean_cm(gp,yct)):.3f}cm  (キャリブ無し!)")

    experiments = [
        ("Global raw (無キャリブ)",        correct_none),
        ("Global + 平均オフセット",         correct_offset),
        ("Global + 残差アフィン2D",         correct_affine2d),
        ("Global + 個人ブレンド",           make_correct_blend(0.473)),
    ]

    for thr in (15.0, 20.0):
        print(f"\n{'='*74}")
        print(f"  Global+Local  frontalキャリブ→層別  (turn>={thr:.0f}deg, cal=正面10%)")
        print(f"{'='*74}")
        print(f"  {'手法':<28}  {'Euc_front':>9}  {'Euc_turn':>9}  {'劣化':>7}")
        print(f"  {'-'*72}")
        # 比較用: 純ローカル 2D iris affine (現行相当) — インライン
        ef_b, et_b = eval_pure_local(Xt, yct, subj, turn_thr_deg=thr)
        print(f"  {'[基準] 純ローカル2D iris':<28}  {ef_b:>9.3f}  "
              f"{et_b:>9.3f}  {et_b-ef_b:>+7.3f}")
        for label, cf in experiments:
            ef, et = eval_global_local(Xt, yct, subj, gm, cf, turn_thr_deg=thr)
            print(f"  {label:<28}  {ef:>9.3f}  {et:>9.3f}  {et-ef:>+7.3f}")
        print(f"  {'-'*72}")
    print(f"\n[{time.time()-t0:.1f}s] Euc=cm 低いほど良い / 劣化=turn-front 小さいほど頭部ロバスト")


if __name__ == "__main__":
    main()
