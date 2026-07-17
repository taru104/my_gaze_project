"""
姿勢ゲート・ハイブリッド: 正面はローカルキャリブ(高精度), 横向きはグローバル(頑健)。

動機(feat_richness_test.pyの発見):
  ローカル7D: 正面2.8cm(最良)だが横向き9.5cmに崩壊。
  グローバルMLP: 全域4-5cmで平坦。
  → キャリブ姿勢に近ければローカル, 遠ければグローバルに滑らかに切替。

ブレンド:
  w(m) = exp(-max(0, m - m_cal)/tau)     m=評価フレームの姿勢magnitude(deg)
  m_cal = キャリブ姿勢の代表値(例: 90%ile)
  pred = w*local + (1-w)*global
  正面(m<=m_cal): w≈1 ローカル / 横向き(m>>m_cal): w→0 グローバル

tau を掃引して最適点を探す。

Usage:
    .venv/Scripts/python.exe benchmarks/pose_gated_hybrid.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import euclidean_cm, POSE_BINS


class Ensemble:
    """train_global_v2.py の保存モデルをunpickleするための定義。"""
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

PROJECT_DIR = Path(__file__).parent.parent
CACHE_7D    = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
MODEL_IN    = PROJECT_DIR / "cache" / "global_mlp_v2.joblib"


def eval_hybrid(Xt, yct, subj, gmodel, tau, alpha=5.0, cal_ratio=0.10,
                mcal_pct=90.0):
    mag = np.sqrt(np.degrees(Xt[:,4])**2 + np.degrees(Xt[:,5])**2)
    gp_full = gmodel.predict(Xt)
    per_bin = {b: [] for b in POSE_BINS}
    all_euc = []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms, gp = Xt[m], yct[m], mag[m], gp_full[m]
        order = np.argsort(ms)
        n_cal = max(8, int(np.ceil(cal_ratio*len(Xs))))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10:
            continue
        # ローカル7D Ridge
        sc = StandardScaler()
        Xc = sc.fit_transform(Xs[idx_cal]); Xe = sc.transform(Xs[idx_rest])
        r = Ridge(alpha=alpha); r.fit(Xc, yc[idx_cal])
        local_pred = r.predict(Xe)
        # ゲート
        m_cal = np.percentile(ms[idx_cal], mcal_pct)
        me = ms[idx_rest]
        if tau <= 0:
            w = (me <= m_cal).astype(float)
        else:
            w = np.exp(-np.maximum(0.0, me - m_cal) / tau)
        w = w[:, None]
        pred = w * local_pred + (1 - w) * gp[idx_rest]
        euc = euclidean_cm(pred, yc[idx_rest])
        all_euc.append(euc)
        for lo, hi in POSE_BINS:
            bm = (me >= lo) & (me < hi)
            if bm.sum() >= 5:
                per_bin[(lo, hi)].append(np.median(euc[bm]))
    all_euc = np.concatenate(all_euc)
    rep = {b: (float(np.median(per_bin[b])) if per_bin[b] else None) for b in POSE_BINS}
    rep["all"] = float(np.median(all_euc))
    return rep


def print_row(label, rep):
    s = f"  {label:<20}"
    for b in POSE_BINS:
        v = rep[b]; s += f" {v:>6.2f}" if v is not None else "   -  "
    s += f"  | {rep['all']:.3f}"
    print(s)


def main():
    t0 = time.time()
    gm = joblib.load(MODEL_IN)
    d = np.load(str(CACHE_7D))
    Xt, yct, subj = d["X"], d["y_cm"], d["subj_id"]
    print(f"[Load] global v2 + test {len(Xt)} frames\n")

    print(f"{'='*90}")
    print(f"  姿勢ゲート・ハイブリッド (tau掃引, alpha=5)")
    print(f"{'='*90}")
    hdr = f"  {'config':<20}"
    for lo, hi in POSE_BINS: hdr += f" {lo:>2}-{hi:<3}"
    hdr += "  |  all"
    print(hdr); print("  " + "-"*86)

    # 参照: グローバルのみ
    gp = gm.predict(Xt)
    mag = np.sqrt(np.degrees(Xt[:,4])**2 + np.degrees(Xt[:,5])**2)
    euc_g = euclidean_cm(gp, yct)
    repg = {b: None for b in POSE_BINS}
    for lo, hi in POSE_BINS:
        bm = (mag>=lo)&(mag<hi)
        if bm.sum()>=20: repg[(lo,hi)] = float(np.median(euc_g[bm]))
    repg["all"] = float(np.median(euc_g))
    print_row("グローバルのみ", repg)

    best = (None, 1e9)
    for tau in (5, 6, 7, 8, 10):
        for alpha in (2.0, 5.0, 10.0):
            rep = eval_hybrid(Xt, yct, subj, gm, tau=tau, alpha=alpha)
            # 横向き重視スコア: 20deg以上のbin平均
            turn_bins = [rep[b] for b in [(20,25),(25,30),(30,40),(40,90)] if rep[b]]
            turn_score = np.mean(turn_bins)
            print_row(f"tau={tau} a={alpha:g}", rep)
            # overall+横向きの複合で最良を選ぶ
            score = rep["all"] + 0.3 * turn_score
            if score < best[1]:
                best = ((tau, alpha), score)
    print("  " + "-"*86)
    print(f"\n[{time.time()-t0:.0f}s] 最良config: tau={best[0][0]} alpha={best[0][1]}")


if __name__ == "__main__":
    main()
