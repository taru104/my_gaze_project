"""
E1: ベイズ的キャリブレーション (NotebookLM Q5)。
グローバル予測を事前分布とし、少数キャリブで事後更新。未観測姿勢では
自動的にグローバルへフォールバック → 姿勢ゲート(tau手動調整)の理論的上位版。

比較(正面キャリブ→bin別 median cm):
  - 姿勢ゲート hybrid (現行ベスト, tau=6)
  - Bayesian残差 : global予測を基準に、残差をBayesianRidgeで学習(自動正則化)
  - Bayesian直接 : globalをprior meanとしBayesianRidgeでyを学習(global予測を特徴に追加)

Usage:
    .venv/Scripts/python.exe benchmarks/bayesian_calib_test.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler

class Ensemble:
    def __init__(self, models): self.models = models
    def predict(self, X): return np.mean([m.predict(X) for m in self.models], axis=0)

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import euclidean_cm, POSE_BINS

ROOT = Path(__file__).parent.parent
CACHE_7D = ROOT / "cache" / "sota_7d_cache.npz"
MODEL = ROOT / "cache" / "global_mlp_v2.joblib"


def pose_mag(X):
    return np.sqrt(np.degrees(X[:,4])**2 + np.degrees(X[:,5])**2)


def run_eval(Xt, yct, subj, gm, method, cal_ratio=0.10):
    mag = pose_mag(Xt)
    gp_full = gm.predict(Xt)
    per_bin = {b: [] for b in POSE_BINS}; all_euc = []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms, gp = Xt[m], yct[m], mag[m], gp_full[m]
        order = np.argsort(ms); n_cal = max(8, int(np.ceil(cal_ratio*len(Xs))))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10: continue
        sc = StandardScaler(); Xc = sc.fit_transform(Xs[idx_cal]); Xe = sc.transform(Xs[idx_rest])
        ycal, gcal = yc[idx_cal], gp[idx_cal]
        grest = gp[idx_rest]

        if method == "gate":
            r = Ridge(alpha=10.0).fit(Xc, ycal); local = r.predict(Xe)
            m_cal = np.percentile(ms[idx_cal], 90); me = ms[idx_rest]
            w = np.exp(-np.maximum(0.0, me-m_cal)/6.0)[:, None]
            pred = w*local + (1-w)*grest
        elif method == "bayes_resid":
            # 残差をBayesianRidge(自動正則化)。x/y独立。
            resid = ycal - gcal
            pred = grest.copy()
            for k in range(2):
                br = BayesianRidge().fit(Xc, resid[:, k])
                pred[:, k] = grest[:, k] + br.predict(Xe)
        elif method == "bayes_direct":
            # global予測を特徴に追加してBayesianRidgeでy直接。
            Xc2 = np.column_stack([Xc, gcal]); Xe2 = np.column_stack([Xe, grest])
            pred = np.zeros_like(grest)
            for k in range(2):
                br = BayesianRidge().fit(Xc2, ycal[:, k])
                pred[:, k] = br.predict(Xe2)
        elif method == "gate_bayes":
            # ベイズ直接(正面で最良)を姿勢ゲートで包む → 正面精度+横向き頑健性
            Xc2 = np.column_stack([Xc, gcal]); Xe2 = np.column_stack([Xe, grest])
            local = np.zeros_like(grest)
            for k in range(2):
                br = BayesianRidge().fit(Xc2, ycal[:, k])
                local[:, k] = br.predict(Xe2)
            m_cal = np.percentile(ms[idx_cal], 90); me = ms[idx_rest]
            w = np.exp(-np.maximum(0.0, me-m_cal)/6.0)[:, None]
            pred = w*local + (1-w)*grest
        else:  # global only
            pred = grest

        euc = euclidean_cm(pred, yc[idx_rest]); all_euc.append(euc)
        me = ms[idx_rest]
        for lo, hi in POSE_BINS:
            bm = (me>=lo)&(me<hi)
            if bm.sum()>=5: per_bin[(lo,hi)].append(np.median(euc[bm]))
    rep = {b:(float(np.median(per_bin[b])) if per_bin[b] else None) for b in POSE_BINS}
    rep["all"] = float(np.median(np.concatenate(all_euc)))
    return rep


def prow(label, rep):
    s=f"  {label:<18}"
    for b in POSE_BINS:
        v=rep[b]; s+=f" {v:>6.2f}" if v is not None else "   -  "
    print(s+f"  | {rep['all']:.3f}")


def main():
    t0=time.time()
    gm = joblib.load(MODEL)
    d = np.load(str(CACHE_7D)); Xt, yct, subj = d["X"], d["y_cm"], d["subj_id"]
    print(f"[Load] test {len(Xt)} frames\n")
    hdr=f"  {'method':<18}"
    for lo,hi in POSE_BINS: hdr+=f" {lo:>2}-{hi:<3}"
    print("="*88); print(hdr+"  |  all"); print("  "+"-"*84)
    prow("global only",   run_eval(Xt,yct,subj,gm,"global"))
    prow("姿勢ゲート(現行)", run_eval(Xt,yct,subj,gm,"gate"))
    prow("Bayes残差",      run_eval(Xt,yct,subj,gm,"bayes_resid"))
    prow("Bayes直接",      run_eval(Xt,yct,subj,gm,"bayes_direct"))
    prow("★ゲート+Bayes",   run_eval(Xt,yct,subj,gm,"gate_bayes"))
    print("  "+"-"*84)
    print(f"\n[{time.time()-t0:.0f}s] 横向きbinで姿勢ゲートを超えれば採用")


if __name__ == "__main__":
    main()
