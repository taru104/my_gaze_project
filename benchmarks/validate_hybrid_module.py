"""
本番モジュール hybrid_calibration.HybridCalibration が
研究ハーネスの結果(overall~4.24cm)を再現するか検証。

Usage:
    .venv/Scripts/python.exe benchmarks/validate_hybrid_module.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from _eval_common import euclidean_cm, POSE_BINS
from hybrid_calibration import HybridCalibration


class Ensemble:  # unpickle用
    def __init__(self, models): self.models = models
    def predict(self, X): return np.mean([m.predict(X) for m in self.models], axis=0)


PROJECT_DIR = Path(__file__).parent.parent
CACHE_7D    = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
MODEL_IN    = PROJECT_DIR / "cache" / "global_mlp_v2.joblib"


def main():
    gm = joblib.load(MODEL_IN)
    d = np.load(str(CACHE_7D))
    Xt, yct, subj = d["X"], d["y_cm"], d["subj_id"]
    mag = np.sqrt(np.degrees(Xt[:,4])**2 + np.degrees(Xt[:,5])**2)

    per_bin = {b: [] for b in POSE_BINS}
    all_euc = []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = Xt[m], yct[m], mag[m]
        order = np.argsort(ms)
        n_cal = max(8, int(np.ceil(0.10*len(Xs))))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10:
            continue
        hyb = HybridCalibration(global_model=gm, tau=6.0, alpha=10.0)
        for i in idx_cal:
            hyb.add(Xs[i], yc[i])
        hyb.fit()
        pred = hyb.predict(Xs[idx_rest])
        euc = euclidean_cm(pred, yc[idx_rest])
        all_euc.append(euc)
        me = ms[idx_rest]
        for lo, hi in POSE_BINS:
            bm = (me>=lo)&(me<hi)
            if bm.sum()>=5:
                per_bin[(lo,hi)].append(np.median(euc[bm]))
    all_euc = np.concatenate(all_euc)

    print("本番モジュール HybridCalibration 検証 (tau=6, alpha=10):")
    s = "  bin別 median:"
    for b in POSE_BINS:
        v = np.median(per_bin[b]) if per_bin[b] else None
        s += f" {v:.2f}" if v is not None else " -"
    print(s)
    print(f"  overall median = {np.median(all_euc):.3f} cm  (研究ハーネス: 4.24)")
    ok = abs(np.median(all_euc) - 4.24) < 0.15
    print(f"  {'[OK] 再現成功' if ok else '[NG] 不一致'}")


if __name__ == "__main__":
    main()
