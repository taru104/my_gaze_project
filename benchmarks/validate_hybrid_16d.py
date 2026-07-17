"""
本番モジュール hybrid_calibration.HybridCalibration が 16D グローバルモデルで
研究結果(rich_hybrid_eval: overall~3.339cm)をエンドツーエンドで再現するか検証。

7D版(validate_hybrid_module.py)の16D対応。global_mlp_16d.joblib + rich_test_cache.npz を使う。

Usage:
    .venv/Scripts/python.exe benchmarks/validate_hybrid_16d.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from _eval_common import euclidean_cm, POSE_BINS
from hybrid_calibration import HybridCalibration

PROJECT_DIR = Path(__file__).parent.parent
TEST_RICH   = PROJECT_DIR / "cache" / "rich_test_cache.npz"
MODEL_IN    = PROJECT_DIR / "cache" / "global_mlp_16d.joblib"
TARGET      = 3.339  # rich_hybrid_eval の 16D rich hybrid overall


def main():
    if not MODEL_IN.exists() or not TEST_RICH.exists():
        print(f"[待機] 必要ファイル未生成: model={MODEL_IN.exists()} test={TEST_RICH.exists()}")
        return
    gm = joblib.load(MODEL_IN)
    d = np.load(str(TEST_RICH))
    Xt, yct, subj = d["X"], d["y_cm"], d["subj_id"]
    assert Xt.shape[1] == 16, f"次元不一致: {Xt.shape[1]}"
    mag = np.sqrt(np.degrees(Xt[:, 4])**2 + np.degrees(Xt[:, 5])**2)

    per_bin = {b: [] for b in POSE_BINS}
    all_euc = []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = Xt[m], yct[m], mag[m]
        order = np.argsort(ms)
        n_cal = max(8, int(np.ceil(0.10 * len(Xs))))
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
            bm = (me >= lo) & (me < hi)
            if bm.sum() >= 5:
                per_bin[(lo, hi)].append(np.median(euc[bm]))
    all_euc = np.concatenate(all_euc)

    print("本番モジュール HybridCalibration 16D検証 (tau=6, alpha=10):")
    s = "  bin別 median:"
    for b in POSE_BINS:
        v = np.median(per_bin[b]) if per_bin[b] else None
        s += f" {v:.2f}" if v is not None else " -"
    print(s)
    overall = float(np.median(all_euc))
    print(f"  overall median = {overall:.3f} cm  (研究ハーネス rich_hybrid_eval: {TARGET})")
    ok = abs(overall - TARGET) < 0.20
    print(f"  {'[OK] 再現成功 → 本番モジュール16D対応 検証済' if ok else '[NG] 不一致(要調査)'}")


if __name__ == "__main__":
    main()
