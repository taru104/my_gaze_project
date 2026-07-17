"""共通評価ユーティリティ (bin別誤差)。各実験から import して使う。"""
import numpy as np

def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))

def feat_2d(X):
    return np.column_stack([(X[:, 0] + X[:, 2]) / 2, (X[:, 1] + X[:, 3]) / 2])

POSE_BINS = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 90)]

def pose_mag(X):
    return np.sqrt(np.degrees(X[:, 4])**2 + np.degrees(X[:, 5])**2)

def bin_report(euc, mag, label="model"):
    """bin別 median を dict で返す + 全体"""
    out = {}
    for lo, hi in POSE_BINS:
        bm = (mag >= lo) & (mag < hi)
        if bm.sum() >= 20:
            out[(lo, hi)] = float(np.median(euc[bm]))
    out["all"] = float(np.median(euc))
    out["mean"] = float(np.mean(euc))
    return out

def print_bins(name, rep):
    s = f"  {name:<24}"
    for lo, hi in POSE_BINS:
        v = rep.get((lo, hi))
        s += f" {v:>6.2f}" if v is not None else "   -  "
    s += f"  | all={rep['all']:.3f} mean={rep['mean']:.3f}"
    print(s)

def print_header():
    hdr = "  " + f"{'model':<24}"
    for lo, hi in POSE_BINS:
        hdr += f" {lo:>2}-{hi:<3}"
    hdr += "  | overall"
    print(hdr)
    print("  " + "-" * 78)
