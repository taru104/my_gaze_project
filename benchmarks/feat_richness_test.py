"""
豊富な特徴(486D眼ランドマーク)が横向きで効くかの検証。
486DキャッシュはTest26人分のみ(train split無し)なので、
被験者内 frontal→turned でローカルキャリブして比較する。
これで「特徴を増やせば横向き精度が上がるか」を判定 → 再抽出投資の是非を決める。

7D vs 486D を同一プロトコル(正面キャリブ→bin別評価)で比較。
486Dは高次元なので強Ridge収縮必須。

Usage:
    .venv/Scripts/python.exe benchmarks/feat_richness_test.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import euclidean_cm, POSE_BINS

PROJECT_DIR = Path(__file__).parent.parent
CACHE_7D    = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
CACHE_486   = PROJECT_DIR / "cache" / "sota_486d_cache.npz"


def eval_local(X, y_cm, subj, pose_src, alpha, cal_ratio=0.10):
    """
    被験者ごと: pose小さい順に cal_ratio 割でキャリブ(StandardScaler+Ridge),
    残りを pose bin 別に誤差集計。pose_src は姿勢magnitude計算用の配列。
    """
    mag = pose_src
    per_bin = {b: [] for b in POSE_BINS}
    all_euc = []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms = X[m], y_cm[m], mag[m]
        order = np.argsort(ms)
        n_cal = max(8, int(np.ceil(cal_ratio * len(Xs))))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10:
            continue
        sc = StandardScaler()
        Xc = sc.fit_transform(Xs[idx_cal]); Xe = sc.transform(Xs[idx_rest])
        r = Ridge(alpha=alpha); r.fit(Xc, yc[idx_cal])
        pred = r.predict(Xe)
        euc = euclidean_cm(pred, yc[idx_rest])
        rm = ms[idx_rest]
        all_euc.append(euc)
        for lo, hi in POSE_BINS:
            bm = (rm >= lo) & (rm < hi)
            if bm.sum() >= 5:
                per_bin[(lo, hi)].append(np.median(euc[bm]))
    all_euc = np.concatenate(all_euc)
    rep = {}
    for b in POSE_BINS:
        rep[b] = float(np.median(per_bin[b])) if per_bin[b] else None
    rep["all"] = float(np.median(all_euc))
    return rep


def main():
    t0 = time.time()
    d7 = np.load(str(CACHE_7D))
    X7, yc7, sid7 = d7["X"], d7["y_cm"], d7["subj_id"]
    mag7 = np.sqrt(np.degrees(X7[:,4])**2 + np.degrees(X7[:,5])**2)

    d4 = np.load(str(CACHE_486))
    X4, yc4, sid4 = d4["X"], d4["y_cm"], d4["subj_id"]
    # 486D特徴には yaw,pitch,roll が末尾3列 (486=161*3+3)。姿勢はそこから。
    yaw4 = np.degrees(X4[:, -3]); pit4 = np.degrees(X4[:, -2])
    mag4 = np.sqrt(pit4**2 + yaw4**2)

    print(f"[Load] 7D:{X7.shape} 486D:{X4.shape}")
    print(f"  486D姿勢列チェック: yaw range[{yaw4.min():.0f},{yaw4.max():.0f}] "
          f"pitch[{pit4.min():.0f},{pit4.max():.0f}]")

    configs = [
        ("7D  Ridge a=1",   X7, yc7, sid7, mag7, 1.0),
        ("7D  Ridge a=5",   X7, yc7, sid7, mag7, 5.0),
        ("486D Ridge a=50", X4, yc4, sid4, mag4, 50.0),
        ("486D Ridge a=200",X4, yc4, sid4, mag4, 200.0),
        ("486D Ridge a=500",X4, yc4, sid4, mag4, 500.0),
    ]

    print(f"\n{'='*88}")
    print(f"  特徴豊富度テスト: 正面10%キャリブ→bin別 median cm (ローカルキャリブ)")
    print(f"{'='*88}")
    hdr = f"  {'model':<16}"
    for lo, hi in POSE_BINS: hdr += f" {lo:>2}-{hi:<3}"
    hdr += "  |  all"
    print(hdr); print("  " + "-"*84)
    for label, X, yc, sid, mag, alpha in configs:
        rep = eval_local(X, yc, sid, mag, alpha)
        s = f"  {label:<16}"
        for b in POSE_BINS:
            v = rep[b]; s += f" {v:>6.2f}" if v is not None else "   -  "
        s += f"  | {rep['all']:.3f}"
        print(s)
    print("  " + "-"*84)
    print(f"\n[{time.time()-t0:.0f}s] 486Dが横向きbinで7Dより低ければ再抽出の価値あり")


if __name__ == "__main__":
    main()
