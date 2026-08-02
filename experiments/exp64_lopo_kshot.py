"""exp64: LOPO k-shot — 文献と直接比較できるプロトコルでの評価。

比較対象: EMC-Gaze (arXiv 2603.12388, 2026)
  "Deployment-Oriented Session-wise Meta-Calibration for Landmark-Based Webcam Gaze Tracking"
  MediaPipe 161ランドマーク + E(3)同変GNN + セッション毎 ridge 較正。
  **MPIIFaceGaze LOPO 16-shot = 8.82° ± 1.21** / Elastic Net ベースライン = 10.83°。
  Tgaze と同じ「ランドマークのみ・GPU無し・少数較正」土俵。ここに 16D+パッチ で挑む。

プロトコル(EMC-Gaze に合わせる):
  - Leave-One-Person-Out: 14人で学習 → 残り1人に k サンプルだけで適応 → その人の残りで評価。
  - k ∈ {0,1,3,5,9,16,45,100}。k=16 が EMC-Gaze の主要比較点。
  - 適応は「グローバル予測をアフィン補正」= EMC-Gaze の "shared encoder + per-session ridge" と同じ構造。
    k<3 は自由度が足りないのでバイアスのみ補正。k>=3 で 2x2+2 のアフィン。
  - seed を 5 本振って中央値(k サンプルの引きの良し悪しを均す)。

指標は exp63 の正しい cm と 度 の両方。

Usage: .venv/Scripts/python.exe experiments/exp64_lopo_kshot.py
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import cv2, scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.decomposition import PCA

REPORT = ROOT / "experiments" / "REPORT8_metrics.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

CACHE = ROOT / "cache" / "mpii_full.npz"
if not CACHE.exists(): CACHE = ROOT / "cache" / "mpii_full_ck.npz"
d = np.load(CACHE)
X16, patch, y, fc, gt, scr, pid = (d["X16"], d["patch"], d["y_norm"],
                                   d["fc"], d["gt"], d["scr"], d["pid"])
pids = sorted(set(pid.tolist()))

def monitor_pose(p):
    m = sio.loadmat(str(ROOT / "MPIIFaceGaze" / p / "Calibration" / "monitorPose.mat"))
    R = cv2.Rodrigues(np.ravel(m["rvects"] if "rvects" in m else m["rvecs"]).astype(np.float64))[0]
    return R, np.ravel(m["tvecs"]).astype(np.float64)
MP = {p: monitor_pose(p) for p in pids}

def err_cm(pred, idx):
    return np.linalg.norm((pred - y[idx]) * scr[idx][:, :2], axis=1) / 10.0

def err_deg(pred, idx):
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        R, T = MP[pid[i]]
        p3 = R @ np.array([pred[k][0] * scr[i][0], pred[k][1] * scr[i][1], 0.0]) + T
        a, b = p3 - fc[i], gt[i] - fc[i]
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        out[k] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out

def zpatch(rows):
    """uint8 パッチ → 眼ごと z-score(appearance.py と同一)。チャンクで float 化しメモリを抑える。"""
    P = patch[rows].astype(np.float32)
    a, b = P[:, :1536], P[:, 1536:]
    f = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return np.hstack([f(a), f(b)])

def build(tr, npca):
    """学習側の設計行列と、テスト行を変換する関数を返す。"""
    if npca <= 0:
        A0 = X16[tr]; tf = lambda te: X16[te]
    else:
        sub = tr if len(tr) <= 12000 else np.random.RandomState(0).choice(tr, 12000, replace=False)
        pca = PCA(n_components=npca, svd_solver="randomized", random_state=0).fit(zpatch(sub))
        A0 = np.hstack([X16[tr], pca.transform(zpatch(tr))])
        tf = lambda te: np.hstack([X16[te], pca.transform(zpatch(te))])
    sc = StandardScaler().fit(A0)
    return sc.transform(A0), (lambda te: sc.transform(tf(te)))

def global_model(tr, npca):
    A, tf = build(tr, npca)
    hs = [HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500).fit(A, y[tr][:, i]) for i in range(2)]
    return lambda te: np.column_stack([h.predict(tf(te)) for h in hs])

def adapt(base_cal, y_cal, base_te, k):
    """k サンプルでグローバル予測を較正。k<3 はバイアスのみ、k>=3 はアフィン(2x2+2)。"""
    if k == 0: return base_te
    if k < 3:  return base_te + (y_cal - base_cal).mean(0)
    D = np.hstack([base_cal, np.ones((len(base_cal), 1))])
    W = Ridge(alpha=1e-3, fit_intercept=False).fit(D, y_cal).coef_
    return np.hstack([base_te, np.ones((len(base_te), 1))]) @ W.T

KS = [0, 1, 3, 5, 9, 16, 45, 100]
SEEDS = 5
log(f"\n## 2. LOPO k-shot（EMC-Gaze 2026 と同一プロトコル）  n={len(X16)}, {len(pids)}人\n")

for npca, name in [(0, "16D幾何のみ"), (16, "16D+目パッチPCA16 (既定)")]:
    res_cm = {k: [] for k in KS}; res_dg = {k: [] for k in KS}
    for p in pids:
        te_all = np.where(pid == p)[0]; tr = np.where(pid != p)[0]
        gm = global_model(tr, npca)
        base_all = gm(te_all)                       # グローバル予測(1回だけ計算し使い回す)
        for k in KS:
            cms, dgs = [], []
            for s in range(SEEDS if k > 0 else 1):
                rs = np.random.RandomState(1000 * s + 7)
                cal = rs.choice(len(te_all), k, replace=False) if k else np.array([], int)
                mask = np.ones(len(te_all), bool); mask[cal] = False
                te = te_all[mask]
                pr = adapt(base_all[cal], y[te_all[cal]], base_all[mask], k)
                cms.append(np.median(err_cm(pr, te))); dgs.append(np.median(err_deg(pr, te)))
            res_cm[k].append(np.median(cms)); res_dg[k].append(np.median(dgs))
        print(f"  [{name}] {p} 完了", flush=True)
    log(f"### {name}")
    log("| k (較正サンプル数) | " + " | ".join(str(k) for k in KS) + " |")
    log("|---|" + "---:|" * len(KS))
    log("| 誤差 cm | " + " | ".join(f"{np.median(res_cm[k]):.2f}" for k in KS) + " |")
    log("| **誤差 度** | " + " | ".join(f"**{np.median(res_dg[k]):.2f}°**" for k in KS) + " |")
    log("")

log("### 比較（同じ MPIIFaceGaze LOPO・ランドマークのみ・少数較正）")
log("| 手法 | 16-shot LOPO |")
log("|---|---:|")
log("| Ridge on raw landmarks (EMC-Gaze論文の報告) | 27.28°※ |")
log("| Elastic Net on raw landmarks (同上) | 10.83° |")
log("| EMC-Gaze (E(3)同変GNN + メタ較正, 2026) | 8.82° |")
log("| **Tgaze (16D + 目パッチ, 線形Huber, CPU)** | 上表 k=16 を参照 |")
log("※ Ridge 27.28° は論文のインタラクティブ評価値。MPII LOPO の該当値は論文に無いため参考。")
