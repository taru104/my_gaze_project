"""exp63: 指標の是正 — 正しい cm と、文献比較可能な「度(angular error)」を導入。

これまでの MPII 実験(exp52-62)は誤差を `sqrt(dx_n^2+dy_n^2) * 30.0` で cm 化していた。
これは正規化誤差の**両軸に同じ30cmを掛ける**ため、実画面が 286.5 x 179.0 mm であることを無視し
**縦誤差を 30/17.9 = 1.68倍に水増し**していた。本実験でまず是正する(数値は下がる方向=これまで過小評価していた)。

さらに MPIIFaceGaze の 3D注釈(顔中心 fc / 注視点 gt)と monitorPose を使い、
予測した画面点を **カメラ座標の3D点に逆投影 → 視線ベクトル → 正解ベクトルとの角度** を計算する。
これで初めて Tgaze の精度が「度」で表現でき、gaze推定の文献(SOTA 3-6°)と直接比較できる。
※ px→3D の逆投影は extract_mpii_full.py で正解 gt と 0.0mm 一致を確認済み(変換は厳密)。

Usage: .venv/Scripts/python.exe experiments/exp63_metrics.py
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
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA

REPORT = ROOT / "experiments" / "REPORT8_metrics.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

CACHE = ROOT / "cache" / "mpii_full.npz"
if not CACHE.exists():
    CACHE = ROOT / "cache" / "mpii_full_ck.npz"
    print(f"[warn] 完成キャッシュが無いのでチェックポイントを使用: {CACHE}")

d = np.load(CACHE)
X16, patch, y, y_px = d["X16"], d["patch"], d["y_norm"], d["y_px"]
fc, gt, scr, pid = d["fc"], d["gt"], d["scr"], d["pid"]
pids = sorted(set(pid.tolist()))


def monitor_pose(p):
    m = sio.loadmat(str(ROOT / "MPIIFaceGaze" / p / "Calibration" / "monitorPose.mat"))
    R = cv2.Rodrigues(np.ravel(m["rvects"] if "rvects" in m else m["rvecs"]).astype(np.float64))[0]
    return R, np.ravel(m["tvecs"]).astype(np.float64)

MP = {p: monitor_pose(p) for p in pids}


def err_cm(pred, idx):
    """正しい cm 誤差: 正規化誤差に per-person の実画面 mm を軸ごとに掛ける。"""
    dmm = (pred - y[idx]) * scr[idx][:, :2]      # [width_mm, height_mm]
    return np.linalg.norm(dmm, axis=1) / 10.0


def err_deg(pred, idx):
    """角度誤差(度): 予測画面点をカメラ座標3Dへ逆投影し、顔中心からの視線ベクトル同士の角度。"""
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        R, T = MP[pid[i]]
        wmm, hmm = scr[i][0], scr[i][1]
        v = np.array([pred[k][0] * wmm, pred[k][1] * hmm, 0.0])
        p3 = R @ v + T
        a, b = p3 - fc[i], gt[i] - fc[i]
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        out[k] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def zpatch(rows):
    """uint8 パッチ → 眼ごと z-score(appearance.py と同一)。必要行だけ float 化してメモリを抑える。"""
    P = patch[rows].astype(np.float32)
    f = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return np.hstack([f(P[:, :1536]), f(P[:, 1536:])])


def fit_predict(tr, te, npca=16):
    """確定チャンピオン: 16D幾何 + 目パッチPCA + Huber。npca=0 で16D単独。"""
    if npca <= 0:
        A0, B0 = X16[tr], X16[te]
    else:
        sub = tr if len(tr) <= 12000 else np.random.RandomState(0).choice(tr, 12000, replace=False)
        pca = PCA(n_components=min(npca, len(sub) - 1), svd_solver="randomized",
                  random_state=0).fit(zpatch(sub))
        A0 = np.hstack([X16[tr], pca.transform(zpatch(tr))])
        B0 = np.hstack([X16[te], pca.transform(zpatch(te))])
    sc = StandardScaler().fit(A0); A, B = sc.transform(A0), sc.transform(B0)
    pr = np.zeros((len(te), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(A, y[tr][:, i]).predict(B)
    return pr


log(f"\n# REPORT8 — 指標の是正と角度(度)評価\n")
log(f"データ: {CACHE.name}  n={len(X16)}  被験者={len(pids)}人 {pids}")
log(f"画面: 幅{np.unique(scr[:,0])} mm / 高さ{np.unique(scr[:,1])} mm  "
    f"→ 旧 `*30.0` は縦を最大 {30/17.90:.2f} 倍に水増ししていた\n")

# ---- 1) 旧指標 vs 新指標 (person-specific 前半/後半 時間分割) ----
log("## 1. 旧cm(誤) vs 新cm(正) vs 度  — person-specific 時間分割(前半学習/後半テスト)")
log("| 被験者 | n | 旧cm(*30) | **新cm(実画面)** | **度** |")
log("|---|---:|---:|---:|---:|")
rows = []
for p in pids:
    idx = np.where(pid == p)[0]
    cut = len(idx) // 2
    tr, te = idx[:cut], idx[cut:]
    pr = fit_predict(tr, te, npca=16)
    old = np.median(np.linalg.norm(pr - y[te], axis=1) * 30.0)
    new = np.median(err_cm(pr, te)); deg = np.median(err_deg(pr, te))
    rows.append((old, new, deg))
    log(f"| {p} | {len(idx)} | {old:.2f} | **{new:.2f}** | **{deg:.2f}°** |")
r = np.array(rows)
log(f"| **中央値** | | {np.median(r[:,0]):.2f} | **{np.median(r[:,1]):.2f}** | "
    f"**{np.median(r[:,2]):.2f}°** |")
log(f"\n→ 旧指標は新指標を {np.median(r[:,0])/np.median(r[:,1]):.2f} 倍に水増ししていた"
    f"(縦を1.68倍していたため)。**これまでの MPII の数字は実際にはもっと良かった**。\n")
