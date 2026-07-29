"""exp55: HOG特徴(勾配ヒストグラム)。生ピクセルより低次元(72D)で照明/個人差にロバスト=過学習に強い。
既存パッチ(mpii_app.npz)から計算(再抽出不要)。主指標=person-indep(他人=過学習を最も暴く)、副=個人内時間分割。
過学習警戒: 低次元・strict split・person-indep重視。mainは触らない。
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CM_W = 30.0
REPORT = ROOT / "experiments" / "REPORT6_beyond16d.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def fit_predict(Xtr, Ytr, Xte, alpha=1e-3):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=800).fit(A, Ytr[:, i]).predict(B)
    return pr
def err(pred, gt): return np.sqrt(np.sum((pred - gt) ** 2, axis=1)) * CM_W

PH, PW = 16, 24  # パッチ形状(1眼)
def hog_patch(p, cs=8, nb=6):
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    gy[1:-1, :] = p[2:, :] - p[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    feats = []
    for r in range(0, PH - cs + 1, cs):
        for c in range(0, PW - cs + 1, cs):
            m = mag[r:r+cs, c:c+cs].ravel(); a = ang[r:r+cs, c:c+cs].ravel()
            b = np.minimum((a / (180.0 / nb)).astype(int), nb - 1)
            hist = np.bincount(b, weights=m, minlength=nb).astype(float)
            hist /= (np.linalg.norm(hist) + 1e-6)
            feats.append(hist)
    return np.concatenate(feats)   # 2x3セル×6 = 36/眼

d = np.load(str(ROOT / "cache" / "mpii_app.npz"))
X16, Xapp, y, pid = d["X16"], d["Xapp"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))
# パッチ→HOG (2眼分連結=72D)
XH = np.zeros((len(Xapp), 72), np.float32)
for i in range(len(Xapp)):
    L = Xapp[i, :384].reshape(PH, PW); R = Xapp[i, 384:].reshape(PH, PW)
    XH[i] = np.concatenate([hog_patch(L), hog_patch(R)])
log(f"\n---\n## exp55: HOG特徴（72D, MPII {len(pids)}人）過学習警戒=person-indep主指標")

def build(tr, te, mode):
    if mode == "16d": return X16[tr], X16[te]
    if mode == "16d+rawPCA20":
        pca = PCA(n_components=min(20, len(tr)-1)).fit(Xapp[tr])
        return np.hstack([X16[tr], pca.transform(Xapp[tr])]), np.hstack([X16[te], pca.transform(Xapp[te])])
    if mode == "16d+HOG":
        return np.hstack([X16[tr], XH[tr]]), np.hstack([X16[te], XH[te]])
    if mode == "16d+HOG_PCA20":
        pca = PCA(n_components=min(20, len(tr)-1)).fit(XH[tr])
        return np.hstack([X16[tr], pca.transform(XH[tr])]), np.hstack([X16[te], pca.transform(XH[te])])
MODES = ["16d", "16d+rawPCA20", "16d+HOG", "16d+HOG_PCA20"]

log(f"\n### (主) person-independent（14人学習→未知1人=過学習を最も暴く, 基準16D=6.17）")
lo = {m: [] for m in MODES}
for p in pids:
    tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
    for m in MODES:
        Xtr, Xte = build(tr, te, m); lo[m].append(np.median(err(fit_predict(Xtr, y[tr], Xte), y[te])))
for m in MODES: log(f"  {m:>15} | {np.median(lo[m]):.3f}cm")

log(f"\n### (副) 個人内 時間分割（前半→後半, 基準16D=5.20）")
inn = {m: [] for m in MODES}
for p in pids:
    idx = np.where(pid == p)[0]; cut = len(idx)//2
    tr, te = idx[:cut], idx[cut:]
    for m in MODES:
        Xtr, Xte = build(tr, te, m); inn[m].append(np.median(err(fit_predict(Xtr, y[tr], Xte), y[te])))
for m in MODES: log(f"  {m:>15} | {np.median(inn[m]):.3f}cm")

blo = min(np.median(lo[m]) for m in MODES); bmode = min(MODES, key=lambda m: np.median(lo[m]))
log(f"\n**判定(過学習警戒)**: person-indep最良={blo:.3f}cm ({bmode})。16D(6.17)を下回れば汎化に効く=過学習でない本物。"
    f"HOGが生パッチPCAより person-indep で良ければ『低次元ロバスト特徴の勝ち』。")
