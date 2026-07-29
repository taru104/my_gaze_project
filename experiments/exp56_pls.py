"""exp56: PLS回帰(教師あり次元圧縮)。PCAは視線と無関係な高分散成分も残す=過学習源。
PLSは「視線と相関する潜在成分だけ」を抽出=過学習に構造的に強い。16D+生パッチ768DをPLSで低次元に。
過学習警戒: person-indep主指標。n_components小さめ。mainは触らない。
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA

CM_W = 30.0
REPORT = ROOT / "experiments" / "REPORT6_beyond16d.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def err(pred, gt): return np.sqrt(np.sum((pred - gt) ** 2, axis=1)) * CM_W

d = np.load(str(ROOT / "cache" / "mpii_app.npz"))
X16, Xapp, y, pid = d["X16"], d["Xapp"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))
Xfull = np.hstack([X16, Xapp])   # 16 + 768 = 784D

def huber_pca(Xtr, Ytr, Xte, npc):
    pca = PCA(n_components=min(npc, len(tr_g)-1)).fit(Xapp[tr_g])  # placeholder, replaced below
    raise RuntimeError

def pred_pls(Xtr, Ytr, Xte, nc):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    m = PLSRegression(n_components=nc, scale=False).fit(A, Ytr)
    return m.predict(B)

def pred_huber_pca(Xtr16, Xtrapp, Ytr, Xte16, Xteapp, npc):
    pca = PCA(n_components=min(npc, len(Xtr16)-1)).fit(Xtrapp)
    Xtr = np.hstack([Xtr16, pca.transform(Xtrapp)]); Xte = np.hstack([Xte16, pca.transform(Xteapp)])
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(A, Ytr[:, i]).predict(B)
    return pr

NCS = [8, 16, 24, 32]
log(f"\n---\n## exp56: PLS(教師あり圧縮) vs PCA（MPII {len(pids)}人）過学習警戒=person-indep主指標")

# person-indep
log(f"\n### (主) person-independent（基準16D=6.17 / 16D+PCA20(exp54)≈5.68）")
log(f"  {'手法':>16} | " + " | ".join(f"nc={n}".rjust(9) for n in NCS))
row_pls = []
for nc in NCS:
    e = []
    for p in pids:
        tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
        e.append(np.median(err(pred_pls(Xfull[tr], y[tr], Xfull[te], nc), y[te])))
    row_pls.append(np.median(e))
log(f"  {'PLS(16D+patch)':>16} | " + " | ".join(f"{v:8.3f}c" for v in row_pls))
# 参考: Huber+PCA(app) 同じncで
row_hp = []
for nc in NCS:
    e = []
    for p in pids:
        tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
        e.append(np.median(err(pred_huber_pca(X16[tr], Xapp[tr], y[tr], X16[te], Xapp[te], nc), y[te])))
    row_hp.append(np.median(e))
log(f"  {'Huber+PCA(app)':>16} | " + " | ".join(f"{v:8.3f}c" for v in row_hp))

# 個人内 時間分割
log(f"\n### (副) 個人内 時間分割（基準16D=5.20）")
row_pls_in, row_hp_in = [], []
for nc in NCS:
    ep, eh = [], []
    for p in pids:
        idx = np.where(pid == p)[0]; cut = len(idx)//2; tr, te = idx[:cut], idx[cut:]
        ep.append(np.median(err(pred_pls(Xfull[tr], y[tr], Xfull[te], nc), y[te])))
        eh.append(np.median(err(pred_huber_pca(X16[tr], Xapp[tr], y[tr], X16[te], Xapp[te], nc), y[te])))
    row_pls_in.append(np.median(ep)); row_hp_in.append(np.median(eh))
log(f"  {'PLS(16D+patch)':>16} | " + " | ".join(f"{v:8.3f}c" for v in row_pls_in))
log(f"  {'Huber+PCA(app)':>16} | " + " | ".join(f"{v:8.3f}c" for v in row_hp_in))

bp = min(row_pls); bh = min(row_hp)
log(f"\n**判定(過学習警戒)**: person-indep PLS最良={bp:.3f} / Huber+PCA最良={bh:.3f} vs 16D=6.17。"
    f"PLSが低ncで16Dを安定して下回れば『教師あり圧縮で過学習を抑えつつ画像情報を汎化に使える』=大きい前進。")
