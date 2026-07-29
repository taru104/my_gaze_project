"""exp54: アピアランスのPCA次元を振り、16D+appの改善が安定か/person-indepを押せるか確認。
既存cache(mpii_app.npz)を使う軽い実験。個人内=時間分割(honest)＋person-indep。GPU不要。mainは触らない。
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

d = np.load(str(ROOT / "cache" / "mpii_app.npz"))
X16, Xapp, y, pid = d["X16"], d["Xapp"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))
NPCS = [20, 40, 80, 120, 160]

def both(tr, te, npc):
    pca = PCA(n_components=min(npc, len(tr) - 1)).fit(Xapp[tr])
    return np.hstack([X16[tr], pca.transform(Xapp[tr])]), np.hstack([X16[te], pca.transform(Xapp[te])])

log(f"\n---\n## exp54: アピアランスPCA次元スイープ（16D+app, MPII {len(pids)}人）")
# 個人内 時間分割 baseline(16d)
in16 = np.median([np.median(err(fit_predict(X16[np.where(pid==p)[0][:len(np.where(pid==p)[0])//2]],
        y[np.where(pid==p)[0][:len(np.where(pid==p)[0])//2]],
        X16[np.where(pid==p)[0][len(np.where(pid==p)[0])//2:]]),
        y[np.where(pid==p)[0][len(np.where(pid==p)[0])//2:]])) for p in pids])
# person-indep baseline(16d)
lo16 = np.median([np.median(err(fit_predict(X16[pid!=p], y[pid!=p], X16[pid==p]), y[pid==p])) for p in pids])
log(f"\n  基準 16D単体: 個人内(時間分割)={in16:.3f}cm / person-indep={lo16:.3f}cm")
log(f"  {'PCA':>5} | {'個人内(時間分割)':>16} | {'person-indep':>13}")
for npc in NPCS:
    inn = []
    for p in pids:
        idx = np.where(pid == p)[0]; cut = len(idx)//2
        tr, te = idx[:cut], idx[cut:]
        Xtr, Xte = both(tr, te, npc)
        inn.append(np.median(err(fit_predict(Xtr, y[tr], Xte), y[te])))
    lop = []
    for p in pids:
        tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
        Xtr, Xte = both(tr, te, npc)
        lop.append(np.median(err(fit_predict(Xtr, y[tr], Xte), y[te])))
    log(f"  {npc:>5} | {np.median(inn):14.3f}cm | {np.median(lop):11.3f}cm")
log("\n**読み**: 個人内でbothが16d(基準)を安定して下回り、最適PCA次元を確認。person-indepが16dを下回れば汎化にも効く=大きい。")
