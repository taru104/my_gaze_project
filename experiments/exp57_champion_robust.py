"""exp57: チャンピオン(16D+生パッチPCA20)の頑健性診断。過学習警戒の総点検。
 (1) X軸/Y軸別: アピアランスが横/縦どちらの視線に効くか(正規化誤差)
 (2) 正則化α・PCA次元を変えても person-indep 改善が安定か(過学習でないかの念押し)
既存cache使用=高速。person-indep主指標。mainは触らない。
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA

REPORT = ROOT / "experiments" / "REPORT6_beyond16d.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

d = np.load(str(ROOT / "cache" / "mpii_app.npz"))
X16, Xapp, y, pid = d["X16"], d["Xapp"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))

def predict(Xtr16, Xtrapp, Ytr, Xte16, Xteapp, npc, alpha, use_app):
    if use_app:
        pca = PCA(n_components=min(npc, len(Xtr16)-1)).fit(Xtrapp)
        Xtr = np.hstack([Xtr16, pca.transform(Xtrapp)]); Xte = np.hstack([Xte16, pca.transform(Xteapp)])
    else:
        Xtr, Xte = Xtr16, Xte16
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=800).fit(A, Ytr[:, i]).predict(B)
    return pr

log(f"\n---\n## exp57: チャンピオン(16D+生パッチPCA20)の頑健性診断（MPII {len(pids)}人, person-indep）")

# (1) X軸/Y軸別(正規化誤差 median|dx|,|dy|)
log(f"\n### (1) X軸/Y軸別 person-indep（正規化誤差, 小さいほど良い）")
dx16, dy16, dxc, dyc = [], [], [], []
for p in pids:
    tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
    p16 = predict(X16[tr], Xapp[tr], y[tr], X16[te], Xapp[te], 20, 1e-3, False)
    pc = predict(X16[tr], Xapp[tr], y[tr], X16[te], Xapp[te], 20, 1e-3, True)
    dx16.append(np.median(np.abs(p16[:,0]-y[te,0]))); dy16.append(np.median(np.abs(p16[:,1]-y[te,1])))
    dxc.append(np.median(np.abs(pc[:,0]-y[te,0]))); dyc.append(np.median(np.abs(pc[:,1]-y[te,1])))
log(f"  {'':>14} | {'X誤差':>8} | {'Y誤差':>8}")
log(f"  {'16D':>14} | {np.median(dx16):.4f} | {np.median(dy16):.4f}")
log(f"  {'16D+appPCA20':>14} | {np.median(dxc):.4f} | {np.median(dyc):.4f}")
ix = (np.median(dx16)-np.median(dxc))/np.median(dx16)*100
iy = (np.median(dy16)-np.median(dyc))/np.median(dy16)*100
log(f"  → 改善: X {ix:+.1f}% / Y {iy:+.1f}%  (アピアランスがどちらの軸に効くか)")

# (2) α×PCA次元 安定性(全設定で16Dを下回れば過学習でない)
log(f"\n### (2) α×PCA次元の安定性 person-indep（cm@幅30, 基準16D=6.17）")
log(f"  {'α\\PCA':>8} | " + " | ".join(f"PCA{n}".rjust(8) for n in [16,24,32]))
for alpha in [1e-3, 1e-2, 1e-1]:
    row = []
    for npc in [16, 24, 32]:
        e = []
        for p in pids:
            tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
            pc = predict(X16[tr], Xapp[tr], y[tr], X16[te], Xapp[te], npc, alpha, True)
            e.append(np.median(np.sqrt(np.sum((pc-y[te])**2,axis=1)))*30.0)
        row.append(np.median(e))
    log(f"  {alpha:>8g} | " + " | ".join(f"{v:7.3f}c" for v in row))
log("\n**判定**: (1)でX/Yどちらに効くか把握。(2)全α×PCAで16D(6.17)を下回れば=改善は設定に依らず安定=過学習でない確定。")
