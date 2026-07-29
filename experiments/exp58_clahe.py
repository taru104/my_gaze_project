"""exp58: CLAHE(局所コントラスト正規化)パッチ vs plainパッチ。照明ロバスト化でperson-indep(他人汎化)を押せるか。
過学習警戒: person-indep主指標、PCA16固定(exp57最適)。mainは触らない。
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
def err(pred, gt): return np.sqrt(np.sum((pred - gt) ** 2, axis=1)) * 30.0

d = np.load(str(ROOT / "cache" / "mpii_app2.npz"))
X16, Xplain, Xclahe, y, pid = d["X16"], d["Xplain"], d["Xclahe"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))
NPC = 16

def predict(tr, te, mode):
    if mode == "16d":
        Xtr, Xte = X16[tr], X16[te]
    else:
        src = {"plain": Xplain, "clahe": Xclahe}[mode]
        pca = PCA(n_components=min(NPC, len(tr) - 1)).fit(src[tr])
        Xtr = np.hstack([X16[tr], pca.transform(src[tr])]); Xte = np.hstack([X16[te], pca.transform(src[te])])
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(A, y[tr][:, i]).predict(B)
    return pr

MODES = ["16d", "plain", "clahe"]
log(f"\n---\n## exp58: CLAHE vs plain パッチ（MPII {len(pids)}人, {len(X16)}枚, PCA16, person-indep主）")

log(f"\n### (主) person-independent（基準16D=6.17, plain(exp57)=5.60）")
lo = {m: [] for m in MODES}
for p in pids:
    tr = np.where(pid != p)[0]; te = np.where(pid == p)[0]
    for m in MODES: lo[m].append(np.median(err(predict(tr, te, m), y[te])))
for m in MODES: log(f"  {m:>7} | {np.median(lo[m]):.3f}cm")

log(f"\n### (副) 個人内 時間分割（基準16D=5.20, plain=5.01）")
inn = {m: [] for m in MODES}
for p in pids:
    idx = np.where(pid == p)[0]; cut = len(idx) // 2; tr, te = idx[:cut], idx[cut:]
    for m in MODES: inn[m].append(np.median(err(predict(tr, te, m), y[te])))
for m in MODES: log(f"  {m:>7} | {np.median(inn[m]):.3f}cm")

lp, lc = np.median(lo["plain"]), np.median(lo["clahe"])
log(f"\n**判定(過学習警戒)**: CLAHEがplainより person-indep で下なら照明ロバスト化が汎化に効く。"
    f"person-indep plain={lp:.3f} / clahe={lc:.3f} → {'★CLAHE採用' if lc < lp - 0.05 else '差なし/plainで十分'}")
