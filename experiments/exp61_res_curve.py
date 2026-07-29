"""exp61: 96x64パッチ(解像度曲線の3点目)。24x16→48x32→96x64で「解像度→精度」を描き、
手作りアピアランスの天井/CNN行きの分岐を客観判定。過学習警戒: PCA16固定, person-indep主。mainは触らない。
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

d = np.load(str(ROOT / "cache" / "mpii_app4.npz"))
X16, Xhi, y, pid = d["X16"], d["Xhi"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))

def predict(tr, te, npc):
    if npc == 0:
        Xtr, Xte = X16[tr], X16[te]
    else:
        pca = PCA(n_components=min(npc, len(tr) - 1)).fit(Xhi[tr])
        Xtr = np.hstack([X16[tr], pca.transform(Xhi[tr])]); Xte = np.hstack([X16[te], pca.transform(Xhi[te])])
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(A, y[tr][:, i]).predict(B)
    return pr

log(f"\n---\n## exp61: 96x64パッチ（{len(X16)}枚, hi次元={Xhi.shape[1]}）解像度曲線3点目")
for npc in [0, 16, 24]:
    lo = [np.median(err(predict(np.where(pid!=p)[0], np.where(pid==p)[0], npc), y[pid==p])) for p in pids]
    inn = []
    for p in pids:
        idx = np.where(pid==p)[0]; cut=len(idx)//2
        inn.append(np.median(err(predict(idx[:cut], idx[cut:], npc), y[idx[cut:]])))
    lab = "16D" if npc==0 else f"16D+96x64_PCA{npc}"
    log(f"  {lab:>18} | person-indep {np.median(lo):.3f}cm | 個人内 {np.median(inn):.3f}cm")
log(f"\n**解像度曲線(PCA16, person-indep / 個人内)**:")
log(f"  24x16=5.56/4.87 → 48x32=5.54/4.46 → 96x64=上記。単調改善なら伸びしろ継続→CNN価値大。頭打ちなら48x32が手作り天井。")
