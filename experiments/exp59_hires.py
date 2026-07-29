"""exp59: 高解像度パッチ(48x32)がアピアランス改善を伸ばすか=頭打ちかの分岐判定。
基準: 24x16 CLAHE(exp58) person-indep5.56/個人内4.87。過学習警戒: person-indep主, PCA低次元。mainは触らない。
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

d = np.load(str(ROOT / "cache" / "mpii_app3.npz"))
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

log(f"\n---\n## exp59: 高解像度パッチ48x32（{len(X16)}枚, hi次元={Xhi.shape[1]}）頭打ち判定")
log(f"  基準: 16D=6.17 / 24x16CLAHE person-indep=5.56・個人内=4.87")
NPCS = [0, 16, 40, 80]
log(f"\n### person-indep（主）")
for npc in NPCS:
    e = [np.median(err(predict(np.where(pid != p)[0], np.where(pid == p)[0], npc), y[pid == p])) for p in pids]
    lab = "16D" if npc == 0 else f"16D+hiPCA{npc}"
    log(f"  {lab:>14} | {np.median(e):.3f}cm")
log(f"\n### 個人内 時間分割（副）")
for npc in NPCS:
    e = []
    for p in pids:
        idx = np.where(pid == p)[0]; cut = len(idx)//2
        e.append(np.median(err(predict(idx[:cut], idx[cut:], npc), y[idx[cut:]])))
    lab = "16D" if npc == 0 else f"16D+hiPCA{npc}"
    log(f"  {lab:>14} | {np.median(e):.3f}cm")
log("\n**判定**: 高解像度が24x16CLAHE(person-indep5.56/個人内4.87)を明確に下回れば『アピアランスに伸びしろ=更に攻める』。"
    "同等以下なら『手作りアピアランスは頭打ち=その先はCNN』の客観的線引き。")
