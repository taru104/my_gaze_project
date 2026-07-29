"""exp53: exp52のアピアランス改善(個人内4.50→2.92cm)が本物か、リーク由来かを厳しい分割で検証。
ランダム50/50は連続フレームの near-duplicate 画像が train/test に漏れ過大評価の恐れ。
→ 時間分割(前半学習/後半評価)＋ブロックk-fold(連続ブロック)で再評価。改善が残れば本物。GPU不要。mainは触らない。
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
NPC = 40
MODES = ["16d", "app", "both"]

def build(tr, te, mode):
    if mode == "16d": return X16[tr], X16[te]
    pca = PCA(n_components=min(NPC, len(tr) - 1)).fit(Xapp[tr])
    Atr, Ate = pca.transform(Xapp[tr]), pca.transform(Xapp[te])
    if mode == "app": return Atr, Ate
    return np.hstack([X16[tr], Atr]), np.hstack([X16[te], Ate])

log(f"\n---\n## exp53: exp52の改善が本物か厳しい分割で検証（MPII {len(pids)}人）")

# (1) 時間分割: 各人 前半学習→後半評価(near-duplicateが時間で分離)
log(f"\n### (1) 時間分割（各人 前半50%学習→後半50%評価）")
res = {m: [] for m in MODES}
for p in pids:
    idx = np.where(pid == p)[0]           # cache順=ファイル順=時間順
    if len(idx) < 40: continue
    cut = len(idx) // 2
    tr, te = idx[:cut], idx[cut:]
    for m in MODES:
        Xtr, Xte = build(tr, te, m)
        res[m].append(np.median(err(fit_predict(Xtr, y[tr], Xte), y[te])))
for m in MODES:
    log(f"  {m:>5} | {np.median(res[m]):.3f}cm")
t16, tboth = np.median(res["16d"]), np.median(res["both"])

# (2) ブロック5-fold: 連続ブロックでfold分け(near-duplicate分離)
log(f"\n### (2) ブロック5-fold（連続ブロックでtrain/test分離, 各人平均）")
res2 = {m: [] for m in MODES}
K = 5
for p in pids:
    idx = np.where(pid == p)[0]
    if len(idx) < 50: continue
    blocks = np.array_split(idx, K)
    for kf in range(K):
        te = blocks[kf]; tr = np.concatenate([blocks[j] for j in range(K) if j != kf])
        for m in MODES:
            Xtr, Xte = build(tr, te, m)
            res2[m].append(np.median(err(fit_predict(Xtr, y[tr], Xte), y[te])))
for m in MODES:
    log(f"  {m:>5} | {np.median(res2[m]):.3f}cm")

log(f"\n**判定**: 時間分割・ブロックfoldでも both<16d が残れば=アピアランス改善は本物(リークでない)。")
imp = (t16 - tboth) / t16 * 100
log(f"  時間分割: 16d={t16:.3f} → both={tboth:.3f}  ({imp:+.1f}%)  "
    f"{'★本物=画像特徴は個人キャリブありで有効' if tboth < t16 - 0.1 else '×リーク疑い/効果消失'}")
