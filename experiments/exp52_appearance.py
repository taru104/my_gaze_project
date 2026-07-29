"""exp52: 画像アピアランス特徴 vs 幾何16D(MPII客観)。目領域の正規化グレースケールパッチ(PCA圧縮)が
16Dに情報を足すか。「幾何が捨てた画像情報が視線に効くか」の客観検証。
 16D単体 / appearance単体 / 16D+appearance を MPII個人内・person-indep で比較。GPU不要。mainは触らない。
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

d = np.load(str(ROOT / "cache" / "mpii_app.npz"))
X16, Xapp, y, pid = d["X16"], d["Xapp"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))
log(f"\n---\n## exp52: 画像アピアランス特徴 vs 幾何16D（MPII {len(pids)}人, {len(X16)}枚, パッチ768D→PCA）")

NPC = 40   # appearance PCA次元
rng = np.random.RandomState(0)

def build(tr, te, mode):
    """mode: '16d' / 'app' / 'both'。PCAはtrで学習。"""
    if mode == "16d":
        return X16[tr], X16[te]
    pca = PCA(n_components=min(NPC, len(tr) - 1)).fit(Xapp[tr])
    Atr, Ate = pca.transform(Xapp[tr]), pca.transform(Xapp[te])
    if mode == "app":
        return Atr, Ate
    return np.hstack([X16[tr], Atr]), np.hstack([X16[te], Ate])

def err(pred, gt): return np.median(np.sqrt(np.sum((pred - gt) ** 2, axis=1))) * CM_W

MODES = ["16d", "app", "both"]

# (A) 個人内
log(f"\n### (A) 個人内キャリブ（各人 半分学習→半分評価, 基準16D≈4.0cm）")
res_in = {m: [] for m in MODES}
for p in pids:
    m = np.where(pid == p)[0]
    if len(m) < 30: continue
    idx = rng.permutation(len(m)); cut = len(m) // 2
    tr, te = m[idx[:cut]], m[idx[cut:]]
    for mode in MODES:
        Xtr, Xte = build(tr, te, mode)
        res_in[mode].append(err(fit_predict(Xtr, y[tr], Xte), y[te]))
for mode in MODES:
    log(f"  {mode:>5} | {np.median(res_in[mode]):.3f}cm")

# (B) person-independent (LOPO)
log(f"\n### (B) person-independent（14人学習→未知1人, 基準16D≈6.18cm）")
res_lo = {m: [] for m in MODES}
for p in pids:
    te = np.where(pid == p)[0]; tr = np.where(pid != p)[0]
    for mode in MODES:
        Xtr, Xte = build(tr, te, mode)
        res_lo[mode].append(err(fit_predict(Xtr, y[tr], Xte), y[te]))
for mode in MODES:
    log(f"  {mode:>5} | {np.median(res_lo[mode]):.3f}cm")

b_in = min(np.median(res_in[m]) for m in MODES)
b_lo = min(np.median(res_lo[m]) for m in MODES)
log(f"\n**判定**: 'both'が'16d'を両方で下回れば=画像情報が幾何に足せる=有意。"
    f"最良 個人内={b_in:.3f}(16d={np.median(res_in['16d']):.3f}) / person-indep={b_lo:.3f}(16d={np.median(res_lo['16d']):.3f})。"
    f"appだけ悪ければ『生パッチ+線形では不足=CNN/正規化が要る』の傍証。")
