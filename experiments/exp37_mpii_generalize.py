"""exp37: 16Dが他人(MPII 15人)でもロバストか。全データセットでのロバスト性検証。
mpii_16d_ck.npz (X:16D, y:正規化screen, pid:15人) で:
 (A)個人内キャリブ: 各人でランダムsplit学習→評価(実機の自分キャリブに相当)
 (B)person-independent: leave-one-person-out(14人学習,1人評価,キャリブなし)
正規化screen誤差＋概算cm(ラップtrap画面幅~30cm仮定,表示のみ)。デバイス非依存。mainは触らない。
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

REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
CM_W = 30.0  # 概算: MPIIラップトップ画面幅~30cm(cm表示は目安のみ、モデルは正規化=デバイス非依存)

d = np.load(str(ROOT / "cache" / "mpii_16d_ck.npz"))
X, y, pid = d["X"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))

def fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2): pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500).fit(A, Ytr[:, i]).predict(B)
    return pr

def err_norm(pred, gt):  # 正規化screen ユークリッド → 概算cm
    e = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    return np.median(e), np.median(e) * CM_W

rng = np.random.RandomState(0)
log("\n---\n## exp37: MPII 15人で16D汎用性（全データセットでのロバスト性）")

# (A) 個人内キャリブ
inner = []
for p in pids:
    m = pid == p; Xp, yp = X[m], y[m]
    idx = rng.permutation(len(Xp)); cut = int(len(Xp) * 0.5)
    tr, te = idx[:cut], idx[cut:]
    e_n, e_cm = err_norm(fit_predict(Xp[tr], yp[tr], Xp[te]), yp[te])
    inner.append((p, e_n, e_cm))
log("\n**(A) 個人内キャリブ（各人で自分のデータ半分キャリブ→半分評価）**")
log("  " + "  ".join(f"{p}={cm:.2f}cm" for p, _, cm in inner[:8]))
log("  " + "  ".join(f"{p}={cm:.2f}cm" for p, _, cm in inner[8:]))
ain = np.median([e for _, e, _ in inner]); acm = np.median([c for _, _, c in inner])
log(f"  → 15人 median: {ain:.4f}(正規化) ≈ {acm:.2f}cm。16Dが他人でも個人キャリブで動くか。")

# (B) person-independent (leave-one-person-out)
lopo = []
for p in pids:
    te_m = pid == p; tr_m = ~te_m
    e_n, e_cm = err_norm(fit_predict(X[tr_m], y[tr_m], X[te_m]), y[te_m])
    lopo.append((p, e_n, e_cm))
log("\n**(B) person-independent（14人学習→未知1人・キャリブなし）**")
bin_ = np.median([e for _, e, _ in lopo]); bcm = np.median([c for _, _, c in lopo])
log(f"  → 15人 median: {bin_:.4f}(正規化) ≈ {bcm:.2f}cm。キャリブなしで未知の人にどれだけ動くか。")
log(f"\n- 個人内 {acm:.2f}cm / person-indep {bcm:.2f}cm。個人キャリブありなら他人でも実用、が示せれば汎用性◎。")
log(f"- ※cmは画面幅30cm仮定の目安。モデル自体は正規化座標=デバイス非依存。SOTA(角度)とは評価軸が別。")
