"""7時間耐久 実験6(実装推奨の裏付け): 正面=個人H1(1.4cm) + 横向き=16D Huber(3.84cm) を
キャリブ姿勢からの|Δyaw|でゲートブレンド。全姿勢で最良の構成を確認する。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import H1Calibration

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
REPORT = ROOT / "experiments" / "REPORT.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

GOOD = ["20260716_130217", "20260717_165617", "20260717_174621"]
Xs, ys = [], []
for sid in GOOD:
    d = np.load(ROOT/"logs"/f"session_{sid}_rich16d.npz"); m = d["has_target"].astype(bool)
    Xs.append(d["X"][m]); ys.append(d["y_norm"][m])
X16 = np.vstack(Xs); Y = np.vstack(ys); X7 = X16[:, :7]

def run(tau):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    PA, PB, PC, G, YD = [], [], [], [], []
    for p in np.unique(ids):
        te, tr = ids==p, ids!=p
        if tr.sum() < 30: continue
        h1 = H1Calibration()
        for f_, t in zip(X7[tr], Y[tr]): h1.add(f_, t[0], t[1], 1.0)
        try: h1.fit()
        except Exception: continue
        sc = StandardScaler().fit(X16[tr])
        hx = HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=800).fit(sc.transform(X16[tr]), Y[tr,0])
        hy = HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=800).fit(sc.transform(X16[tr]), Y[tr,1])
        pa = np.array([h1.predict(f_) for f_ in X7[te]])
        pb = np.column_stack([hx.predict(sc.transform(X16[te])), hy.predict(sc.transform(X16[te]))])
        ref = np.mean(np.abs(np.degrees(X16[tr,5])))
        yawte = np.abs(np.degrees(X16[te,5]))
        w = np.exp(-np.abs(yawte - ref) / tau)[:, None]
        pc = w * pa + (1 - w) * pb
        PA.append(pa); PB.append(pb); PC.append(pc); G.append(Y[te]); YD.append(yawte)
    PA, PB, PC, G, YD = np.vstack(PA), np.vstack(PB), np.vstack(PC), np.vstack(G), np.concatenate(YD)
    return PA, PB, PC, G, YD

t0 = time.time()
log(f"\n---\n## 実験6: H1×16D ハイブリッド(実装推奨の最良構成)  全体|0-10 10-20 20-30 30+")
PA, PB, PC, G, YD = run(tau=10.0)
for tag, P in [("A:個人H1(7D)", PA), ("B:16D Huber", PB), ("C:ゲートブレンド", PC)]:
    e = euc(P, G)
    log(f"  {tag:16s}: {np.median(e):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(e, YD)))
# tau sweep for C
log(f"### ゲートブレンド tau感度(全体/30+)")
for tau in [6, 10, 15, 25]:
    _, _, PC, G, YD = run(tau=tau)
    e = euc(PC, G); bv = binstats(e, YD)
    log(f"  tau={tau:2d}: 全体{np.median(e):.2f} 30+{bv[3]:.2f}")
log(f"\n(実験6 完了 {(time.time()-t0)/60:.1f}分)")
log(f"\n→ Cが正面はA並み・横向きはB並みなら、実装推奨は『正面H1+横向き16DのゲートブレンドHybridCalibration』。")
