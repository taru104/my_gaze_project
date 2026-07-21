"""7時間耐久 実験8: OneEuro時間平滑後の実効精度(実利用に近い数字)。
各セッションを時系列順にし、前半でキャリブ(16D Huber fit)→後半を連続予測してOneEuroで平滑。
生予測 vs 平滑後 の姿勢bin別誤差。実利用ではフィルタで平滑されるので、これが体感に近い。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from filters import OneEuroFilter2D
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
REPORT = ROOT / "experiments" / "REPORT.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

GOOD = ["20260716_130217", "20260717_165617", "20260717_174621"]
t0 = time.time()
log(f"\n---\n## 実験8: OneEuro時間平滑後の実効精度(実利用に近い)  全体|0-10 10-20 20-30 30+")
raw_all, sm_all, y_all, yd_all = [], [], [], []
for sid in GOOD:
    d = np.load(ROOT/"logs"/f"session_{sid}_rich16d.npz"); m = d["has_target"].astype(bool)
    X = d["X"][m][:, :16]; Y = d["y_norm"][m]; TS = d["time_s"][m]
    order = np.argsort(TS); X, Y, TS = X[order], Y[order], TS[order]
    half = len(X)//2
    sc = StandardScaler().fit(X[:half])
    hx = HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=800).fit(sc.transform(X[:half]), Y[:half,0])
    hy = HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=800).fit(sc.transform(X[:half]), Y[:half,1])
    Xte, Yte, TSte = X[half:], Y[half:], TS[half:]
    raw = np.column_stack([hx.predict(sc.transform(Xte)), hy.predict(sc.transform(Xte))])
    f = OneEuroFilter2D(min_cutoff=1.0, beta=0.05)
    sm = np.zeros_like(raw); prev = TSte[0]
    for i in range(len(raw)):
        dt = TSte[i] - prev if i > 0 else 1/30.0
        sm[i] = f.update(raw[i], float(np.clip(dt, 1e-3, 0.5))); prev = TSte[i]
    raw_all.append(raw); sm_all.append(sm); y_all.append(Yte); yd_all.append(np.abs(np.degrees(Xte[:,5])))
raw_all = np.vstack(raw_all); sm_all = np.vstack(sm_all); y_all = np.vstack(y_all); yd_all = np.concatenate(yd_all)
er = euc(raw_all, y_all); es = euc(sm_all, y_all)
log(f"  16D 生予測      : {np.median(er):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(er, yd_all)))
log(f"  16D +OneEuro平滑 : {np.median(es):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(es, yd_all)))
log(f"→ 平滑は静止時のジッタを下げるが姿勢変化には遅延。実利用の体感はこの中間。")
log(f"\n(実験8 完了 {(time.time()-t0)/60:.1f}分)")
