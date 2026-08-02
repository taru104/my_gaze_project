"""Step2: リンバス楕円 A/B/C 評価。MPII15人 person-specific 5-fold・線形Huber固定・ペア比較。
baseline/A(中心)/B(形+傾き)/C(両方) を 全体＋|yaw|bin別 で。フォールバック率も出す。
Usage: .venv/Scripts/python.exe experiments/exp62_limbus_eval.py [cacheのtag(既定 '')]"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

CM = 30.0
tag = sys.argv[1] if len(sys.argv) > 1 else ""
REPORT = ROOT / "experiments" / "REPORT7_limbus.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

d = np.load(str(ROOT / "cache" / f"mpii_limbus{tag}.npz"))
X16, A, B, fbL, fbR, y, pid = d["X16"], d["A"], d["B"], d["fbL"], d["fbR"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))

def build(arm):
    X = X16.copy()
    if arm in ("A", "C"): X[:, :4] = A[:, :4]
    if arm in ("B", "C"):
        X[:, 14] = B[:, 0]; X[:, 15] = B[:, 1]
        X = np.hstack([X, B[:, 2:6]])
    return X

def fp(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); Aa, Bb = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=600).fit(Aa, Ytr[:, i]).predict(Bb)
    return pr

# 固定fold(全arm共通=ペア比較)
rng = np.random.RandomState(0)
folds = {}
for p in pids:
    mi = np.where(pid == p)[0]
    perm = rng.permutation(len(mi))
    folds[p] = [mi[perm[j::5]] for j in range(5)]

yaw_deg = np.degrees(np.abs(X16[:, 5]))
BINS = [(0, 10), (10, 20), (20, 30), (30, 90)]
ARMS = ["baseline", "A", "B", "C"]

log(f"\n## Step2: A/B/C 評価（MPII{len(pids)}人, person-specific 5-fold, 線形Huber, cache='{tag or 'full'}'）")
fball = 100 * (fbL.mean() + fbR.mean()) / 2
fb_pp = [100 * (fbL[pid == p].mean() + fbR[pid == p].mean()) / 2 for p in pids]
log(f"フォールバック率: 全体 {fball:.1f}%  被験者別レンジ {min(fb_pp):.0f}〜{max(fb_pp):.0f}%")
log(f"（※楕円失敗フレームはMediaPipe値にフォールバック=その分baselineと同じ。40%超なら結果は割引いて解釈）")
log(f"\n| 腕 | 全体 | 0-10° | 10-20° | 20-30° | 30+ |")
log(f"|---|---|---|---|---|---|")

results = {}
for arm in ARMS:
    X = build(arm)
    errs = np.full(len(X16), np.nan)
    for p in pids:
        fl = folds[p]
        for kf in range(5):
            te = fl[kf]; tr = np.concatenate([fl[j] for j in range(5) if j != kf])
            pr = fp(X[tr], y[tr], X[te])
            errs[te] = np.sqrt(np.sum((pr - y[te]) ** 2, axis=1)) * CM
    results[arm] = errs
    row = [f"{np.nanmedian(errs):.3f}"]
    for lo, hi in BINS:
        m = (yaw_deg >= lo) & (yaw_deg < hi) & ~np.isnan(errs)
        row.append(f"{np.nanmedian(errs[m]):.3f}" if m.sum() >= 30 else "n/a")
    log(f"| {arm} | " + " | ".join(row) + " |")

# 判定
base = np.nanmedian(results["baseline"])
best = min(ARMS[1:], key=lambda a: np.nanmedian(results[a]))
bv = np.nanmedian(results[best])
log(f"\n**判定**: baseline全体={base:.3f}cm。最良腕={best}({bv:.3f}cm, {(base-bv)/base*100:+.1f}%)。")
if bv < base - 0.02:
    log(f"→ {best}がbaselineを下回った。bin別で横向きの改善を確認。Step3(ダウンサンプル)へ。")
else:
    log(f"→ baselineを下回らず。停止条件によりここで終了。原因考察を research_log に追記。")
