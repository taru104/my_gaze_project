"""exp13: 多姿勢キャリブ「範囲内」の実機精度を姿勢bin別に測る。
exp12は未キャリブ姿勢(>=20°)への外挿=厳しめ。exp13は「全姿勢からキャリブして、その範囲内で使う」
実機シナリオ。各セッションのhas_target点を train/test 分割(全姿勢混在)し、testを姿勢bin別に評価。
公知手法(7D幾何 + Huber回帰)のみ。mainは触らない。
"""
import sys, glob
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from raw_landmark_logger import load_raw_landmarks
from features import build_7d_features
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT2.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, a): self.x = float(a[0]); self.y = float(a[1]); self.z = float(a[2])

def extract_points(binp):
    d = load_raw_landmarks(binp)
    idx = np.where(d["has_target"])[0]
    pts = []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        lms = [LM(p) for p in d["landmarks"][k]]
        r = build_7d_features(lms, int(d["img_w"][k]), int(d["img_h"][k]))
        if r is None: continue
        feat, yaw, _, _ = r
        pts.append((np.asarray(feat, float), abs(np.degrees(yaw)), np.asarray(t, float)))
    return pts

def fit_predict(train, test):
    Xtr = np.array([p[0] for p in train]); Ytr = np.array([p[2] for p in train])
    Xte = np.array([p[0] for p in test])
    sc = StandardScaler().fit(Xtr)
    Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    pred = np.zeros((len(test), 2))
    for a in range(2):
        m = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(Xtr2, Ytr[:, a])
        pred[:, a] = m.predict(Xte2)
    return pred

def euc(pred, test):
    tgt = np.array([p[2] for p in test]); dd = pred - tgt
    return np.hypot(dd[:, 0] * SW, dd[:, 1] * SH)

BINS = [(0, 10), (10, 20), (20, 30), (30, 90)]
rng = np.random.RandomState(0)
log("\n---\n## exp13: 多姿勢キャリブ範囲内の実機精度(train/test, 姿勢bin別, 7D+Huber)")
binerr = {b: [] for b in BINS}
overall = []
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: pts = extract_points(binp)
    except Exception: continue
    if len(pts) < 40: continue
    yaws = np.array([p[1] for p in pts])
    if (yaws >= 20).sum() < 5: continue  # 横向きが含まれるセッションのみ(多姿勢の意味がある)
    idx = rng.permutation(len(pts)); cut = int(len(pts) * 0.7)
    train = [pts[i] for i in idx[:cut]]; test = [pts[i] for i in idx[cut:]]
    if len(train) < 20 or len(test) < 8: continue
    e = euc(fit_predict(train, test), test)
    overall += list(e); n_sess += 1
    for j, p in enumerate(test):
        for (lo, hi) in BINS:
            if lo <= p[1] < hi: binerr[(lo, hi)].append(e[j]); break
log(f"\n**姿勢bin別({n_sess}セッション, 全姿勢からキャリブ→範囲内で評価)**")
for (lo, hi) in BINS:
    es = np.array(binerr[(lo, hi)])
    if len(es):
        log(f"  |yaw| {lo:2d}-{hi:2d}°: n={len(es):5d}  median={np.median(es):.2f}cm  mean={es.mean():.2f}cm")
if overall:
    o = np.array(overall)
    log(f"  overall : n={len(o):5d}  median={np.median(o):.2f}cm  mean={o.mean():.2f}cm")
    log("\n### exp13 の意味")
    log("- exp12(正面のみ→横向き外挿)は横向き7cm台。exp13(多姿勢キャリブ→範囲内)は上表の通り。")
    log("- 横向きbinがexp12より大きく下がれば『横向きもキャリブすれば横向きは実用精度』を実証=main改善の方向確定。")
