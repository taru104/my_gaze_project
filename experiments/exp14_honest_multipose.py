"""exp14: exp13のhonest版。target位置(画面上の点)でgroup-splitし、testのターゲット位置はtrainに
無い状態で評価する(leave-target-out=未知の画面位置を当てる)。隣接フレーム相関による楽観を排除。
姿勢は全姿勢が train/test 双方に入る(=多姿勢キャリブ範囲内)。7D幾何+Huber、公知手法のみ。mainは触らない。
"""
import sys, glob
from pathlib import Path
from collections import defaultdict
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
log("\n---\n## exp14: honest評価(target位置group-split=未知の画面位置, 姿勢は範囲内, 7D+Huber)")
binerr = {b: [] for b in BINS}
overall = []
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: pts = extract_points(binp)
    except Exception: continue
    if len(pts) < 40: continue
    yaws = np.array([p[1] for p in pts])
    if (yaws >= 20).sum() < 5: continue
    groups = defaultdict(list)
    for p in pts:
        groups[(round(float(p[2][0]), 1), round(float(p[2][1]), 1))].append(p)
    gkeys = list(groups.keys())
    if len(gkeys) < 4: continue                 # 位置が少なすぎると group-split できない
    order = rng.permutation(len(gkeys))
    cut = max(2, int(len(gkeys) * 0.7))
    train = [p for i in order[:cut] for p in groups[gkeys[i]]]
    test  = [p for i in order[cut:] for p in groups[gkeys[i]]]
    if len(train) < 20 or len(test) < 8: continue
    e = euc(fit_predict(train, test), test)
    overall += list(e); n_sess += 1
    for j, p in enumerate(test):
        for (lo, hi) in BINS:
            if lo <= p[1] < hi: binerr[(lo, hi)].append(e[j]); break
log(f"\n**honest 姿勢bin別({n_sess}セッション, 未知の画面位置を多姿勢キャリブで当てる)**")
for (lo, hi) in BINS:
    es = np.array(binerr[(lo, hi)])
    if len(es):
        log(f"  |yaw| {lo:2d}-{hi:2d}°: n={len(es):5d}  median={np.median(es):.2f}cm  mean={es.mean():.2f}cm")
if overall:
    o = np.array(overall)
    log(f"  overall : n={len(o):5d}  median={np.median(o):.2f}cm  mean={o.mean():.2f}cm")
    log("\n### exp14 の位置づけ")
    log("- これが応募書に載せられる honest な数値(未知画面位置×姿勢範囲内)。exp13(ランダム)より辛いはず。")
    log("- exp12(正面のみ外挿)の横向き7.86cmと比べ、横向きbinがどれだけ下がるかが多姿勢キャリブの正味の効果。")
