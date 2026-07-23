"""exp16: exp15のタップ適応で正面点に重みwを付け、正面を維持したまま横向きを改善できるか。
train = 正面初期キャリブ(weight=w) + 横向きタップ40個(weight=1)。w=1,3,5,10 でトレードオフ曲線。
HuberRegressor(sample_weight)。公知手法のみ。mainは触らない。
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

def fit_predict_w(train, weights, test):
    Xtr = np.array([p[0] for p in train]); Ytr = np.array([p[2] for p in train])
    Xte = np.array([p[0] for p in test]); w = np.asarray(weights, float)
    sc = StandardScaler().fit(Xtr)
    Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    pred = np.zeros((len(test), 2))
    for a in range(2):
        m = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(Xtr2, Ytr[:, a], sample_weight=w)
        pred[:, a] = m.predict(Xte2)
    return pred

def euc(pred, test):
    tgt = np.array([p[2] for p in test]); dd = pred - tgt
    return np.hypot(dd[:, 0] * SW, dd[:, 1] * SH)

WS = [1, 3, 5, 10]
N_TAP = 40
rng = np.random.RandomState(0)
log("\n---\n## exp16: 正面重み付きタップ適応(正面w × 横向きタップ40個, 7D+Huber)")
front_err = {w: [] for w in WS}
side_err  = {w: [] for w in WS}
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: pts = extract_points(binp)
    except Exception: continue
    front = [p for p in pts if p[1] < 10]
    side  = [p for p in pts if p[1] >= 20]
    if len(front) < 24 or len(side) < 60: continue
    front = [front[i] for i in rng.permutation(len(front))]
    side  = [side[i]  for i in rng.permutation(len(side))]
    fc = int(len(front) * 0.6); tf_train, tf_test = front[:fc], front[fc:]
    sc_ = int(len(side) * 0.5); sp_pool, sp_test = side[:sc_], side[sc_:]
    if len(tf_test) < 6 or len(sp_test) < 10 or len(sp_pool) < N_TAP: continue
    n_sess += 1
    tap = sp_pool[:N_TAP]
    for w in WS:
        train = tf_train + tap
        weights = [w] * len(tf_train) + [1.0] * len(tap)
        front_err[w] += list(euc(fit_predict_w(train, weights, tf_test), tf_test))
        side_err[w]  += list(euc(fit_predict_w(train, weights, sp_test), sp_test))
log(f"\n**正面重み w ごとの誤差({n_sess}セッション横断, median, タップ40個固定)**")
log(f"  {'w':>3} | {'正面test(2.1cm維持狙い)':>20} | {'横向きtest(下げたい)':>18}")
for w in WS:
    fe = np.median(front_err[w]) if front_err[w] else float('nan')
    se = np.median(side_err[w])  if side_err[w]  else float('nan')
    log(f"  {w:>3} | {fe:>18.2f}cm | {se:>16.2f}cm")
if n_sess:
    log("\n### exp16の判定")
    log("- 比較基準: exp15(w=1相当)は正面2.40cm/横向き5.56cm。タップ無し初期は正面2.12cm/横向き7.73cm。")
    best = min(WS, key=lambda w: np.median(front_err[w]) + 0.5 * np.median(side_err[w]))
    log(f"- 正面重視で正面を2.1cm付近に戻しつつ横向きも下がるwがあれば、案2は『正面無犠牲で横向き改善』に格上げ。")
    log(f"- 参考ベスト(正面+0.5*横向き最小): w={best} → 正面{np.median(front_err[best]):.2f}cm/横向き{np.median(side_err[best]):.2f}cm")
