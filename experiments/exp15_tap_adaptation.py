"""exp15: 推奨案2『タップのオンライン適応』のオフライン検証。
シナリオ: 正面点(|yaw|<10°)だけで初期学習 → 横向き点(|yaw|>=20°=タップ代理)をN個ずつ足して再fit。
検証: (a)横向きtest誤差が下がるか (b)正面test誤差が維持されるか(exp14で混在は正面を犠牲にしたので要確認)。
7D幾何+Huber、公知手法のみ。mainは触らない。
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

NS = [0, 5, 10, 20, 40]
rng = np.random.RandomState(0)
log("\n---\n## exp15: タップのオンライン適応(正面初期学習→横向きタップをN個追加, 7D+Huber)")
# 各Nで、正面test誤差・横向きtest誤差をセッション横断でプール
front_err = {n: [] for n in NS}
side_err  = {n: [] for n in NS}
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: pts = extract_points(binp)
    except Exception: continue
    front = [p for p in pts if p[1] < 10]
    side  = [p for p in pts if p[1] >= 20]
    if len(front) < 24 or len(side) < 60: continue
    front = [front[i] for i in rng.permutation(len(front))]
    side  = [side[i]  for i in rng.permutation(len(side))]
    fc = int(len(front) * 0.6)
    tf_train, tf_test = front[:fc], front[fc:]
    sc_ = int(len(side) * 0.5)
    sp_pool, sp_test = side[:sc_], side[sc_:]
    if len(tf_test) < 6 or len(sp_test) < 10 or len(sp_pool) < max(NS): continue
    n_sess += 1
    for n in NS:
        train = tf_train + sp_pool[:n]
        front_err[n] += list(euc(fit_predict(train, tf_test), tf_test))
        side_err[n]  += list(euc(fit_predict(train, sp_test), sp_test))
log(f"\n**タップ追加数 N ごとの誤差({n_sess}セッション横断, median)**")
log(f"  {'N':>3} | {'正面test(維持したい)':>18} | {'横向きtest(下げたい)':>18}")
for n in NS:
    fe = np.median(front_err[n]) if front_err[n] else float('nan')
    se = np.median(side_err[n])  if side_err[n]  else float('nan')
    log(f"  {n:>3} | {fe:>16.2f}cm | {se:>16.2f}cm")
if n_sess:
    f0, fN = np.median(front_err[0]), np.median(front_err[NS[-1]])
    s0, sN = np.median(side_err[0]),  np.median(side_err[NS[-1]])
    log(f"\n### exp15の判定")
    log(f"- 横向き: {s0:.2f}→{sN:.2f}cm ({(s0-sN)/s0*100:+.0f}%)  正面: {f0:.2f}→{fN:.2f}cm ({(f0-fN)/f0*100:+.0f}%)")
    verdict = ("横向き改善&正面維持=タップ適応は有効" if (sN < s0 * 0.9 and fN < f0 * 1.2)
               else "正面が犠牲=単純追加はNG。タップは横向き専用補正に限定すべき")
    log(f"- 判定: {verdict}")
