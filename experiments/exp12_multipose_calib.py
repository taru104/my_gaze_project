"""exp12: 多姿勢キャリブの価値を既存データで実証。
仮説「横向きが崩れるのは正面キャリブからの外挿だから。中間姿勢を学習に足せば横向きが改善する」。
.bin(has_target点)から7D特徴+yaw+正解を再構成し、
  test = 横向き(|yaw|>=20°)
  A(正面のみ学習): |yaw|<10° の点で学習
  B(多姿勢学習):  |yaw|<20° の点で学習(中間10-20°を追加。testの>=20°は不使用)
で test の誤差を比較。公知手法(7D幾何 + Huber回帰)のみ。mainは触らない。
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
    tgt = np.array([p[2] for p in test])
    d = pred - tgt
    return np.hypot(d[:, 0] * SW, d[:, 1] * SH)

log("\n---\n## exp12: 多姿勢キャリブの外挿効果(7D+Huber, 公知手法)")
errA_all, errB_all = [], []
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try:
        pts = extract_points(binp)
    except Exception as e:
        continue
    yaws = np.array([p[1] for p in pts]) if pts else np.array([])
    test  = [p for p in pts if p[1] >= 20]
    trainA = [p for p in pts if p[1] < 10]
    trainB = [p for p in pts if p[1] < 20]
    if len(test) < 5 or len(trainA) < 8 or len(trainB) < len(trainA) + 3:
        continue  # 各群が十分あるセッションだけ
    predA = fit_predict(trainA, test); eA = euc(predA, test)
    predB = fit_predict(trainB, test); eB = euc(predB, test)
    errA_all += list(eA); errB_all += list(eB); n_sess += 1
    log(f"  {Path(binp).name[:24]}: test={len(test)} "
        f"A正面のみ(n={len(trainA)})={np.median(eA):.2f}cm  "
        f"B多姿勢(n={len(trainB)})={np.median(eB):.2f}cm")
if errA_all:
    a, b = np.array(errA_all), np.array(errB_all)
    log(f"\n**集計({n_sess}セッション, 横向きtest点{len(a)}個)**")
    log(f"  A 正面のみキャリブ: median={np.median(a):.2f}cm  mean={a.mean():.2f}cm")
    log(f"  B 多姿勢キャリブ:   median={np.median(b):.2f}cm  mean={b.mean():.2f}cm")
    imp = (np.median(a) - np.median(b)) / np.median(a) * 100
    log(f"  → 多姿勢キャリブで横向き誤差 {imp:+.1f}% (正なら改善=多姿勢データが効く)")
else:
    log("  条件を満たすセッションが不足。閾値やtest定義の調整が必要。")
