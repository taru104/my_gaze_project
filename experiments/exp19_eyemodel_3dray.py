"""exp19: z非依存の3D眼球モデル(Orlosky本来のやり方)。フェーズ3の本命。
exp17/18の敗因=MediaPipeのz直接使用。ここではzを一切使わず:
  solvePnPで頭部姿勢(R,t) → 眼球中心を3D顔モデル(mm)に定義しカメラ座標へ →
  虹彩2Dピクセルをカメラ光線に逆投影 → 眼球球(中心=眼球中心,半径EYE_R)との交点=虹彩3D →
  視線方向 = 虹彩3D - 眼球中心。
視線方向を特徴に多点キャリブで画面座標へHuber回帰、honest評価、姿勢bin別。7Dと比較。
公知手法(Orlosky公開コード+標準ピンホール幾何)。mainは触らない。
"""
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from raw_landmark_logger import load_raw_landmarks
from features import build_7d_features, _FACE_3D_MODEL, _FACE_2D_IDX, _DIST_COEFFS
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT3_eyemodel.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

L_IRIS, R_IRIS = 468, 473
EYE_R = 12.0  # 眼球半径mm(標準)
# 眼球中心を顔モデル(mm,鼻尖原点)に定義: 外眼角(33:-43.3,32.7,-26 / 263:+43.3,...)から内・奥
EYE_L_MODEL = np.array([-31.0, 33.0, -39.0])
EYE_R_MODEL = np.array([ 31.0, 33.0, -39.0])

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def eye_gaze_dir(lm, w, h):
    """z非依存でカメラ座標系の視線方向(左右平均)を返す。失敗時None。"""
    face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in _FACE_2D_IDX], dtype=np.float64)
    f = float(w)
    cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    R, _ = cv2.Rodrigues(rvec); t = tvec.ravel()
    def one(eye_model, iris_idx):
        eye_cam = R @ eye_model + t
        ix, iy = lm[iris_idx].x * w, lm[iris_idx].y * h
        ray = np.array([(ix - w / 2.0) / f, (iy - h / 2.0) / f, 1.0]); ray /= np.linalg.norm(ray)
        L = -eye_cam
        b = 2.0 * ray @ L; c = L @ L - EYE_R ** 2; disc = b * b - 4.0 * c
        if disc < 0:
            tt = -(ray @ L) / (ray @ ray)            # 接近点(球に届かない時)
        else:
            tt = (-b - np.sqrt(disc)) / 2.0          # 手前の交点
        iris3d = tt * ray
        g = iris3d - eye_cam
        n = np.linalg.norm(g)
        return g / n if n > 1e-9 else None
    gl = one(EYE_L_MODEL, L_IRIS); gr = one(EYE_R_MODEL, R_IRIS)
    if gl is None or gr is None: return None
    g = gl + gr; n = np.linalg.norm(g)
    return g / n if n > 1e-9 else None

def euc(pred, tgt):
    dd = pred - tgt; return np.hypot(dd[:, 0] * SW, dd[:, 1] * SH)

def fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    pred = np.zeros((len(Xte), 2))
    for a in range(2):
        pred[:, a] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(Xtr2, Ytr[:, a]).predict(Xte2)
    return pred

BINS = [(0,10),(10,20),(20,30),(30,90)]
rng = np.random.RandomState(0)
log("\n---\n## exp19: z非依存3D眼球モデル(solvePnP+球面逆投影) vs 7D（honest, 姿勢bin別）")
eye_bin = {b: [] for b in BINS}; d7_bin = {b: [] for b in BINS}
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 40: continue
    Xeye, X7, Y, YAW = [], [], [], []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        w, h = float(d["img_w"][k]), float(d["img_h"][k])
        lm = [LM(p) for p in d["landmarks"][k]]
        g = eye_gaze_dir(lm, w, h)
        if g is None: continue
        r7 = build_7d_features(lm, int(w), int(h))
        if r7 is None: continue
        feat7, yaw, _, _ = r7
        Xeye.append(g); X7.append(np.asarray(feat7, float)); Y.append(np.asarray(t, float))
        YAW.append(abs(np.degrees(yaw)))
    Xeye = np.array(Xeye); X7 = np.array(X7); Y = np.array(Y); YAW = np.array(YAW)
    if len(Y) < 40 or (YAW >= 20).sum() < 5: continue
    groups = defaultdict(list)
    for i, t in enumerate(Y): groups[(round(t[0], 1), round(t[1], 1))].append(i)
    gk = list(groups.keys())
    if len(gk) < 4: continue
    order = rng.permutation(len(gk)); cut = max(2, int(len(gk) * 0.7))
    tr = [i for j in order[:cut] for i in groups[gk[j]]]
    te = [i for j in order[cut:] for i in groups[gk[j]]]
    if len(tr) < 20 or len(te) < 8: continue
    n_sess += 1
    pe = euc(fit_predict(Xeye[tr], Y[tr], Xeye[te]), Y[te])
    p7 = euc(fit_predict(X7[tr],  Y[tr], X7[te]),  Y[te])
    for j, i in enumerate(te):
        for (lo, hi) in BINS:
            if lo <= YAW[i] < hi:
                eye_bin[(lo, hi)].append(pe[j]); d7_bin[(lo, hi)].append(p7[j]); break
log(f"\n**z非依存3D眼球モデル vs 7D統計（{n_sess}セッション, honest, median cm）**")
log(f"  {'|yaw|':>8} | {'眼球3D(z無)':>12} | {'7D統計':>10}")
for (lo, hi) in BINS:
    e = eye_bin[(lo, hi)]; s = d7_bin[(lo, hi)]
    if e and s:
        log(f"  {lo:2d}-{hi:2d}° | {np.median(e):>10.2f}cm | {np.median(s):>8.2f}cm")
alle = [v for b in BINS for v in eye_bin[b]]; alls = [v for b in BINS for v in d7_bin[b]]
if alle:
    log(f"  overall | {np.median(alle):>10.2f}cm | {np.median(alls):>8.2f}cm")
    log(f"- exp17(z直接,鼻PCA)8.81 / exp18(z直接,PnP)13.49 → exp19(z非依存)={np.median(alle):.2f}cm")
