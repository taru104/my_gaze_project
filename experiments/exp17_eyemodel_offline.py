"""exp17: 眼球モデル(Orlosky/Webcam3D式)のオフライン検証。フェーズ3。
Webcam3DTracker/MonitorTracking.py のコアを logs/*.bin に適用:
  鼻PCAで頭部座標系R → 眼球中心を頭部にロック(正面点で) → 視線方向=虹彩3D-眼球中心3D。
その視線方向(3D)を特徴に、多点キャリブで画面座標へHuber回帰し、honest(target-group-split)で姿勢bin別精度。
比較: 同じ点で従来の7D幾何特徴も回帰し、どちらが姿勢ロバストか(exp14の再現条件)。
狙い: 統計7Dの「正面vs横向きトレードオフ(正面すら6.86cm)」が眼球モデルで消えるか。
公知手法(Orlosky公開コード)。mainは触らない。
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
REPORT = ROOT / "experiments" / "REPORT3_eyemodel.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

NOSE = [4,45,275,220,440,1,5,51,281,44,274,241,461,125,354,218,438,195,167,393,165,391,3,248]
L_IRIS, R_IRIS = 468, 473
BASE_R = 20.0

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def head_frame(P, R_ref):
    """P:(478,3) in (x*w,y*h,z*w). 鼻PCAで頭部座標系。R_ref[0]で符号を時系列安定化。"""
    pts = P[NOSE]
    center = pts.mean(0)
    cov = np.cov((pts - center).T)
    ev, evec = np.linalg.eigh(cov)
    R = evec[:, np.argsort(-ev)]
    if np.linalg.det(R) < 0: R[:, 2] *= -1
    if R_ref[0] is not None:
        for i in range(3):
            if np.dot(R[:, i], R_ref[0][:, i]) < 0: R[:, i] *= -1
    R_ref[0] = R.copy()
    return center, R

def nose_scale(P):
    pts = P[NOSE]; d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    iu = np.triu_indices(len(pts), 1)
    return dist[iu].mean() if len(iu[0]) else 1.0

def euc(pred, tgt):
    dd = pred - tgt
    return np.hypot(dd[:, 0] * SW, dd[:, 1] * SH)

def fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    pred = np.zeros((len(Xte), 2))
    for a in range(2):
        pred[:, a] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(Xtr2, Ytr[:, a]).predict(Xte2)
    return pred

BINS = [(0,10),(10,20),(20,30),(30,90)]
rng = np.random.RandomState(0)
log("\n---\n## exp17: 眼球モデル視線方向のオフライン検証（honest, 姿勢bin別）")
eye_bin = {b: [] for b in BINS}; d7_bin = {b: [] for b in BINS}
n_sess = 0
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 40: continue
    # --- パス1: 時系列で頭部座標系・虹彩3D・yaw・7D特徴を収集(R符号を安定化) ---
    R_ref = [None]; recs = []
    for k in idx:  # idxは昇順=frame順=時系列
        t = d["target"][k]
        if np.isnan(t).any(): continue
        w, h = float(d["img_w"][k]), float(d["img_h"][k])
        lm = d["landmarks"][k]
        P = lm * np.array([w, h, w])
        center, R = head_frame(P, R_ref)
        iris_l = P[L_IRIS]; iris_r = P[R_IRIS]
        r7 = build_7d_features([LM(p) for p in lm], int(w), int(h))
        if r7 is None: continue
        feat7, yaw, _, _ = r7
        recs.append(dict(center=center, R=R, il=iris_l, ir=iris_r,
                         ns=nose_scale(P), t=np.asarray(t, float),
                         yaw=abs(np.degrees(yaw)), f7=np.asarray(feat7, float)))
    if len(recs) < 40: continue
    # --- 正面点(|yaw|<10)で眼球中心オフセットをロック(平均) ---
    front = [r for r in recs if r["yaw"] < 10]
    if len(front) < 5: continue
    ol, or_, scs = [], [], []
    for r in front:
        cam_local = r["R"].T @ np.array([0, 0, 1.0])
        ol.append(r["R"].T @ (r["il"] - r["center"]) + BASE_R * cam_local)
        or_.append(r["R"].T @ (r["ir"] - r["center"]) + BASE_R * cam_local)
        scs.append(r["ns"])
    off_l = np.mean(ol, 0); off_r = np.mean(or_, 0); calib_ns = np.mean(scs)
    # --- パス2: 各点で眼球中心→視線方向(3D, 左右平均) を特徴に ---
    Xeye, X7, Y, YAW = [], [], [], []
    for r in recs:
        sr = r["ns"] / calib_ns if calib_ns else 1.0
        sph_l = r["center"] + r["R"] @ (off_l * sr)
        sph_r = r["center"] + r["R"] @ (off_r * sr)
        gl = r["il"] - sph_l; gr = r["ir"] - sph_r
        nl, nr = np.linalg.norm(gl), np.linalg.norm(gr)
        if nl < 1e-6 or nr < 1e-6: continue
        gaze = (gl / nl + gr / nr) / 2.0
        Xeye.append(gaze); X7.append(r["f7"]); Y.append(r["t"]); YAW.append(r["yaw"])
    Xeye = np.array(Xeye); X7 = np.array(X7); Y = np.array(Y); YAW = np.array(YAW)
    if len(Y) < 40 or (YAW >= 20).sum() < 5: continue
    # --- honest: target位置でgroup-split ---
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
log(f"\n**眼球モデル vs 7D統計（{n_sess}セッション, honest, median cm）**")
log(f"  {'|yaw|':>8} | {'眼球モデル':>10} | {'7D統計(exp14相当)':>16}")
for (lo, hi) in BINS:
    e = eye_bin[(lo, hi)]; s = d7_bin[(lo, hi)]
    if e and s:
        log(f"  {lo:2d}-{hi:2d}° | {np.median(e):>8.2f}cm | {np.median(s):>14.2f}cm")
alle = [v for b in BINS for v in eye_bin[b]]; alls = [v for b in BINS for v in d7_bin[b]]
if alle:
    log(f"  overall | {np.median(alle):>8.2f}cm | {np.median(alls):>14.2f}cm")
    log("\n### 見方")
    log("- 眼球モデルの横向き(20-30°,30+)が7Dより下がれば、物理モデルが頭部ロバストで有効=方針正しい。")
    log("- 正面(0-10°)が7Dの1.4cm級には届かなくても、姿勢間のブレが小さければ『分けなくていい』の証拠。")
