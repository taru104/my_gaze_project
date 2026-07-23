"""exp20: 眼球モデルのパラメータ(眼球中心の内寄せx_inset・奥行きz_off・半径eye_r)をグリッド最適化。
exp19は推定固定値。ここでキャリブデータに最も合うパラメータを探し、視線方向の質の上限を見る。
高速化: solvePnP(R,t)と虹彩カメラ光線は一度だけ計算しキャッシュ→球交点だけパラメータで再計算。
最良パラメータでの姿勢bin別を7Dと比較。z非依存。honest評価。mainは触らない。
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

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def euc(pred, tgt):
    dd = pred - tgt; return np.hypot(dd[:, 0] * SW, dd[:, 1] * SH)

def fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    pred = np.zeros((len(Xte), 2))
    for a in range(2):
        pred[:, a] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=600).fit(Xtr2, Ytr[:, a]).predict(Xte2)
    return pred

def sphere_hit(ray, eye_cam, r):
    L = -eye_cam; b = 2.0 * ray @ L; c = L @ L - r * r; disc = b * b - 4.0 * c
    tt = -(ray @ L) / (ray @ ray) if disc < 0 else (-b - np.sqrt(disc)) / 2.0
    return tt * ray

# --- パス1: 各セッションの点ごとに R,t,虹彩光線,7D,target,yaw をキャッシュ ---
sessions = []
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 40: continue
    pts = []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        w, h = float(d["img_w"][k]), float(d["img_h"][k]); f = w
        lm = [LM(p) for p in d["landmarks"][k]]
        face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in _FACE_2D_IDX], dtype=np.float64)
        cam = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        R, _ = cv2.Rodrigues(rvec); tv = tvec.ravel()
        def ray(iris_idx):
            ix, iy = lm[iris_idx].x * w, lm[iris_idx].y * h
            r = np.array([(ix - w/2.0)/f, (iy - h/2.0)/f, 1.0]); return r / np.linalg.norm(r)
        r7 = build_7d_features(lm, int(w), int(h))
        if r7 is None: continue
        feat7, yaw, _, _ = r7
        pts.append(dict(R=R, t=tv, ray_l=ray(L_IRIS), ray_r=ray(R_IRIS),
                        f7=np.asarray(feat7, float), tgt=np.asarray(t, float), yaw=abs(np.degrees(yaw))))
    if len(pts) >= 40:
        sessions.append(pts)

def gaze_feats(pts, x_inset, z_off, eye_r):
    eyeL = np.array([-x_inset, 33.0, z_off]); eyeR = np.array([x_inset, 33.0, z_off])
    out = []
    for p in pts:
        ecl = p["R"] @ eyeL + p["t"]; ecr = p["R"] @ eyeR + p["t"]
        il = sphere_hit(p["ray_l"], ecl, eye_r); ir = sphere_hit(p["ray_r"], ecr, eye_r)
        gl = il - ecl; gr = ir - ecr
        nl, nr = np.linalg.norm(gl), np.linalg.norm(gr)
        if nl < 1e-9 or nr < 1e-9: out.append(None); continue
        g = gl/nl + gr/nr; ng = np.linalg.norm(g)
        out.append(g/ng if ng > 1e-9 else None)
    return out

BINS = [(0,10),(10,20),(20,30),(30,90)]
rng = np.random.RandomState(0)
log(f"\n---\n## exp20: 眼球パラメータ グリッド最適化（{len(sessions)}セッション, honest）")

GRID = [(xi, zo, er) for xi in (28.0, 34.0) for zo in (-30.0, -45.0, -60.0) for er in (12.0, 16.0)]
best = None
for (xi, zo, er) in GRID:
    allerr = []
    for pts in sessions:
        gz = gaze_feats(pts, xi, zo, er)
        X, Y, YAW = [], [], []
        for p, g in zip(pts, gz):
            if g is None: continue
            X.append(g); Y.append(p["tgt"]); YAW.append(p["yaw"])
        X = np.array(X); Y = np.array(Y); YAW = np.array(YAW)
        if len(Y) < 40 or (YAW >= 20).sum() < 5: continue
        groups = defaultdict(list)
        for i, t in enumerate(Y): groups[(round(t[0],1), round(t[1],1))].append(i)
        gk = list(groups.keys())
        if len(gk) < 4: continue
        order = rng.permutation(len(gk)); cut = max(2, int(len(gk)*0.7))
        tr = [i for j in order[:cut] for i in groups[gk[j]]]
        te = [i for j in order[cut:] for i in groups[gk[j]]]
        if len(tr) < 20 or len(te) < 8: continue
        allerr += list(euc(fit_predict(X[tr], Y[tr], X[te]), Y[te]))
    if allerr:
        m = float(np.median(allerr))
        log(f"  x_inset={xi:.0f} z_off={zo:.0f} eye_r={er:.0f} → overall {m:.2f}cm")
        if best is None or m < best[0]: best = (m, xi, zo, er)
if best:
    m, xi, zo, er = best
    log(f"\n**最良パラメータ: x_inset={xi:.0f}, z_off={zo:.0f}, eye_r={er:.0f} → overall {m:.2f}cm**")
    log(f"- 参考: exp19(固定値)11.85cm, 7D統計5.71cm。最適化で {m:.2f}cm。")
    log(f"- これでも7Dに届かなければ『回帰特徴としての眼球モデルは筋が悪い』確定→幾何直接計算(画面インチ)へ全振り。")
