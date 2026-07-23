"""exp21: 距離ロバスト性の直接検証（ユーザの狙い「距離ごとの誤差をなくす」の核心）。
キャリブ距離と異なる距離のフレームで、7D統計 vs 眼球モデル(z非依存,最良パラメータ)の
誤差劣化を比較。眼球モデルが距離変化に強ければ、遠い距離binで7Dより崩れにくいはず。
学習=キャリブ距離付近の点、テスト=距離変化binで層別(target-group-splitでhonest)。mainは触らない。
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
XI, ZO, ER = 28.0, -30.0, 16.0  # exp20最良パラメータ

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def sphere_hit(ray, eye_cam, r):
    L = -eye_cam; b = 2.0*ray@L; c = L@L - r*r; disc = b*b - 4.0*c
    tt = -(ray@L)/(ray@ray) if disc < 0 else (-b - np.sqrt(disc))/2.0
    return tt*ray

def euc(pred, tgt):
    dd = pred - tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

def fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    pred = np.zeros((len(Xte), 2))
    for a in range(2):
        pred[:,a] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=600).fit(sc.transform(Xtr), Ytr[:,a]).predict(sc.transform(Xte))
    return pred

# 収集
sessions = []
for binp in sorted(glob.glob(str(ROOT/"logs"/"*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 60: continue
    pts = []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        w, h = float(d["img_w"][k]), float(d["img_h"][k]); f = w
        lm = [LM(p) for p in d["landmarks"][k]]
        face_2d = np.array([[lm[i].x*w, lm[i].y*h] for i in _FACE_2D_IDX], dtype=np.float64)
        cam = np.array([[f,0,w/2.0],[0,f,h/2.0],[0,0,1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        R,_ = cv2.Rodrigues(rvec); tv = tvec.ravel()
        def ray(i):
            ix, iy = lm[i].x*w, lm[i].y*h
            r = np.array([(ix-w/2.0)/f,(iy-h/2.0)/f,1.0]); return r/np.linalg.norm(r)
        ecl = R@np.array([-XI,33.0,ZO]) + tv; ecr = R@np.array([XI,33.0,ZO]) + tv
        il = sphere_hit(ray(L_IRIS), ecl, ER); ir = sphere_hit(ray(R_IRIS), ecr, ER)
        gl = il-ecl; gr = ir-ecr
        nl, nr = np.linalg.norm(gl), np.linalg.norm(gr)
        if nl<1e-9 or nr<1e-9: continue
        gaze = gl/nl + gr/nr; ng = np.linalg.norm(gaze)
        if ng<1e-9: continue
        r7 = build_7d_features(lm, int(w), int(h))
        if r7 is None: continue
        f7, yaw, _, _ = r7
        pts.append(dict(eye=gaze/ng, f7=np.asarray(f7,float), tgt=np.asarray(t,float),
                        yaw=abs(np.degrees(yaw)), dist=float(np.linalg.norm(tv))))
    if len(pts) >= 60: sessions.append(pts)

DBINS = [(0.0,0.05),(0.05,0.12),(0.12,10.0)]  # |Δdist|/calib_dist の相対変化
rng = np.random.RandomState(0)
log(f"\n---\n## exp21: 距離ロバスト性（{len(sessions)}セッション, 学習=キャリブ距離付近, テスト=距離変化bin別）")
eye_b = {b: [] for b in DBINS}; d7_b = {b: [] for b in DBINS}
for pts in sessions:
    dist = np.array([p["dist"] for p in pts])
    front = [p["dist"] for p in pts if p["yaw"] < 12]
    if len(front) < 5: continue
    cd = np.median(front)
    rel = np.abs(dist - cd) / cd
    # target group split(honest)
    groups = defaultdict(list)
    for i, p in enumerate(pts): groups[(round(p["tgt"][0],1), round(p["tgt"][1],1))].append(i)
    gk = list(groups.keys())
    if len(gk) < 4: continue
    order = rng.permutation(len(gk)); cut = max(2, int(len(gk)*0.7))
    trg = set(gk[j] for j in order[:cut]); teg = set(gk[j] for j in order[cut:])
    # 学習=trainグループ かつ キャリブ距離付近(rel<0.05)
    tr = [i for i, p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in trg and rel[i] < 0.05]
    if len(tr) < 20: continue
    Xe = np.array([pts[i]["eye"] for i in tr]); X7 = np.array([pts[i]["f7"] for i in tr])
    Y = np.array([pts[i]["tgt"] for i in tr])
    for (lo, hi) in DBINS:
        te = [i for i, p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg and lo <= rel[i] < hi]
        if len(te) < 6: continue
        Xte_e = np.array([pts[i]["eye"] for i in te]); Xte_7 = np.array([pts[i]["f7"] for i in te])
        Yte = np.array([pts[i]["tgt"] for i in te])
        eye_b[(lo,hi)] += list(euc(fit_predict(Xe, Y, Xte_e), Yte))
        d7_b[(lo,hi)]  += list(euc(fit_predict(X7, Y, Xte_7), Yte))
log(f"\n**距離変化bin別（学習はキャリブ距離±5%）, median cm**")
log(f"  {'距離変化':>12} | {'眼球モデル':>10} | {'7D統計':>10} | 差")
names = {DBINS[0]:"±5%以内(近)", DBINS[1]:"5-12%(中)", DBINS[2]:">12%(遠)"}
for b in DBINS:
    e = eye_b[b]; s = d7_b[b]
    if e and s:
        me, ms = np.median(e), np.median(s)
        mark = " ★眼球勝ち" if me < ms else ""
        log(f"  {names[b]:>12} | {me:>8.2f}cm | {ms:>8.2f}cm | {ms-me:+.2f}{mark}")
log("- 見方: 距離変化が大きいbin(遠)で、7Dが大きく劣化し眼球が保てば、距離ロバスト=ユーザの狙い達成の証拠。")
