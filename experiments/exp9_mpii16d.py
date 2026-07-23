"""追加実験(ユーザ指摘対応): MPIIを16Dで抽出し15人で 7D vs 16D をperson-independent検証。
自分1人の「16Dが良い」結論を、15人の公開データセットで裏付ける/覆す。
person-independent(他人でキャリブ無し) + 個人内(その人のデータでLOO) の両方で 7D vs 16D 姿勢bin別。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, cv2, scipy.io as sio
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
sys.path.insert(0, str(Path(__file__).parent.parent))
from rich16d import rich_16d_from_lms, lms_to_array
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
MPII = ROOT / "MPIIFaceGaze"; MODEL = ROOT / "face_landmarker.task"
REPORT = ROOT / "experiments" / "REPORT.md"
CK = ROOT / "cache" / "mpii_16d_ck.npz"
CM = np.array([30.0, 19.0]); BINS = [(0,10),(10,20),(20,30),(30,90)]; LIMIT = 800
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

def extract():
    if CK.exists():
        d = np.load(CK); return d["X"], d["y"], d["pid"]
    opts = mp_vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                           running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
    lm = mp_vision.FaceLandmarker.create_from_options(opts)
    X, Y, PID = [], [], []
    for pi in range(15):
        pid = f"p{pi:02d}"
        ss = sio.loadmat(str(MPII/pid/"Calibration"/"screenSize.mat"))
        wpx, hpx = float(ss["width_pixel"][0][0]), float(ss["height_pixel"][0][0])
        lines = open(MPII/pid/f"{pid}.txt").read().strip().splitlines()[:LIMIT]
        ok = 0
        for ln in lines:
            f = ln.split()
            img = cv2.imread(str(MPII/pid/f[0]))
            if img is None: continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks: continue
            feat = rich_16d_from_lms(lms_to_array(res.face_landmarks[0]), w, h)
            if feat is None or not np.isfinite(feat).all(): continue
            X.append(feat)  # 16D全部
            Y.append([float(f[1])/wpx, float(f[2])/hpx]); PID.append(pid); ok += 1
        print(f"{pid}: {ok}", flush=True)
    X, Y, PID = np.array(X,np.float32), np.array(Y,np.float32), np.array(PID)
    np.savez_compressed(CK, X=X, y=Y, pid=PID)
    return X, Y, PID

def fit_pred(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); a,b = sc.transform(Xtr), sc.transform(Xte)
    mx = HuberRegressor(max_iter=800).fit(a, Ytr[:,0]); my = HuberRegressor(max_iter=800).fit(a, Ytr[:,1])
    return np.column_stack([mx.predict(b), my.predict(b)])

t0 = time.time()
log(f"\n---\n## 実験9(データセット検証): MPII 15人で 7D vs 16D")
X16, Y, PID = extract()
X7 = X16[:, :7]
log(f"MPII 16D抽出 {len(X16)}フレーム/15人 (各人上限{LIMIT})")
persons = sorted(set(PID.tolist()))

# (1) person-independent (他人・キャリブ無し)
log(f"### (1) person-independent (他人・キャリブ無し) 姿勢bin別 median Euc(cm近似)  全体|0-10 10-20 20-30 30+")
for feat, Xf in [("7D ", X7), ("16D", X16)]:
    P,G,YD = [],[],[]
    for p in persons:
        te, tr = PID==p, PID!=p
        P.append(fit_pred(Xf[tr], Y[tr], Xf[te])); G.append(Y[te]); YD.append(np.abs(np.degrees(X16[te,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    log(f"  {feat}: {np.median(e):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(e, YD)))

# (2) 個人内 (その人のデータで5-fold) = 個人キャリブ相当
from sklearn.model_selection import KFold
log(f"### (2) 個人内(その人データで5-fold=個人キャリブ相当) 姿勢bin別")
for feat, Xf in [("7D ", X7), ("16D", X16)]:
    P,G,YD = [],[],[]
    for p in persons:
        idx = np.where(PID==p)[0]
        if len(idx) < 50: continue
        Xp, Yp, Hp = Xf[idx], Y[idx], np.abs(np.degrees(X16[idx,5]))
        kf = KFold(5, shuffle=True, random_state=0); pr = np.zeros_like(Yp)
        for tr,te in kf.split(Xp): pr[te] = fit_pred(Xp[tr], Yp[tr], Xp[te])
        P.append(pr); G.append(Yp); YD.append(Hp)
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    log(f"  {feat}: {np.median(e):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(e, YD)))
log(f"→ 15人の公開データで 16D>7D(特に横向き)が再現すれば、自分1人の結論が汎用的と裏付け。")
log(f"\n(実験9 完了 {(time.time()-t0)/60:.1f}分)")
