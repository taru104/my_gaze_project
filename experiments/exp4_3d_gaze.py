"""7時間耐久 実験4: 3D視線ベクトル幾何は姿勢ロバストか。
MPIIの3D gaze方向GT(pxx.txt: gt[Dim25-27] - fc[Dim22-24])を使い、7D特徴→gaze角(pitch/yaw)を
person-independentで回帰。angular error(度)を頭部姿勢bin別に見る。
3D gaze角が頭部姿勢に対してロバスト(横向きでも角度誤差が増えない)なら、画面座標直接回帰(H1)より
姿勢ロバストな道である証拠。--limit で各人上限。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import cv2, scipy.io as sio
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
CKG = ROOT / "cache" / "mpii_gaze_ck.npz"
LIMIT = 800
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

def gaze_to_angle(v):
    v = v / (np.linalg.norm(v) + 1e-9)
    pitch = np.arcsin(-v[1]); yaw = np.arctan2(-v[0], -v[2])
    return np.degrees([pitch, yaw])

def extract():
    if CKG.exists():
        d = np.load(CKG); return d["X"], d["G"], d["pid"]
    opts = mp_vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                           running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
    lm = mp_vision.FaceLandmarker.create_from_options(opts)
    X, G, PID = [], [], []
    for pi in range(15):
        pid = f"p{pi:02d}"
        lines = open(MPII/pid/f"{pid}.txt").read().strip().splitlines()[:LIMIT]
        ok = 0
        for ln in lines:
            f = ln.split()
            if len(f) < 27: continue
            fc = np.array([float(f[21]), float(f[22]), float(f[23])])
            gt = np.array([float(f[24]), float(f[25]), float(f[26])])
            ga = gaze_to_angle(gt - fc)
            img = cv2.imread(str(MPII/pid/f[0]))
            if img is None: continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks: continue
            feat = rich_16d_from_lms(lms_to_array(res.face_landmarks[0]), w, h)
            if feat is None or not np.isfinite(feat).all(): continue
            X.append(feat[:7]); G.append(ga); PID.append(pid); ok += 1
        print(f"{pid}: {ok}", flush=True)
    X, G, PID = np.array(X,np.float32), np.array(G,np.float32), np.array(PID)
    np.savez_compressed(CKG, X=X, G=G, pid=PID)
    return X, G, PID

t0 = time.time()
log(f"\n---\n## 実験4: 3D視線ベクトル幾何(MPII gaze角, person-independent)")
X, G, PID = extract()
log(f"抽出 {len(X)}フレーム/15人 (各人上限{LIMIT})")
BINS = [(0,10),(10,20),(20,30),(30,90)]
persons = sorted(set(PID.tolist()))
P, GG, HY = [], [], []
for p in persons:
    te, tr = PID==p, PID!=p
    sc = StandardScaler().fit(X[tr])
    mp_ = HuberRegressor(max_iter=800).fit(sc.transform(X[tr]), G[tr,0])
    my_ = HuberRegressor(max_iter=800).fit(sc.transform(X[tr]), G[tr,1])
    pr = np.column_stack([mp_.predict(sc.transform(X[te])), my_.predict(sc.transform(X[te]))])
    P.append(pr); GG.append(G[te]); HY.append(np.abs(np.degrees(X[te,5])))  # 頭部yaw
P, GG, HY = np.vstack(P), np.vstack(GG), np.concatenate(HY)
ang = np.linalg.norm(P - GG, axis=1)  # angular error(度)
log(f"### person-independent gaze角 angular error(度) 頭部yaw帯別")
log(f"  全体 median={np.median(ang):.2f}°  (参考: 50cmで1度≈0.87cm)")
for lo,hi in BINS:
    m = (HY>=lo)&(HY<hi)
    if m.sum(): log(f"  頭部yaw {lo:2d}-{hi:2d}: median={np.median(ang[m]):.2f}° (n={int(m.sum())})")
log(f"→ 頭部yawが増えても角度誤差が増えなければ3D gaze角は姿勢ロバスト。cm換算は距離依存だが")
log(f"  1度≈0.87cm(50cm)なので、全体{np.median(ang):.1f}°≈{np.median(ang)*0.87:.1f}cm相当。")
log(f"\n(実験4 完了 {(time.time()-t0)/60:.1f}分)")
