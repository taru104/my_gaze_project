"""Step0: リンバス実験の事前2測定。
0-1: MPIIの虹彩は何px写っているか(中央値・レンジ)。アプリは640x480@54cmで約14px。
0-2: exp42のアスペクト比アブレーションを|yaw|bin別に再計算(形の感度はsin視線角∝→横向きでのみ効くはず)。"""
import sys, glob, random
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from rich16d import lms_to_array
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

REPORT = ROOT / "experiments" / "REPORT7_limbus.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

MPII = ROOT / "MPIIFaceGaze"; MODEL = ROOT / "face_landmarker.task"
L_IRIS=[468,469,470,471,472]; R_IRIS=[473,474,475,476,477]

# ---------- 0-1: 虹彩px ----------
def make_lm():
    return mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=mp_vision.RunningMode.IMAGE, num_faces=1))
lm = make_lm()
diams=[]
rng = random.Random(0)
for pid in ['p00','p03','p06','p09','p12']:
    lines = open(MPII/pid/f"{pid}.txt").read().strip().splitlines()
    rng.shuffle(lines)
    for ln in lines[:40]:
        imgp = ln.split()[0]
        img = cv2.imread(str(MPII/pid/imgp))
        if img is None: continue
        h,w = img.shape[:2]
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
        if not res.face_landmarks: continue
        a = lms_to_array(res.face_landmarks[0])
        for IR in (L_IRIS, R_IRIS):
            c = np.array([a[IR[0],0]*w, a[IR[0],1]*h])
            d = 2.0*np.mean([np.hypot(a[i,0]*w-c[0], a[i,1]*h-c[1]) for i in IR[1:]])
            if d>1: diams.append(d)
diams=np.array(diams)
log("\n# REPORT7 — リンバス楕円フィット実験(MPII)")
log("\n## Step0-1: MPIIの虹彩サイズ(px)")
log(f"  虹彩直径 中央値={np.median(diams):.1f}px  レンジ={np.percentile(diams,10):.0f}〜{np.percentile(diams,90):.0f}px (10-90%tile)  n={len(diams)}")
log(f"  参考: アプリ実機は640x480@54cmで約14px。MPIIが大きいほど『MPIIで効いても実機で効くとは限らない』")

# ---------- 0-2: アスペクト比アブレーション by |yaw| bin ----------
d = np.load(str(ROOT/"cache"/"mpii_16d_ck.npz"))
X,y,pid = d["X"],d["y"],d["pid"]
yaw_deg = np.degrees(np.abs(X[:,5]))
ASP=[14,15]  # L_asp,R_asp
def fp(Xtr,Ytr,Xte):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for i in range(2): pr[:,i]=HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=600).fit(A,Ytr[:,i]).predict(B)
    return pr
def euc(p,g): return np.median(np.sqrt(np.sum((p-g)**2,axis=1)))*30.0
rng2=np.random.RandomState(0)
log("\n## Step0-2: アスペクト比アブレーション |yaw|bin別 (Δ=抜くと誤差増→重要 / 負=冗長)")
log(f"  {'bin':>8} | {'n':>6} | {'16D(cm)':>8} | {'asp抜き':>8} | {'Δ(重要度)':>10}")
BINS=[(0,10),(10,20),(20,30),(30,90)]
for lo,hi in BINS:
    m=(yaw_deg>=lo)&(yaw_deg<hi)
    if m.sum()<200:
        log(f"  {str(lo)+'-'+str(hi):>8} | {m.sum():>6} | (サンプル不足)"); continue
    Xb,yb=X[m],y[m]
    idx=rng2.permutation(len(Xb)); fold=np.array_split(idx,5)
    e_full,e_noasp=[],[]
    keep_no=[i for i in range(16) if i not in ASP]
    for kf in range(5):
        te=fold[kf]; tr=np.concatenate([fold[j] for j in range(5) if j!=kf])
        e_full.append(euc(fp(Xb[tr],yb[tr],Xb[te]),yb[te]))
        e_noasp.append(euc(fp(Xb[np.ix_(tr,keep_no)],yb[tr],Xb[np.ix_(te,keep_no)]),yb[te]))
    ef,en=np.median(e_full),np.median(e_noasp)
    tag='重要(横向きで効く)' if en-ef>0.03 else ('冗長' if en-ef<-0.03 else '中立')
    log(f"  {str(lo)+'-'+str(hi):>8} | {m.sum():>6} | {ef:>7.3f} | {en:>7.3f} | {en-ef:>+9.3f} {tag}")
log("  → 30+でΔ正(抜くと悪化)なら形(B腕)の期待値↑。全binで冗長ならB薄いと予想し進める(実験は続行)。")
