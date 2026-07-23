"""exp26: 「しっかりキャリブしたら何cmか」＝実機の上限を測る。目標3cmの現実性を見る。
exp21-25は学習をキャリブ距離±5%に絞る超厳しい条件だった(外挿)。ここでは学習に全距離・全姿勢を入れ
(多点キャリブ)、テストは未知の画面位置のみ(target-group-split)。姿勢・距離は学習範囲内=補間。
7D / 眼球 / ブレンド を全体＋姿勢bin別に。デバイス非依存。mainは触らない。
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
XI, ZO, ER = 28.0, -30.0, 16.0

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def sphere_hit(ray, ec, r):
    L=-ec; b=2.0*ray@L; c=L@L-r*r; disc=b*b-4.0*c
    tt=-(ray@L)/(ray@ray) if disc<0 else (-b-np.sqrt(disc))/2.0
    return tt*ray

def euc(pred, tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

def fit_model(Xtr, Ytr):
    sc=StandardScaler().fit(Xtr)
    ms=[HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=600).fit(sc.transform(Xtr), Ytr[:,a]) for a in range(2)]
    return sc, ms

def predict(model, Xte):
    sc, ms=model; return np.stack([ms[a].predict(sc.transform(Xte)) for a in range(2)], axis=1)

sessions=[]
for binp in sorted(glob.glob(str(ROOT/"logs"/"*_landmarks.bin"))):
    try: d=load_raw_landmarks(binp)
    except Exception: continue
    idx=np.where(d["has_target"])[0]
    if len(idx)<60: continue
    pts=[]
    for k in idx:
        t=d["target"][k]
        if np.isnan(t).any(): continue
        w,h=float(d["img_w"][k]),float(d["img_h"][k]); f=w
        lm=[LM(p) for p in d["landmarks"][k]]
        face_2d=np.array([[lm[i].x*w, lm[i].y*h] for i in _FACE_2D_IDX], dtype=np.float64)
        cam=np.array([[f,0,w/2.0],[0,f,h/2.0],[0,0,1.0]], dtype=np.float64)
        ok,rvec,tvec=cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        R,_=cv2.Rodrigues(rvec); tv=tvec.ravel()
        def ray(i):
            ix,iy=lm[i].x*w, lm[i].y*h
            r=np.array([(ix-w/2.0)/f,(iy-h/2.0)/f,1.0]); return r/np.linalg.norm(r)
        ecl=R@np.array([-XI,33.0,ZO])+tv; ecr=R@np.array([XI,33.0,ZO])+tv
        il=sphere_hit(ray(L_IRIS),ecl,ER); ir=sphere_hit(ray(R_IRIS),ecr,ER)
        gl=il-ecl; gr=ir-ecr; nl,nr=np.linalg.norm(gl),np.linalg.norm(gr)
        if nl<1e-9 or nr<1e-9: continue
        gz=gl/nl+gr/nr; ng=np.linalg.norm(gz)
        if ng<1e-9: continue
        r7=build_7d_features(lm,int(w),int(h))
        if r7 is None: continue
        f7,yaw,_,_=r7
        pts.append(dict(f7=np.asarray(f7,float), eye=gz/ng, tgt=np.asarray(t,float),
                        dist=float(np.linalg.norm(tv)), yaw=abs(np.degrees(yaw))))
    if len(pts)>=60: sessions.append(pts)

YBINS=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log(f"\n---\n## exp26: しっかりキャリブ(全距離・全姿勢を学習, 未知位置のみtest)（{len(sessions)}セッション, 目標3cm）")
yb={b:{"7d":[],"eye":[],"blend":[]} for b in YBINS}
allr={"7d":[],"eye":[],"blend":[]}
for pts in sessions:
    dist=np.array([p["dist"] for p in pts]); front=[p["dist"] for p in pts if p["yaw"]<12]
    cd=np.median(front) if front else np.median(dist); rel=np.abs(dist-cd)/cd
    groups=defaultdict(list)
    for i,p in enumerate(pts): groups[(round(p["tgt"][0],1),round(p["tgt"][1],1))].append(i)
    gk=list(groups.keys())
    if len(gk)<5: continue
    order=rng.permutation(len(gk)); cut=max(3,int(len(gk)*0.7))
    trg=set(gk[j] for j in order[:cut]); teg=set(gk[j] for j in order[cut:])
    tr=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in trg]  # 全距離・全姿勢
    te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg]
    if len(tr)<30 or len(te)<10: continue
    Y=np.array([pts[i]["tgt"] for i in tr])
    m7=fit_model(np.array([pts[i]["f7"] for i in tr]),Y); me=fit_model(np.array([pts[i]["eye"] for i in tr]),Y)
    Yte=np.array([pts[i]["tgt"] for i in te])
    p7=predict(m7,np.array([pts[i]["f7"] for i in te])); pe=predict(me,np.array([pts[i]["eye"] for i in te]))
    w=np.clip((rel[te]-0.05)/0.10,0.0,1.0)[:,None]; pb=(1-w)*p7+w*pe
    e7=euc(p7,Yte); ee=euc(pe,Yte); eb=euc(pb,Yte)
    allr["7d"]+=list(e7); allr["eye"]+=list(ee); allr["blend"]+=list(eb)
    for j,i in enumerate(te):
        for (lo,hi) in YBINS:
            if lo<=pts[i]["yaw"]<hi:
                yb[(lo,hi)]["7d"].append(e7[j]); yb[(lo,hi)]["eye"].append(ee[j]); yb[(lo,hi)]["blend"].append(eb[j]); break
log(f"\n**姿勢bin別 median cm（多点キャリブ=補間, 未知位置のみtest）**")
log(f"  {'|yaw|':>8} | {'7D':>6} | {'眼球':>6} | {'ブレンド':>7}")
for b in YBINS:
    r=yb[b]
    if r["7d"]:
        log(f"  {b[0]:2d}-{b[1]:2d}° | {np.median(r['7d']):>4.2f} | {np.median(r['eye']):>4.2f} | {np.median(r['blend']):>5.2f}")
log(f"\n- 全体: 7D={np.median(allr['7d']):.2f} / 眼球={np.median(allr['eye']):.2f} / ブレンド={np.median(allr['blend']):.2f}cm")
log(f"- exp22(近距離のみ学習=外挿)ブレンド7.18cm → exp26(全条件学習=補間)。しっかりキャリブで3cmに近づくか。")
