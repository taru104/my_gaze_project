"""exp28: 特徴の情報量を増やして5.71cmの壁を破れるか。7D vs 16D(rich) vs 16D+眼球3D(19D)。
線形Huber・多点キャリブ・honest(未知画面位置)・姿勢bin別。GPU不要。デバイス非依存。mainは触らない。
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
from rich16d import rich_16d_from_lms
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
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

def eye_dir(lm, w, h):
    face_2d=np.array([[lm[i].x*w, lm[i].y*h] for i in _FACE_2D_IDX], dtype=np.float64)
    f=float(w); cam=np.array([[f,0,w/2.0],[0,f,h/2.0],[0,0,1.0]], dtype=np.float64)
    ok,rvec,tvec=cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    R,_=cv2.Rodrigues(rvec); tv=tvec.ravel()
    def ray(i):
        ix,iy=lm[i].x*w, lm[i].y*h
        r=np.array([(ix-w/2.0)/f,(iy-h/2.0)/f,1.0]); return r/np.linalg.norm(r)
    ecl=R@np.array([-XI,33.0,ZO])+tv; ecr=R@np.array([XI,33.0,ZO])+tv
    il=sphere_hit(ray(L_IRIS),ecl,ER); ir=sphere_hit(ray(R_IRIS),ecr,ER)
    gl=il-ecl; gr=ir-ecr; nl,nr=np.linalg.norm(gl),np.linalg.norm(gr)
    if nl<1e-9 or nr<1e-9: return None
    g=gl/nl+gr/nr; ng=np.linalg.norm(g)
    return g/ng if ng>1e-9 else None

def euc(pred, tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

def fit_predict(Xtr, Ytr, Xte):
    sc=StandardScaler().fit(Xtr); Xtr2,Xte2=sc.transform(Xtr),sc.transform(Xte)
    pred=np.zeros((len(Xte),2))
    for a in range(2):
        pred[:,a]=HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=600).fit(Xtr2, Ytr[:,a]).predict(Xte2)
    return pred

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
        w,h=float(d["img_w"][k]),float(d["img_h"][k])
        lmarr=d["landmarks"][k]
        try: f16=rich_16d_from_lms(lmarr, int(w), int(h))
        except Exception: f16=None
        if f16 is None: continue
        f16=np.asarray(f16,float)
        lm=[LM(p) for p in lmarr]
        eye=eye_dir(lm,w,h)
        if eye is None: continue
        yaw=abs(np.degrees(float(f16[5])))
        pts.append(dict(f16=f16, eye=eye, tgt=np.asarray(t,float), yaw=yaw))
    if len(pts)>=60: sessions.append(pts)

def getX(pts, ids, kind):
    if kind=="7D":  return np.array([pts[i]["f16"][:7] for i in ids])
    if kind=="16D": return np.array([pts[i]["f16"] for i in ids])
    if kind=="19D": return np.array([np.concatenate([pts[i]["f16"], pts[i]["eye"]]) for i in ids])

KINDS=["7D","16D","19D"]
YBINS=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log(f"\n---\n## exp28: 特徴拡張 7D/16D/16D+眼球（多点キャリブ, honest, {len(sessions)}セッション）")
overall={k:[] for k in KINDS}; ybin={k:{b:[] for b in YBINS} for k in KINDS}
for pts in sessions:
    groups=defaultdict(list)
    for i,p in enumerate(pts): groups[(round(p["tgt"][0],1),round(p["tgt"][1],1))].append(i)
    gk=list(groups.keys())
    if len(gk)<5: continue
    order=rng.permutation(len(gk)); cut=max(3,int(len(gk)*0.7))
    trg=set(gk[j] for j in order[:cut]); teg=set(gk[j] for j in order[cut:])
    tr=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in trg]
    te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg]
    if len(tr)<30 or len(te)<10: continue
    Ytr=np.array([pts[i]["tgt"] for i in tr]); Yte=np.array([pts[i]["tgt"] for i in te])
    yaws=[pts[i]["yaw"] for i in te]
    for k in KINDS:
        e=euc(fit_predict(getX(pts,tr,k), Ytr, getX(pts,te,k)), Yte)
        overall[k]+=list(e)
        for j in range(len(te)):
            for (lo,hi) in YBINS:
                if lo<=yaws[j]<hi: ybin[k][(lo,hi)].append(e[j]); break
log(f"\n**特徴別 median cm（honest, 多点キャリブ, 線形Huber）**")
log(f"  {'特徴':>6} | {'全体':>6} | {'0-10':>5} | {'10-20':>5} | {'20-30':>5} | {'30+':>5}")
best=None
for k in KINDS:
    o=np.median(overall[k]) if overall[k] else float('nan')
    cells=[f"{np.median(ybin[k][b]):.2f}" if ybin[k][b] else "--" for b in YBINS]
    log(f"  {k:>6} | {o:>5.2f} | {cells[0]:>5} | {cells[1]:>5} | {cells[2]:>5} | {cells[3]:>5}")
    if not np.isnan(o) and (best is None or o<best[1]): best=(k,o)
if best:
    log(f"\n- 最良: {best[0]}={best[1]:.2f}cm（7Dの5.71cm比）。特徴の情報量で壁を破れるか。次は虹彩楕円/両眼輻輳/局所回帰。")
