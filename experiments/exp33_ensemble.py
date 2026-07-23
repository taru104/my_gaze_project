"""exp33: 予測アンサンブル。7D/16D/眼球の各予測を重み付き平均し、16D単独(4.34cm)を下げるか。
結合特徴(exp24)は過学習で失敗したが、予測レベルの平均は安定。均等平均と16D重視を試す。honest。mainは触らない。
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
from features import _FACE_3D_MODEL, _FACE_2D_IDX, _DIST_COEFFS
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
    __slots__=("x","y","z")
    def __init__(s,a): s.x=float(a[0]); s.y=float(a[1]); s.z=float(a[2])
def sphere_hit(ray,ec,r):
    L=-ec; b=2.0*ray@L; c=L@L-r*r; disc=b*b-4.0*c
    tt=-(ray@L)/(ray@ray) if disc<0 else (-b-np.sqrt(disc))/2.0
    return tt*ray
def eye_dir(lm,w,h):
    face_2d=np.array([[lm[i].x*w,lm[i].y*h] for i in _FACE_2D_IDX],dtype=np.float64)
    f=float(w); cam=np.array([[f,0,w/2.0],[0,f,h/2.0],[0,0,1.0]],dtype=np.float64)
    ok,rvec,tvec=cv2.solvePnP(_FACE_3D_MODEL,face_2d,cam,_DIST_COEFFS,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    R,_=cv2.Rodrigues(rvec); tv=tvec.ravel()
    def ray(i):
        ix,iy=lm[i].x*w,lm[i].y*h
        r=np.array([(ix-w/2.0)/f,(iy-h/2.0)/f,1.0]); return r/np.linalg.norm(r)
    ecl=R@np.array([-XI,33.0,ZO])+tv; ecr=R@np.array([XI,33.0,ZO])+tv
    il=sphere_hit(ray(L_IRIS),ecl,ER); ir=sphere_hit(ray(R_IRIS),ecr,ER)
    gl=il-ecl; gr=ir-ecr; nl,nr=np.linalg.norm(gl),np.linalg.norm(gr)
    if nl<1e-9 or nr<1e-9: return None
    g=gl/nl+gr/nr; ng=np.linalg.norm(g)
    return g/ng if ng>1e-9 else None
def euc(pred,tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)
def fit_model(Xtr,Ytr):
    sc=StandardScaler().fit(Xtr); ms=[HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=500).fit(sc.transform(Xtr),Ytr[:,i]) for i in range(2)]
    return sc,ms
def predict(m,Xte):
    sc,ms=m; return np.stack([ms[i].predict(sc.transform(Xte)) for i in range(2)],axis=1)

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
        arr=d["landmarks"][k]
        try: f16=rich_16d_from_lms(arr,int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        lm=[LM(p) for p in arr]; eye=eye_dir(lm,w,h)
        if eye is None: continue
        pts.append(dict(f16=np.asarray(f16,float), eye=eye, tgt=np.asarray(t,float)))
    if len(pts)>=60: sessions.append(pts)

rng=np.random.RandomState(0)
log(f"\n---\n## exp33: 予測アンサンブル（{len(sessions)}セッション, honest）")
res={"16D単独":[], "16D+7D平均":[], "16D+7D+眼球":[], "16D重み0.6":[]}
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
    Y=np.array([pts[i]["tgt"] for i in tr]); Yte=np.array([pts[i]["tgt"] for i in te])
    X16tr=np.array([pts[i]["f16"] for i in tr]); X7tr=X16tr[:,:7]; Xetr=np.array([pts[i]["eye"] for i in tr])
    X16te=np.array([pts[i]["f16"] for i in te]); X7te=X16te[:,:7]; Xete=np.array([pts[i]["eye"] for i in te])
    p16=predict(fit_model(X16tr,Y),X16te); p7=predict(fit_model(X7tr,Y),X7te); pe=predict(fit_model(Xetr,Y),Xete)
    res["16D単独"]+=list(euc(p16,Yte))
    res["16D+7D平均"]+=list(euc((p16+p7)/2,Yte))
    res["16D+7D+眼球"]+=list(euc((p16+p7+pe)/3,Yte))
    res["16D重み0.6"]+=list(euc(0.6*p16+0.25*p7+0.15*pe,Yte))
log(f"\n**アンサンブル別 median cm（honest）**")
best=None
for k,v in res.items():
    if v:
        m=np.median(v); log(f"  {k:>14} | {m:.2f}cm")
        if best is None or m<best[1]: best=(k,m)
if best: log(f"\n- 最良: {best[0]}={best[1]:.2f}cm（16D単独4.34cm比）。予測平均で分散低減し3cmに近づくか。")
