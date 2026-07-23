"""exp32: 最終構成候補。16D(現状ベスト特徴)×眼球モデル(距離ロバスト)を距離ブレンド。
16Dの近距離精度＋眼球の距離ロバストを統合。距離bin別にhonest評価。目標3cm。GPU不要。mainは触らない。
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
    if not ok: return None,None
    R,_=cv2.Rodrigues(rvec); tv=tvec.ravel()
    def ray(i):
        ix,iy=lm[i].x*w,lm[i].y*h
        r=np.array([(ix-w/2.0)/f,(iy-h/2.0)/f,1.0]); return r/np.linalg.norm(r)
    ecl=R@np.array([-XI,33.0,ZO])+tv; ecr=R@np.array([XI,33.0,ZO])+tv
    il=sphere_hit(ray(L_IRIS),ecl,ER); ir=sphere_hit(ray(R_IRIS),ecr,ER)
    gl=il-ecl; gr=ir-ecr; nl,nr=np.linalg.norm(gl),np.linalg.norm(gr)
    if nl<1e-9 or nr<1e-9: return None,None
    g=gl/nl+gr/nr; ng=np.linalg.norm(g)
    return (g/ng if ng>1e-9 else None), float(np.linalg.norm(tv))

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
        lm=[LM(p) for p in arr]
        eye,dist=eye_dir(lm,w,h)
        if eye is None: continue
        pts.append(dict(f16=np.asarray(f16,float), eye=eye, tgt=np.asarray(t,float),
                        dist=dist, yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

DBINS=[(0.0,0.05),(0.05,0.12),(0.12,10.0)]
names={DBINS[0]:"±5%(近)",DBINS[1]:"5-12%(中)",DBINS[2]:">12%(遠)"}
rng=np.random.RandomState(0)
log(f"\n---\n## exp32: 最終構成 16D×眼球 距離ブレンド（{len(sessions)}セッション, honest）")
res={b:{"16D":[],"eye":[],"blend":[]} for b in DBINS}
for pts in sessions:
    dist=np.array([p["dist"] for p in pts]); front=[p["dist"] for p in pts if p["yaw"]<12]
    if len(front)<5: continue
    cd=np.median(front); rel=np.abs(dist-cd)/cd
    groups=defaultdict(list)
    for i,p in enumerate(pts): groups[(round(p["tgt"][0],1),round(p["tgt"][1],1))].append(i)
    gk=list(groups.keys())
    if len(gk)<5: continue
    order=rng.permutation(len(gk)); cut=max(3,int(len(gk)*0.7))
    trg=set(gk[j] for j in order[:cut]); teg=set(gk[j] for j in order[cut:])
    tr=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in trg]
    if len(tr)<30: continue
    Y=np.array([pts[i]["tgt"] for i in tr])
    m16=fit_model(np.array([pts[i]["f16"] for i in tr]),Y); me=fit_model(np.array([pts[i]["eye"] for i in tr]),Y)
    for b in DBINS:
        lo,hi=b
        te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg and lo<=rel[i]<hi]
        if len(te)<6: continue
        Yte=np.array([pts[i]["tgt"] for i in te])
        p16=predict(m16,np.array([pts[i]["f16"] for i in te])); pe=predict(me,np.array([pts[i]["eye"] for i in te]))
        w=np.clip((rel[te]-0.05)/0.10,0.0,1.0)[:,None]; pb=(1-w)*p16+w*pe
        res[b]["16D"]+=list(euc(p16,Yte)); res[b]["eye"]+=list(euc(pe,Yte)); res[b]["blend"]+=list(euc(pb,Yte))
log(f"\n**距離bin別 median cm（16D×眼球ブレンド）**")
log(f"  {'距離変化':>10} | {'16D':>6} | {'眼球':>6} | {'ブレンド':>7}")
for b in DBINS:
    r=res[b]
    if r["16D"] and r["blend"]:
        log(f"  {names[b]:>10} | {np.median(r['16D']):>4.2f} | {np.median(r['eye']):>4.2f} | {np.median(r['blend']):>5.2f}")
a16=[v for b in DBINS for v in res[b]["16D"]]; ab=[v for b in DBINS for v in res[b]["blend"]]
if ab: log(f"\n- 全体: 16D={np.median(a16):.2f} / ブレンド={np.median(ab):.2f}cm。7D×眼球(exp22)7.18→16D×眼球で改善したか。")
