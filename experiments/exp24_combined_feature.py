"""exp24: 質を上げる。7D幾何特徴 + 眼球モデル視線方向(3D) を結合(10D)して1回帰。
別々に予測してブレンド(exp22)するより、両方の情報を1度に使えるか。デバイス非依存(正規化target)。
距離bin別に 7D / ブレンド / 結合10D を比較。目標3cm級に近づけるか。honest評価。mainは触らない。
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

DBINS=[(0.0,0.05),(0.05,0.12),(0.12,10.0)]
names={DBINS[0]:"±5%(近)",DBINS[1]:"5-12%(中)",DBINS[2]:">12%(遠)"}
rng=np.random.RandomState(0)
log(f"\n---\n## exp24: 7D+眼球 結合特徴10D（{len(sessions)}セッション, honest, 目標3cm）")
res={b:{"7d":[],"blend":[],"comb":[]} for b in DBINS}
for pts in sessions:
    dist=np.array([p["dist"] for p in pts])
    front=[p["dist"] for p in pts if p["yaw"]<12]
    if len(front)<5: continue
    cd=np.median(front); rel=np.abs(dist-cd)/cd
    groups=defaultdict(list)
    for i,p in enumerate(pts): groups[(round(p["tgt"][0],1),round(p["tgt"][1],1))].append(i)
    gk=list(groups.keys())
    if len(gk)<4: continue
    order=rng.permutation(len(gk)); cut=max(2,int(len(gk)*0.7))
    trg=set(gk[j] for j in order[:cut]); teg=set(gk[j] for j in order[cut:])
    tr=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in trg and rel[i]<0.05]
    if len(tr)<20: continue
    Y=np.array([pts[i]["tgt"] for i in tr])
    X7=np.array([pts[i]["f7"] for i in tr]); Xe=np.array([pts[i]["eye"] for i in tr])
    Xc=np.concatenate([X7, Xe], axis=1)
    m7=fit_model(X7,Y); me=fit_model(Xe,Y); mc=fit_model(Xc,Y)
    for b in DBINS:
        lo,hi=b
        te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg and lo<=rel[i]<hi]
        if len(te)<6: continue
        Yte=np.array([pts[i]["tgt"] for i in te])
        X7t=np.array([pts[i]["f7"] for i in te]); Xet=np.array([pts[i]["eye"] for i in te])
        Xct=np.concatenate([X7t, Xet], axis=1)
        p7=predict(m7,X7t); pe=predict(me,Xet); pc=predict(mc,Xct)
        w=np.clip((rel[te]-0.05)/0.10,0.0,1.0)[:,None]
        pb=(1-w)*p7+w*pe
        res[b]["7d"]+=list(euc(p7,Yte)); res[b]["blend"]+=list(euc(pb,Yte)); res[b]["comb"]+=list(euc(pc,Yte))
log(f"\n**距離bin別 median cm**")
log(f"  {'距離変化':>10} | {'7D':>7} | {'ブレンド':>8} | {'結合10D':>8}")
for b in DBINS:
    r=res[b]
    if r["7d"] and r["comb"]:
        log(f"  {names[b]:>10} | {np.median(r['7d']):>5.2f} | {np.median(r['blend']):>6.2f} | {np.median(r['comb']):>6.2f}")
a7=[v for b in DBINS for v in res[b]["7d"]]; ab=[v for b in DBINS for v in res[b]["blend"]]; ac=[v for b in DBINS for v in res[b]["comb"]]
if ac:
    log(f"\n- 全体: 7D={np.median(a7):.2f} / ブレンド={np.median(ab):.2f} / 結合10D={np.median(ac):.2f}cm")
    log(f"- 結合がブレンド(7.18)より下がれば採用。目標3cm級にはまだ距離があるので次は時系列平滑/両眼輻輳。")
