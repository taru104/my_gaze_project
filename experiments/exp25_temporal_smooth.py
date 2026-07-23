"""exp25: 時系列平滑で質を上げる。実機で一番効く「フレーム間ジッター除去」の効果をオフラインで測る。
同一ターゲット(=同じ画面位置を連続で見た点群)ごとに予測を時系列平均し、ノイズを低減。
7D / ブレンド それぞれで、平滑なし vs 平滑あり を距離bin別に比較。目標3cm。honest評価。mainは触らない。
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
WIN = 5  # 時系列平滑の窓(フレーム)

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

def smooth_by_target(pred, tgt_keys, win):
    """同一ターゲット(連続で見た点)ごとに、予測を窓winの移動平均で平滑。"""
    out = pred.copy()
    groups = defaultdict(list)
    for i, k in enumerate(tgt_keys): groups[k].append(i)
    for k, ids in groups.items():
        ids = sorted(ids)  # 元の順(=時系列)
        for pos, i in enumerate(ids):
            lo = max(0, pos - win + 1)
            sel = ids[lo:pos + 1]
            out[i] = pred[sel].mean(axis=0)
    return out

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
log(f"\n---\n## exp25: 時系列平滑(窓{WIN})（{len(sessions)}セッション, honest, 目標3cm）")
res={b:{"7d":[],"7dS":[],"blend":[],"blendS":[]} for b in DBINS}
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
    m7=fit_model(np.array([pts[i]["f7"] for i in tr]),Y); me=fit_model(np.array([pts[i]["eye"] for i in tr]),Y)
    for b in DBINS:
        lo,hi=b
        te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg and lo<=rel[i]<hi]
        if len(te)<6: continue
        Yte=np.array([pts[i]["tgt"] for i in te])
        keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
        p7=predict(m7,np.array([pts[i]["f7"] for i in te])); pe=predict(me,np.array([pts[i]["eye"] for i in te]))
        w=np.clip((rel[te]-0.05)/0.10,0.0,1.0)[:,None]; pb=(1-w)*p7+w*pe
        p7s=smooth_by_target(p7, keys, WIN); pbs=smooth_by_target(pb, keys, WIN)
        res[b]["7d"]+=list(euc(p7,Yte)); res[b]["7dS"]+=list(euc(p7s,Yte))
        res[b]["blend"]+=list(euc(pb,Yte)); res[b]["blendS"]+=list(euc(pbs,Yte))
log(f"\n**距離bin別 median cm（S=時系列平滑あり）**")
log(f"  {'距離変化':>10} | {'7D':>6} | {'7D+平滑':>7} | {'ブレンド':>7} | {'ブレンド+平滑':>10}")
for b in DBINS:
    r=res[b]
    if r["7d"] and r["blendS"]:
        log(f"  {names[b]:>10} | {np.median(r['7d']):>4.2f} | {np.median(r['7dS']):>5.2f} | {np.median(r['blend']):>5.2f} | {np.median(r['blendS']):>8.2f}")
def med(k): return np.median([v for b in DBINS for v in res[b][k]])
log(f"\n- 全体: 7D={med('7d'):.2f} / 7D+平滑={med('7dS'):.2f} / ブレンド={med('blend'):.2f} / ブレンド+平滑={med('blendS'):.2f}cm")
log(f"- 時系列平滑がジッターを消してどれだけ下がるか。実機はさらにキャリブ近傍なので、これより良くなる想定。")
