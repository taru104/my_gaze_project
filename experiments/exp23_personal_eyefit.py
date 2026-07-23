"""exp23: 眼球パラメータ(眼球中心の内寄せ・奥行き・半径)を個人(セッション)ごとにキャリブfit。
画面サイズには一切依存しない(デバイス非依存)。train点で各人に最適なパラメータを選び、test点で
7D / 眼球 / 距離ブレンド を評価。exp22(全人固定パラメータ)より個人適応で良くなるか。
狙い: 眼球モデル単独の絶対精度を上げ、ブレンド全体を改善する。honest評価。mainは触らない。
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

SW, SH = 30.9, 17.4  # cm換算は「評価表示のみ」。モデルは画面サイズ非依存(正規化target)。
REPORT = ROOT / "experiments" / "REPORT3_eyemodel.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
L_IRIS, R_IRIS = 468, 473

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def sphere_hit(ray, ec, r):
    L=-ec; b=2.0*ray@L; c=L@L-r*r; disc=b*b-4.0*c
    tt=-(ray@L)/(ray@ray) if disc<0 else (-b-np.sqrt(disc))/2.0
    return tt*ray

def eye_feat(p, xi, zo, er):
    ecl=p["R"]@np.array([-xi,33.0,zo])+p["t"]; ecr=p["R"]@np.array([xi,33.0,zo])+p["t"]
    il=sphere_hit(p["rl"],ecl,er); ir=sphere_hit(p["rr"],ecr,er)
    gl=il-ecl; gr=ir-ecr; nl,nr=np.linalg.norm(gl),np.linalg.norm(gr)
    if nl<1e-9 or nr<1e-9: return None
    g=gl/nl+gr/nr; ng=np.linalg.norm(g)
    return g/ng if ng>1e-9 else None

def euc(pred, tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

def fit_model(Xtr, Ytr):
    sc=StandardScaler().fit(Xtr)
    ms=[HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500).fit(sc.transform(Xtr), Ytr[:,a]) for a in range(2)]
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
        r7=build_7d_features(lm,int(w),int(h))
        if r7 is None: continue
        f7,yaw,_,_=r7
        pts.append(dict(R=R,t=tv,rl=ray(L_IRIS),rr=ray(R_IRIS),f7=np.asarray(f7,float),
                        tgt=np.asarray(t,float),dist=float(np.linalg.norm(tv)),yaw=abs(np.degrees(yaw))))
    if len(pts)>=60: sessions.append(pts)

GRID=[(xi,zo,er) for xi in (26.,30.,34.) for zo in (-20.,-30.,-42.) for er in (12.,16.,20.)]
DBINS=[(0.0,0.05),(0.05,0.12),(0.12,10.0)]
names={DBINS[0]:"±5%(近)",DBINS[1]:"5-12%(中)",DBINS[2]:">12%(遠)"}
rng=np.random.RandomState(0)
log(f"\n---\n## exp23: 眼球パラメータの個人キャリブfit（画面サイズ非依存, {len(sessions)}セッション, honest）")
res={b:{"7d":[],"eyeFix":[],"eyePers":[],"blend":[]} for b in DBINS}
FIX=(28.,-30.,16.)  # exp20全人固定
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
    Ytr=np.array([pts[i]["tgt"] for i in tr])
    # --- 個人パラメータ選択: train点でeye特徴→train回帰、train誤差最小 ---
    bestp, beste = FIX, 1e9
    for pr in GRID:
        Xe=[]; ok=True
        for i in tr:
            g=eye_feat(pts[i],*pr)
            if g is None: ok=False; break
            Xe.append(g)
        if not ok: continue
        Xe=np.array(Xe); m=fit_model(Xe,Ytr)
        e=np.median(euc(predict(m,Xe),Ytr))
        if e<beste: beste=e; bestp=pr
    # モデル: 7D, 眼球(固定FIX), 眼球(個人bestp) を train でfit
    X7tr=np.array([pts[i]["f7"] for i in tr])
    XeFix=np.array([eye_feat(pts[i],*FIX) for i in tr])
    XePer=np.array([eye_feat(pts[i],*bestp) for i in tr])
    m7=fit_model(X7tr,Ytr); mef=fit_model(XeFix,Ytr); mep=fit_model(XePer,Ytr)
    for b in DBINS:
        lo,hi=b
        te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg and lo<=rel[i]<hi]
        if len(te)<6: continue
        Yte=np.array([pts[i]["tgt"] for i in te])
        p7=predict(m7, np.array([pts[i]["f7"] for i in te]))
        pef=predict(mef, np.array([eye_feat(pts[i],*FIX) for i in te]))
        pep=predict(mep, np.array([eye_feat(pts[i],*bestp) for i in te]))
        w=np.clip((rel[te]-0.05)/0.10,0.0,1.0)[:,None]
        pb=(1-w)*p7 + w*pep   # ブレンドは個人眼球を使用
        res[b]["7d"]+=list(euc(p7,Yte)); res[b]["eyeFix"]+=list(euc(pef,Yte))
        res[b]["eyePers"]+=list(euc(pep,Yte)); res[b]["blend"]+=list(euc(pb,Yte))
log(f"\n**距離bin別 median cm（眼球は個人キャリブfit）**")
log(f"  {'距離変化':>10} | {'7D':>7} | {'眼球固定':>8} | {'眼球個人':>8} | {'ブレンド':>8}")
for b in DBINS:
    r=res[b]
    if r["7d"] and r["eyePers"] and r["blend"]:
        log(f"  {names[b]:>10} | {np.median(r['7d']):>5.2f} | {np.median(r['eyeFix']):>6.2f} | {np.median(r['eyePers']):>6.2f} | {np.median(r['blend']):>6.2f}")
allb=[v for b in DBINS for v in res[b]["blend"]]; all7=[v for b in DBINS for v in res[b]["7d"]]
allef=[v for b in DBINS for v in res[b]["eyeFix"]]; allep=[v for b in DBINS for v in res[b]["eyePers"]]
if allb:
    log(f"\n- 全体: 7D={np.median(all7):.2f} / 眼球固定={np.median(allef):.2f} / 眼球個人={np.median(allep):.2f} / ブレンド={np.median(allb):.2f}cm")
    log(f"- exp22(固定)ブレンド7.18cm → exp23(個人fit)ブレンド={np.median(allb):.2f}cm。個人適応で眼球が改善しブレンドも良くなるか。")
