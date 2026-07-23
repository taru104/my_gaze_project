"""exp30: 局所加重回帰(LWR)。テスト点の姿勢(yaw)に近い学習点を加重して局所線形Huber回帰。
グローバル16D(4.67cm)より下がるか。sigma(加重幅deg)を振る。honest多点キャリブ。GPU不要。mainは触らない。
"""
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from raw_landmark_logger import load_raw_landmarks
from rich16d import rich_16d_from_lms
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

def euc(pred,tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

def global_fit(Xtr,Ytr,Xte):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for i in range(2): pr[:,i]=HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=500).fit(A,Ytr[:,i]).predict(B)
    return pr

def lwr_fit(Xtr,Ytr,yaw_tr,Xte,yaw_te,sigma):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for j in range(len(Xte)):
        wt=np.exp(-((yaw_te[j]-yaw_tr)**2)/(2*sigma*sigma)); wt=np.maximum(wt,1e-4)
        for i in range(2):
            pr[j,i]=HuberRegressor(epsilon=1.35,alpha=1e-2,max_iter=200).fit(A,Ytr[:,i],sample_weight=wt).predict(B[j:j+1])[0]
    return pr

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
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        f16=np.asarray(f16,float)
        pts.append(dict(f16=f16, tgt=np.asarray(t,float), yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

SIGMAS=[10.0,20.0]
rng=np.random.RandomState(0)
log(f"\n---\n## exp30: 局所加重回帰LWR（16D, honest, {len(sessions)}セッション）")
res={"global":[]}; res.update({f"LWR σ={s:.0f}":[] for s in SIGMAS})
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
    Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    ytr=np.array([pts[i]["yaw"] for i in tr]); yte=np.array([pts[i]["yaw"] for i in te])
    res["global"]+=list(euc(global_fit(Xtr,Ytr,Xte),Yte))
    for s in SIGMAS:
        res[f"LWR σ={s:.0f}"]+=list(euc(lwr_fit(Xtr,Ytr,ytr,Xte,yte,s),Yte))
log(f"\n**手法別 median cm（16D, honest）**")
best=None
for k,v in res.items():
    if v:
        m=np.median(v); log(f"  {k:>12} | {m:.2f}cm")
        if best is None or m<best[1]: best=(k,m)
if best: log(f"\n- 最良: {best[0]}={best[1]:.2f}cm（グローバル16D 4.67cm比）。局所化で下がるか。次はアンサンブル/キャリブ密度。")
