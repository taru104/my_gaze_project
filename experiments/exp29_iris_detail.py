"""exp29: 16Dに虹彩の詳細形状(5点×2目を目頭目尻で正規化=20D)と両眼輻輳を追加。
16D vs 36D vs 37D(+輻輳)を線形Huber・多点キャリブ・honestで比較。過学習に注意。GPU不要。mainは触らない。
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
from features import _geo_normalize
from rich16d import rich_16d_from_lms
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
LEFT_IRIS=[468,469,470,471,472]; RIGHT_IRIS=[473,474,475,476,477]

class LM:
    __slots__=("x","y","z")
    def __init__(s,a): s.x=float(a[0]); s.y=float(a[1]); s.z=float(a[2])

def iris_detail(lm,w,h):
    out=[]
    for ids,oi,ii in [(LEFT_IRIS,33,362),(RIGHT_IRIS,263,133)]:
        o=np.array([lm[oi].x*w,lm[oi].y*h]); inn=np.array([lm[ii].x*w,lm[ii].y*h])
        for pi in ids:
            p=np.array([lm[pi].x*w,lm[pi].y*h])
            out+=list(_geo_normalize(p,inn,o))
    return np.array(out,float)  # 20D

def vergence(lm,w,h):
    # 両眼虹彩中心の相対水平差(輻輳の代理)。目間距離で正規化。
    Lc=np.array([lm[468].x*w,lm[468].y*h]); Rc=np.array([lm[473].x*w,lm[473].y*h])
    Lo=np.array([lm[33].x*w,lm[33].y*h]); Ro=np.array([lm[263].x*w,lm[263].y*h])
    base=np.linalg.norm(Ro-Lo)+1e-6
    return np.array([(Rc[0]-Lc[0])/base, (Rc[1]-Lc[1])/base],float)  # 2D

def euc(pred,tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)
def fit_predict(Xtr,Ytr,Xte):
    sc=StandardScaler().fit(Xtr); a,b=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for i in range(2): pr[:,i]=HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=600).fit(a,Ytr[:,i]).predict(b)
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
        arr=d["landmarks"][k]
        try: f16=rich_16d_from_lms(arr,int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        lm=[LM(p) for p in arr]
        f16=np.asarray(f16,float)
        pts.append(dict(f16=f16, iris=iris_detail(lm,w,h), verg=vergence(lm,w,h),
                        tgt=np.asarray(t,float), yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

def getX(pts,ids,kind):
    if kind=="16D": return np.array([pts[i]["f16"] for i in ids])
    if kind=="36D": return np.array([np.concatenate([pts[i]["f16"],pts[i]["iris"]]) for i in ids])
    if kind=="38D": return np.array([np.concatenate([pts[i]["f16"],pts[i]["iris"],pts[i]["verg"]]) for i in ids])
KINDS=["16D","36D","38D"]
YBINS=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log(f"\n---\n## exp29: 虹彩詳細+輻輳 16D/36D/38D（多点キャリブ, honest, {len(sessions)}セッション）")
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
        e=euc(fit_predict(getX(pts,tr,k),Ytr,getX(pts,te,k)),Yte)
        overall[k]+=list(e)
        for j in range(len(te)):
            for (lo,hi) in YBINS:
                if lo<=yaws[j]<hi: ybin[k][(lo,hi)].append(e[j]); break
log(f"\n**特徴別 median cm（honest, 線形Huber）**")
log(f"  {'特徴':>6} | {'全体':>6} | {'0-10':>5} | {'10-20':>5} | {'20-30':>5} | {'30+':>5}")
best=None
for k in KINDS:
    o=np.median(overall[k]) if overall[k] else float('nan')
    c=[f"{np.median(ybin[k][b]):.2f}" if ybin[k][b] else "--" for b in YBINS]
    log(f"  {k:>6} | {o:>5.2f} | {c[0]:>5} | {c[1]:>5} | {c[2]:>5} | {c[3]:>5}")
    if not np.isnan(o) and (best is None or o<best[1]): best=(k,o)
if best: log(f"\n- 最良: {best[0]}={best[1]:.2f}cm（exp28の16D=4.67cm比）。虹彩詳細で下がれば採用、悪化なら過学習。")
