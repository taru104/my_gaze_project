"""exp41: 1cm台の本命=実機側の改善。時系列平滑の窓長を掃引し、静止精度の到達点を見る。
回帰調整(exp40)は効かず→honest壁(4.71cm,未知位置補間)は回帰では下がらない。1cm台は実機シナリオ
(補間+平滑)で狙う。16D・実機シナリオ・平滑窓win掃引。mainは触らない。
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
REPORT = ROOT / "experiments" / "REPORT5_sota_transfer.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(pred,tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)
def fit_predict(Xtr,Ytr,Xte):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for i in range(2): pr[:,i]=HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=500).fit(A,Ytr[:,i]).predict(B)
    return pr
def smooth(pred, keys, win):
    if win<=1: return pred
    out=pred.copy(); g=defaultdict(list)
    for i,k in enumerate(keys): g[k].append(i)
    for k,ids in g.items():
        ids=sorted(ids)
        for pos,i in enumerate(ids): out[i]=pred[ids[max(0,pos-win+1):pos+1]].mean(axis=0)
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
        w,h=float(d["img_w"][k]),float(d["img_h"][k])
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float), yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

WINS=[1,3,5,10,20,40,80]
rng=np.random.RandomState(0)
log("\n---\n## exp41: 時系列平滑の窓長掃引（16D 実機シナリオ, 静止精度の到達点）")
res={w:[] for w in WINS}; ang={w:[] for w in WINS}
IRIS_MM=11.7
def iris_dist(P,w,h):
    l=np.linalg.norm((P[469]-P[471])*np.array([w,h,0])); r=np.linalg.norm((P[474]-P[476])*np.array([w,h,0]))
    return (IRIS_MM/10.0)*w/((l+r)/2.0) if (l+r)>1e-3 else 55.0
for pts, binp in [(s,None) for s in sessions]:
    n=len(pts); order=rng.permutation(n); cut=int(n*0.7)
    tr=order[:cut]; te=order[cut:]
    Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
    pred0=fit_predict(Xtr,Ytr,Xte)
    for w in WINS:
        e=euc(smooth(pred0,keys,w),Yte); res[w]+=list(e)
log(f"\n**平滑窓 win ごとの実機シナリオ精度（median cm, @54cmで角度換算）**")
for w in WINS:
    cm=np.median(res[w]); a=np.degrees(np.arctan(cm/54.0))
    log(f"  win={w:>3} | {cm:.2f}cm ≈ {a:.2f}°")
log(f"- 窓を長くすると静止精度↑(遅延も↑)。1cm台(≈1.06°)が見えるか。次はexp42特徴アブレーション/exp44密キャリブ。")
