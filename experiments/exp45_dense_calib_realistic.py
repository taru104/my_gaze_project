"""exp45: 密キャリブの実機シナリオ効果。学習点数(キャリブ量)を増やすと実機精度が1cm台に近づくか、
それとも飽和(16Dの限界)か。test固定20%、学習プール80%から比率を変えて学習。16D+平滑win3。
exp44(虹彩サブピクセル)はスキップ: MediaPipe虹彩は端点少+既にサブピクセルで効果薄と判断。mainは触らない。
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
def smooth(pred, keys, win=3):
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
    if len(idx)<100: continue
    pts=[]
    for k in idx:
        t=d["target"][k]
        if np.isnan(t).any(): continue
        w,h=float(d["img_w"][k]),float(d["img_h"][k])
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float)))
    if len(pts)>=100: sessions.append(pts)

RATIOS=[0.2,0.4,0.6,0.8,1.0]
rng=np.random.RandomState(0)
log("\n---\n## exp45: 密キャリブの実機シナリオ効果（16D+平滑win3, 学習点数↑で1cm台に近づくか）")
res={r:[] for r in RATIOS}; npt={r:[] for r in RATIOS}
for pts in sessions:
    n=len(pts); order=rng.permutation(n); tcut=int(n*0.8)
    pool=order[:tcut]; te=order[tcut:]
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
    for r in RATIOS:
        nk=max(20,int(len(pool)*r)); tr=pool[:nk]
        Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
        pred=smooth(fit_predict(Xtr,Ytr,Xte),keys,3)
        res[r]+=list(euc(pred,Yte)); npt[r].append(len(tr))
log(f"\n**学習点数(キャリブ量)別 実機シナリオ精度（median cm, @54cmで角度）**")
for r in RATIOS:
    cm=np.median(res[r]); a=np.degrees(np.arctan(cm/54.0))
    log(f"  比率{int(r*100):>3}% (学習≈{int(np.mean(npt[r]))}点) | {cm:.2f}cm ≈ {a:.2f}°")
prev=None; sat=True
for r in RATIOS:
    m=np.median(res[r])
    if prev is not None and prev-m>0.05: sat=False
    prev=m
log(f"  → {'点↑でまだ改善中=密キャリブで1cm台に近づける可能性' if not sat else '飽和=16Dの解像度限界に近い'}。")
