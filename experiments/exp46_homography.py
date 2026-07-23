"""exp46: homographyキャリブ。虹彩2D位置→画面2Dを射影変換(透視)でマッピング。線形回帰との比較。
視線は透視投影を含むので射影変換が合うかも。虹彩2D単独/虹彩2D→homography残差を16Dで補正 も試す。honest+実機。mainは触らない。
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
def huber(Xtr,Ytr,Xte):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for i in range(2): pr[:,i]=HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=500).fit(A,Ytr[:,i]).predict(B)
    return pr
def homog(src_tr,dst_tr,src_te):
    H,_=cv2.findHomography(src_tr.astype(np.float64), dst_tr.astype(np.float64), 0)
    if H is None: return None
    return cv2.perspectiveTransform(src_te.reshape(-1,1,2).astype(np.float64), H).reshape(-1,2)

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
        iris2d=(f16[0:2]+f16[2:4])/2.0
        pts.append(dict(f16=f16, iris2d=iris2d, tgt=np.asarray(t,float)))
    if len(pts)>=60: sessions.append(pts)

rng=np.random.RandomState(0)
log("\n---\n## exp46: homographyキャリブ（虹彩2D→画面 射影変換）honest")
res={"16D線形":[], "homography(虹彩2D)":[], "homography+16D残差":[]}
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
    X16tr=np.array([pts[i]["f16"] for i in tr]); X16te=np.array([pts[i]["f16"] for i in te])
    ir_tr=np.array([pts[i]["iris2d"] for i in tr]); ir_te=np.array([pts[i]["iris2d"] for i in te])
    Ytr=np.array([pts[i]["tgt"] for i in tr]); Yte=np.array([pts[i]["tgt"] for i in te])
    res["16D線形"]+=list(euc(huber(X16tr,Ytr,X16te),Yte))
    ph=homog(ir_tr,Ytr,ir_te)
    if ph is not None:
        res["homography(虹彩2D)"]+=list(euc(ph,Yte))
        # homographyの残差を16Dで補正: 学習側の残差を16Dで回帰しtestに足す
        ph_tr=homog(ir_tr,Ytr,ir_tr)
        if ph_tr is not None:
            resid=Ytr-ph_tr
            corr=huber(X16tr,resid,X16te)
            res["homography+16D残差"]+=list(euc(ph+corr,Yte))
log(f"\n**手法別 honest median cm**")
for k,v in res.items():
    if v: log(f"  {k:>18} | {np.median(v):.2f}cm")
b=min((np.median(v) for v in res.values() if v))
log(f"  → 最良={b:.2f}cm（16D線形4.71cm）。射影変換で改善するか。改善なければ16D幾何の限界確定=朝に正直報告。")
