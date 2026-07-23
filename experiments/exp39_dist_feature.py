"""exp39: 虹彩から測った実距離(cm)を16Dに足す(17D)。距離情報で精度が上がるか。1cm台への一手。
honest多点キャリブ+実機シナリオ。16D vs 17D。デバイス非依存(距離はcmだが個人キャリブが吸収)。mainは触らない。
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
IRIS_MM = 11.7
REPORT = ROOT / "experiments" / "REPORT5_sota_transfer.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

def iris_diam_px(P, w, h):
    l = np.linalg.norm((P[469]-P[471]) * np.array([w, h, 0]))
    r = np.linalg.norm((P[474]-P[476]) * np.array([w, h, 0]))
    return (l + r) / 2.0
def euc_cm(pred, tgt):
    dd = pred - tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)
def fit_predict(Xtr,Ytr,Xte):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    pr=np.zeros((len(Xte),2))
    for i in range(2): pr[:,i]=HuberRegressor(epsilon=1.35,alpha=1e-3,max_iter=500).fit(A,Ytr[:,i]).predict(B)
    return pr
def smooth(pred, keys, win=5):
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
        arr=np.asarray(d["landmarks"][k],float)
        diam=iris_diam_px(arr,w,h)
        if diam<1e-3: continue
        dist=(IRIS_MM/10.0)*w/diam
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), dist=dist, tgt=np.asarray(t,float), yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

def getX(pts, ids, kind):
    if kind=="16D": return np.array([pts[i]["f16"] for i in ids])
    if kind=="17D": return np.array([np.concatenate([pts[i]["f16"],[pts[i]["dist"]]]) for i in ids])
YB=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log("\n---\n## exp39: 虹彩距離を16Dに追加(17D)")
# honest
hov={"16D":[], "17D":[]}
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
    for k in ["16D","17D"]:
        hov[k]+=list(euc_cm(fit_predict(getX(pts,tr,k),Ytr,getX(pts,te,k)),Yte))
# 実機シナリオ
rov={"16D":[], "17D":[]}
for pts in sessions:
    n=len(pts); order=rng.permutation(n); cut=int(n*0.7)
    tr=order[:cut]; te=order[cut:]
    Ytr=np.array([pts[i]["tgt"] for i in tr]); Yte=np.array([pts[i]["tgt"] for i in te])
    keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
    for k in ["16D","17D"]:
        pred=smooth(fit_predict(getX(pts,tr,k),Ytr,getX(pts,te,k)), keys, 5)
        rov[k]+=list(euc_cm(pred,Yte))
log(f"  honest:      16D={np.median(hov['16D']):.2f}cm  17D(+距離)={np.median(hov['17D']):.2f}cm")
log(f"  実機シナリオ: 16D={np.median(rov['16D']):.2f}cm  17D(+距離)={np.median(rov['17D']):.2f}cm")
d=np.median(hov['16D'])-np.median(hov['17D'])
log(f"  → 距離追加で honest {d:+.3f}cm ({'改善' if d>0 else '悪化/変化なし'})。改善なら採用、次はdata normalization/正則化調整。")
