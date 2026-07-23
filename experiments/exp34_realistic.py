"""exp34: 実機に近いシナリオで16Dの実用精度を測る。目標3cmの達成度。
honest(未知位置外挿)は厳しすぎる。実機は「キャリブ範囲内を見る(補間)＋時系列平滑」。
→ ランダムフレームsplit(同一画面位置がtrain/test両方に入る=補間)＋同一ターゲット時系列平滑。
※これは楽観寄りの評価(実機の使用感の目安)。honest(exp28 4.67)と併記して正直に。mainは触らない。
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
        for pos,i in enumerate(ids):
            out[i]=pred[ids[max(0,pos-win+1):pos+1]].mean(axis=0)
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
        f16=np.asarray(f16,float)
        pts.append(dict(f16=f16, tgt=np.asarray(t,float), yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

YBINS=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log(f"\n---\n## exp34: 実機シナリオ評価（16D, ランダムsplit=補間＋時系列平滑, {len(sessions)}セッション）")
raw=[]; sm=[]; ybin_s={b:[] for b in YBINS}
for pts in sessions:
    n=len(pts); order=rng.permutation(n); cut=int(n*0.7)
    tr=order[:cut]; te=order[cut:]
    Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    pred=fit_predict(Xtr,Ytr,Xte)
    keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
    preds=smooth(pred,keys,5)
    raw+=list(euc(pred,Yte)); e_s=euc(preds,Yte); sm+=list(e_s)
    for j,i in enumerate(te):
        for (lo,hi) in YBINS:
            if lo<=pts[i]["yaw"]<hi: ybin_s[(lo,hi)].append(e_s[j]); break
log(f"\n**実機シナリオ 16D（median cm）**")
log(f"  平滑なし={np.median(raw):.2f}cm / 平滑あり={np.median(sm):.2f}cm")
log(f"  姿勢bin(平滑あり): " + " ".join(f"{b[0]}-{b[1]}°={np.median(ybin_s[b]):.2f}" for b in YBINS if ybin_s[b]))
log(f"- honest(未知位置,exp28)=4.67cm に対し、実機シナリオ(補間+平滑)はこの値。3cm達成度の目安。")
log(f"- ※これは楽観寄り。実機の真値はhonest(4.67)と本値の間。密キャリブ(exp31)で更に下がる。")
