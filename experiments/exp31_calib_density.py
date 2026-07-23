"""exp31: キャリブ密度の効果。学習に使う画面位置の割合(=キャリブ点数)を変え、未知位置の補間精度を見る。
実機で「画面を細かくたくさんキャリブする」に相当。密なキャリブで3cmに近づくか。16D線形Huber・honest。mainは触らない。
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

sessions=[]
for binp in sorted(glob.glob(str(ROOT/"logs"/"*_landmarks.bin"))):
    try: d=load_raw_landmarks(binp)
    except Exception: continue
    idx=np.where(d["has_target"])[0]
    if len(idx)<80: continue
    pts=[]
    for k in idx:
        t=d["target"][k]
        if np.isnan(t).any(): continue
        w,h=float(d["img_w"][k]),float(d["img_h"][k])
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float)))
    if len(pts)>=80: sessions.append(pts)

# テスト位置は固定(全体の30%)、学習に使う残り70%のうち何割を実際に使うかで密度を変える
DENS=[0.3,0.5,0.7,1.0]  # 学習可能プールのうち使う割合
rng=np.random.RandomState(0)
log(f"\n---\n## exp31: キャリブ密度（16D, honest, {len(sessions)}セッション）")
res={f"密度{int(dd*100)}%":[] for dd in DENS}; npts={f"密度{int(dd*100)}%":[] for dd in DENS}
for pts in sessions:
    groups=defaultdict(list)
    for i,p in enumerate(pts): groups[(round(p["tgt"][0],1),round(p["tgt"][1],1))].append(i)
    gk=list(groups.keys())
    if len(gk)<6: continue
    order=rng.permutation(len(gk)); cut=int(len(gk)*0.7)
    poolg=[gk[j] for j in order[:cut]]; teg=set(gk[j] for j in order[cut:])
    te=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in teg]
    if len(te)<10: continue
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    for dd in DENS:
        nkeep=max(2,int(len(poolg)*dd)); keepg=set(poolg[:nkeep])
        tr=[i for i,p in enumerate(pts) if (round(p["tgt"][0],1),round(p["tgt"][1],1)) in keepg]
        if len(tr)<20: continue
        Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
        res[f"密度{int(dd*100)}%"]+=list(euc(fit_predict(Xtr,Ytr,Xte),Yte))
        npts[f"密度{int(dd*100)}%"].append(len(tr))
log(f"\n**キャリブ密度別 median cm（16D, honest, テスト位置固定）**")
for dd in DENS:
    k=f"密度{int(dd*100)}%"
    if res[k]:
        log(f"  {k:>8} (学習≈{int(np.mean(npts[k]))}点) | {np.median(res[k]):.2f}cm")
log("- 密度を上げて誤差が下がるなら『実機でたくさんキャリブ点を取る』が3cmへの直接策。頭打ちなら補間の限界。")
