"""exp42: 16D特徴アブレーション。各次元を1つ抜いて効く次元/冗長を特定。冗長除去でノイズ減り改善するか。
honest多点キャリブ。mainは触らない。
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
NAMES=['Lx','Ly','Rx','Ry','pitch','yaw','dist','roll','L_EAR','R_EAR','L_ivert','R_ivert','L_idiam','R_idiam','L_asp','R_asp']

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
    if len(idx)<60: continue
    pts=[]
    for k in idx:
        t=d["target"][k]
        if np.isnan(t).any(): continue
        w,h=float(d["img_w"][k]),float(d["img_h"][k])
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float)))
    if len(pts)>=60: sessions.append(pts)

# splitを固定して全keep設定を同一splitで比較
rng=np.random.RandomState(0)
splits=[]
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
    X=np.array([p["f16"] for p in pts]); Y=np.array([p["tgt"] for p in pts])
    splits.append((X,Y,tr,te))

def evalkeep(keep):
    err=[]
    for (X,Y,tr,te) in splits:
        err+=list(euc(fit_predict(X[np.ix_(tr,keep)],Y[tr],X[np.ix_(te,keep)]),Y[te]))
    return np.median(err)

log("\n---\n## exp42: 特徴アブレーション（16D各次元を抜く, honest, {}セッション）".format(len(splits)))
base=evalkeep(list(range(16)))
log(f"\n  baseline 16D = {base:.3f}cm")
log(f"  {'抜いた次元':>10} | {'15D誤差':>8} | {'Δ(正=重要/負=冗長)':>16}")
redundant=[]
for d in range(16):
    keep=[i for i in range(16) if i!=d]
    e=evalkeep(keep); delta=e-base
    tag = "冗長候補" if delta<-0.002 else ("重要" if delta>0.01 else "")
    log(f"  {NAMES[d]:>10} | {e:.3f}cm | {delta:+.3f} {tag}")
    if delta<-0.002: redundant.append(d)
if redundant:
    keep=[i for i in range(16) if i not in redundant]
    e=evalkeep(keep)
    log(f"\n  冗長{[NAMES[d] for d in redundant]}を全除去({len(keep)}D) = {e:.3f}cm (base {base:.3f})")
    log(f"  → {'改善!採用候補' if e<base-0.01 else '変化小'}。次はexp43 SOTA論文の移植。")
else:
    log(f"\n  冗長次元なし=16Dは全次元が寄与。次はexp43 SOTA論文の移植で新軸を探す。")
