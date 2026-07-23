"""exp27: 回帰手法の探索。これまで線形Huberのみ→非線形(RF/GBM/KNN/SVR)で5-7cmの壁を破れるか。
7D特徴・多点キャリブ(全距離姿勢学習)・honest(未知画面位置test)・姿勢bin別。GPU不要。mainは触らない。
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
from features import build_7d_features
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

class LM:
    __slots__ = ("x","y","z")
    def __init__(self, a): self.x=float(a[0]); self.y=float(a[1]); self.z=float(a[2])

def euc(pred, tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

METHODS = {
    "Huber(線形)": lambda: HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=600),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=0),
    "GradBoost":    lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0),
    "KNN(k=12)":    lambda: KNeighborsRegressor(n_neighbors=12, weights="distance"),
    "SVR(rbf)":     lambda: SVR(kernel="rbf", C=10.0, gamma="scale"),
}

def fit_predict(ctor, Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
    pred = np.zeros((len(Xte), 2))
    for a in range(2):
        pred[:, a] = ctor().fit(Xtr2, Ytr[:, a]).predict(Xte2)
    return pred

# 収集(7D特徴 + yaw)
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
        lm=[LM(p) for p in d["landmarks"][k]]
        r7=build_7d_features(lm,int(w),int(h))
        if r7 is None: continue
        f7,yaw,_,_=r7
        pts.append(dict(f7=np.asarray(f7,float), tgt=np.asarray(t,float), yaw=abs(np.degrees(yaw))))
    if len(pts)>=60: sessions.append(pts)

YBINS=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log(f"\n---\n## exp27: 回帰手法の探索（7D, 多点キャリブ, honest, {len(sessions)}セッション）")
overall={m:[] for m in METHODS}; ybin={m:{b:[] for b in YBINS} for m in METHODS}
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
    Xtr=np.array([pts[i]["f7"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
    Xte=np.array([pts[i]["f7"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    yaws=[pts[i]["yaw"] for i in te]
    for m, ctor in METHODS.items():
        e=euc(fit_predict(ctor, Xtr, Ytr, Xte), Yte)
        overall[m]+=list(e)
        for j in range(len(te)):
            for (lo,hi) in YBINS:
                if lo<=yaws[j]<hi: ybin[m][(lo,hi)].append(e[j]); break
log(f"\n**手法別 median cm（honest, 多点キャリブ）**")
log(f"  {'手法':>14} | {'全体':>6} | {'0-10':>5} | {'10-20':>5} | {'20-30':>5} | {'30+':>5}")
best=None
for m in METHODS:
    o=np.median(overall[m]) if overall[m] else float('nan')
    cells=[f"{np.median(ybin[m][b]):.2f}" if ybin[m][b] else "--" for b in YBINS]
    log(f"  {m:>14} | {o:>5.2f} | {cells[0]:>5} | {cells[1]:>5} | {cells[2]:>5} | {cells[3]:>5}")
    if not np.isnan(o) and (best is None or o<best[1]): best=(m,o)
if best:
    log(f"\n- 最良手法: {best[0]} = {best[1]:.2f}cm（線形Huberの5.71cmと比較）。非線形が壁を破れるか。")
