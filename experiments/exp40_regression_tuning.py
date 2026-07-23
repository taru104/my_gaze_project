"""exp40: 正則化/回帰の調整で16Dの壁を下げる。Huber alpha掃引/Ridge/2次多項式展開(交差項)を比較。
過学習はhonestで弾く。最良を実機シナリオでも確認。mainは触らない。
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
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT5_sota_transfer.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

def euc(pred,tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)

def predict(kind, alpha, Xtr, Ytr, Xte):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    if kind=="poly2":
        pf=PolynomialFeatures(2, include_bias=False); A=pf.fit_transform(A); B=pf.transform(B)
    pr=np.zeros((len(Xte),2))
    for i in range(2):
        m = Ridge(alpha=alpha) if kind in ("ridge","poly2") else HuberRegressor(epsilon=1.35,alpha=alpha,max_iter=500)
        pr[:,i]=m.fit(A,Ytr[:,i]).predict(B)
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
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float)))
    if len(pts)>=60: sessions.append(pts)

CONFIGS=[("huber",1e-4),("huber",1e-3),("huber",1e-2),("huber",1e-1),
         ("ridge",1.0),("ridge",10.0),("ridge",100.0),
         ("poly2",10.0),("poly2",100.0),("poly2",1000.0)]
rng=np.random.RandomState(0)
log("\n---\n## exp40: 正則化/回帰の調整（16D, honest多点キャリブ）")
res={c:[] for c in CONFIGS}
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
    for (kind,a) in CONFIGS:
        res[(kind,a)]+=list(euc(predict(kind,a,Xtr,Ytr,Xte),Yte))
log(f"\n**設定別 honest median cm（基準: Huber1e-3=4.71cm）**")
best=None
for c in CONFIGS:
    m=np.median(res[c]); log(f"  {c[0]:>6} α={c[1]:<7} | {m:.3f}cm")
    if best is None or m<best[1]: best=(c,m)
log(f"\n  → 最良: {best[0][0]} α={best[0][1]} = {best[1]:.3f}cm")
# 最良を実機シナリオでも
bk,ba=best[0]
rov={"16D_Huber1e-3":[], "best":[]}
for pts in sessions:
    n=len(pts); order=rng.permutation(n); cut=int(n*0.7)
    tr=order[:cut]; te=order[cut:]
    Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
    rov["16D_Huber1e-3"]+=list(euc(smooth(predict("huber",1e-3,Xtr,Ytr,Xte),keys),Yte))
    rov["best"]+=list(euc(smooth(predict(bk,ba,Xtr,Ytr,Xte),keys),Yte))
log(f"  実機シナリオ: 従来Huber1e-3={np.median(rov['16D_Huber1e-3']):.2f}cm  最良({bk} α={ba})={np.median(rov['best']):.2f}cm")
log(f"  → 改善したら採用。次はexp41 data normalization。")
