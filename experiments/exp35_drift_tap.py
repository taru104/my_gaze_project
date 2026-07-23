"""exp35: 実機の時間ドリフト(『16dでも崩れる』)をタップ適応で潰せるか。
セッションを時系列で前半(キャリブ)→後半(使用)に分け、後半=ドリフト条件。
(A)前半キャリブのみ (B)+後半タップをN個ずつ学習に追加 で後半の誤差を比較。16D。mainは触らない。
"""
import sys, glob
from pathlib import Path
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
    if len(idx)<100: continue
    pts=[]
    for k in idx:  # idx昇順=時系列
        t=d["target"][k]
        if np.isnan(t).any(): continue
        w,h=float(d["img_w"][k]),float(d["img_h"][k])
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float)))
    if len(pts)>=100: sessions.append(pts)

NS=[0,10,30,60]
log(f"\n---\n## exp35: 時間ドリフト×タップ適応（16D, 前半キャリブ→後半使用, {len(sessions)}セッション）")
err={n:[] for n in NS}
for pts in sessions:
    half=len(pts)//2
    cal=pts[:half]; use=pts[half:]  # 時系列前半=キャリブ, 後半=使用(ドリフト)
    # 後半をタップ源(前から順)とテスト(後ろ)に分ける
    ncut=len(use)//2
    tap=use[:ncut]; test=use[ncut:]
    if len(cal)<30 or len(tap)<60 or len(test)<20: continue
    Xtest=np.array([p["f16"] for p in test]); Ytest=np.array([p["tgt"] for p in test])
    for n in NS:
        train=cal + tap[:n]
        Xtr=np.array([p["f16"] for p in train]); Ytr=np.array([p["tgt"] for p in train])
        err[n]+=list(euc(fit_predict(Xtr,Ytr,Xtest),Ytest))
log(f"\n**後半(ドリフト)の誤差 median cm（タップN個追加）**")
base=None
for n in NS:
    if err[n]:
        m=np.median(err[n]); log(f"  タップ{n:>2}個 | {m:.2f}cm")
        if n==0: base=m
if base is not None and err[NS[-1]]:
    fin=np.median(err[NS[-1]])
    log(f"\n- ドリフト誤差: タップ0={base:.2f}→{NS[-1]}個={fin:.2f}cm ({(base-fin)/base*100:+.0f}%)。")
    log(f"- タップ適応でドリフトが補正され誤差が下がれば、実機の『16dでも崩れる』を潰せる=3cm実機安定の鍵。")
