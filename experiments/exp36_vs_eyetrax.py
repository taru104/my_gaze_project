"""exp36: EyeTraxを超えているか直接検証。EyeTrax特徴(486D=161眼領域点×3+pose)を.binランドマークから
再現し、EyeTrax相当(StandardScaler+Ridge) vs あなたの16D(StandardScaler+Huber)を全く同じ条件
(同一logs・honest多点キャリブ・姿勢bin)で比較。cm誤差。GPU不要。mainは触らない。
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
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

# EyeTrax constants.py より (161点)
LEFT=[107,66,105,63,70,55,65,52,53,46,468,469,470,471,472,133,33,173,157,158,159,160,161,246,
      155,154,153,145,144,163,7,243,190,56,28,27,29,30,247,130,25,110,24,23,22,26,112,
      244,189,221,222,223,224,225,113,226,31,228,229,230,231,232,233,193,245,128,121,120,119,118,117,111,35,124,143,156]
RIGHT=[336,296,334,293,300,285,295,282,283,276,473,476,475,474,477,362,263,398,384,385,386,387,388,466,
       382,381,380,374,373,390,249,463,414,286,258,257,259,260,467,359,255,339,254,253,252,256,341,
       464,413,441,442,443,444,445,342,446,261,448,449,450,451,452,453,417,465,357,350,349,348,347,346,340,265,353,372,383]
MUTUAL=[4,10,151,9,152,234,454,58,288]
SUBSET=LEFT+RIGHT+MUTUAL

def eyetrax_feat(P):
    """P:(478,3) MediaPipe正規化 → EyeTrax 486D特徴(眼領域正規化+pose)。gaze.py extract_features と同一。"""
    left=P[33]; right=P[263]; top=P[10]
    ec=(left+right)/2.0; sh=P-ec
    xa=right-left; xa=xa/(np.linalg.norm(xa)+1e-9)
    ya=top-ec; ya=ya/(np.linalg.norm(ya)+1e-9); ya=ya-np.dot(ya,xa)*xa; ya=ya/(np.linalg.norm(ya)+1e-9)
    za=np.cross(xa,ya); za=za/(np.linalg.norm(za)+1e-9)
    R=np.column_stack((xa,ya,za)); rot=(R.T@sh.T).T
    lc=R.T@(left-ec); rc=R.T@(right-ec); inter=np.linalg.norm(rc-lc)
    if inter>1e-7: rot=rot/inter
    feat=rot[SUBSET].flatten()
    yaw=np.arctan2(R[1,0],R[0,0]); pitch=np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2)); roll=np.arctan2(R[2,1],R[2,2])
    return np.concatenate([feat,[yaw,pitch,roll]]), np.degrees(abs(np.arctan2(R[1,0],R[0,0])))

def euc(pred,tgt):
    dd=pred-tgt; return np.hypot(dd[:,0]*SW, dd[:,1]*SH)
def fit_ridge(Xtr,Ytr,Xte,alpha):
    sc=StandardScaler().fit(Xtr); A,B=sc.transform(Xtr),sc.transform(Xte)
    r=Ridge(alpha=alpha).fit(A,Ytr); return r.predict(B)
def fit_huber(Xtr,Ytr,Xte):
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
        arr=d["landmarks"][k]
        try: f16=rich_16d_from_lms(arr,int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        et,yaw=eyetrax_feat(np.asarray(arr,float))
        pts.append(dict(et=et, f16=np.asarray(f16,float), tgt=np.asarray(t,float), yaw=abs(np.degrees(float(f16[5])))))
    if len(pts)>=60: sessions.append(pts)

YBINS=[(0,10),(10,20),(20,30),(30,90)]
rng=np.random.RandomState(0)
log(f"\n---\n## exp36: ★EyeTrax vs 16D 直接対決（同一logs・honest多点キャリブ, {len(sessions)}セッション）")
ov={"EyeTrax486D":[], "16D(あなた)":[]}; yb={m:{b:[] for b in YBINS} for m in ov}
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
    yaws=[pts[i]["yaw"] for i in te]
    alpha=max(1.0, 5000.0/len(tr))  # EyeTrax tuned(過去評価と同じ適応的shrinkage)
    pe=fit_ridge(np.array([pts[i]["et"] for i in tr]),Ytr,np.array([pts[i]["et"] for i in te]),alpha)
    p16=fit_huber(np.array([pts[i]["f16"] for i in tr]),Ytr,np.array([pts[i]["f16"] for i in te]))
    ee=euc(pe,Yte); e16=euc(p16,Yte)
    ov["EyeTrax486D"]+=list(ee); ov["16D(あなた)"]+=list(e16)
    for j in range(len(te)):
        for (lo,hi) in YBINS:
            if lo<=yaws[j]<hi: yb["EyeTrax486D"][(lo,hi)].append(ee[j]); yb["16D(あなた)"][(lo,hi)].append(e16[j]); break
log(f"\n**median cm（同一条件, honest多点キャリブ）**")
log(f"  {'手法':>14} | {'全体':>6} | {'0-10':>5} | {'10-20':>5} | {'20-30':>5} | {'30+':>5}")
for m in ov:
    o=np.median(ov[m]); c=[f"{np.median(yb[m][b]):.2f}" if yb[m][b] else "--" for b in YBINS]
    log(f"  {m:>14} | {o:>5.2f} | {c[0]:>5} | {c[1]:>5} | {c[2]:>5} | {c[3]:>5}")
e,m16=np.median(ov["EyeTrax486D"]),np.median(ov["16D(あなた)"])
log(f"\n- **{'16Dの勝ち' if m16<e else 'EyeTraxの勝ち'}**: EyeTrax={e:.2f}cm vs 16D={m16:.2f}cm (差{e-m16:+.2f}cm, {(e-m16)/e*100:+.0f}%)")
