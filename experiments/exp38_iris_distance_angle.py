"""exp38: 虹彩の実サイズ(約11.7mm)からカメラまでの実距離を計測し、16Dの誤差を正確なcmと角度(°)で計算。
今まで距離50cm仮定で角度換算していたのを、虹彩から実測して正確化。SOTA(角度)と正しく比較する。
距離 = 虹彩実径(mm) × 焦点距離(px) / 虹彩ピクセル径。焦点距離f_px≈img_w(近似)。mainは触らない。
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

SW, SH = 30.9, 17.4   # ユーザ画面(推定) 幅/高さ cm
IRIS_MM = 11.7        # ヒト虹彩の水平径(HVID, 個人差ほぼゼロ)
REPORT = ROOT / "experiments" / "REPORT4_allmethods.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

def iris_diam_px(P, w, h):
    # 水平径: 左目469-471, 右目474-476 (features.py LEFT/RIGHT_IRIS_H_EDGES)
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

all_dist=[]; sessions=[]
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
        dist_cm=(IRIS_MM/10.0)*w/diam    # f_px≈w(近似)
        try: f16=rich_16d_from_lms(d["landmarks"][k],int(w),int(h))
        except Exception: f16=None
        if f16 is None: continue
        pts.append(dict(f16=np.asarray(f16,float), tgt=np.asarray(t,float), dist=dist_cm,
                        yaw=abs(np.degrees(float(f16[5])))))
        all_dist.append(dist_cm)
    if len(pts)>=60: sessions.append(pts)

log("\n---\n## exp38: 虹彩から実距離を計測→正確なcm/角度で誤差計算")
ad=np.array(all_dist)
log(f"\n**虹彩から計測した顔-カメラ距離(全{len(ad)}点, 虹彩{IRIS_MM}mm, f≈img_w近似)**")
log(f"  median={np.median(ad):.1f}cm  範囲[{np.percentile(ad,10):.1f}, {np.percentile(ad,90):.1f}]cm  (妥当なら40-70cm付近)")

# 16D実機シナリオ(補間+平滑)の誤差を、各点の実距離で角度換算
rng=np.random.RandomState(0)
cm_all=[]; ang_all=[]; ang_by=defaultdict(list)
YB=[(0,10),(10,20),(20,30),(30,90)]
for pts in sessions:
    n=len(pts); order=rng.permutation(n); cut=int(n*0.7)
    tr=order[:cut]; te=order[cut:]
    Xtr=np.array([pts[i]["f16"] for i in tr]); Ytr=np.array([pts[i]["tgt"] for i in tr])
    Xte=np.array([pts[i]["f16"] for i in te]); Yte=np.array([pts[i]["tgt"] for i in te])
    keys=[(round(pts[i]["tgt"][0],2),round(pts[i]["tgt"][1],2)) for i in te]
    pred=smooth(fit_predict(Xtr,Ytr,Xte), keys, 5)
    ecm=euc_cm(pred,Yte)
    for j,i in enumerate(te):
        dist=pts[i]["dist"]
        ang=np.degrees(np.arctan(ecm[j]/dist))   # 角度誤差 = atan(画面cm誤差 / 顔-画面距離)
        cm_all.append(ecm[j]); ang_all.append(ang)
        for (lo,hi) in YB:
            if lo<=pts[i]["yaw"]<hi: ang_by[(lo,hi)].append(ang); break
log(f"\n**16D実機シナリオ 誤差(虹彩実距離で正確計算)**")
log(f"  距離誤差: median={np.median(cm_all):.2f}cm")
log(f"  ★角度誤差: median={np.median(ang_all):.2f}° (虹彩実距離ベース。従来の50cm仮定より正確)")
log(f"  姿勢bin角度: " + " ".join(f"{b[0]}-{b[1]}°={np.median(ang_by[b]):.2f}°" for b in YB if ang_by[b]))
log(f"\n**SOTA(角度)との正確な比較**")
log(f"  あなた(個人キャリブ,リアルタイム,実環境)={np.median(ang_all):.2f}° / L2CS(person-indep)=3.92° / MPIIFaceGaze SOTA GEM=2.32° / 個人キャリブ理想=1.1-2.2°")
