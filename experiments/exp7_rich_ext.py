"""7時間耐久 実験7: 生ランドマーク478点から虹彩楕円フィット特徴を追加(16D→22D)。
横向きで虹彩は扁平・傾くので、楕円のアスペクト比・傾き角が横向きに効くか。線形Huber。16Dと比較。
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent.parent))
from rich16d import rich_16d_from_lms, _LEFT_IRIS, _RIGHT_IRIS
from raw_landmark_logger import load_raw_landmarks
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
CM = np.array([30.9, 17.4]); BINS = [(0,10),(10,20),(20,30),(30,90)]
REPORT = ROOT / "experiments" / "REPORT.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def euc(P, G): return np.linalg.norm((P - G) * CM, axis=1)
def binstats(e, yd): return [np.median(e[(yd>=lo)&(yd<hi)]) if ((yd>=lo)&(yd<hi)).sum() else np.nan for lo,hi in BINS]

def ellipse_feats(lms, idx, w, h):
    pts = np.array([[lms[i][0]*w, lms[i][1]*h] for i in idx], dtype=np.float32)
    try:
        (_,_), (MA, ma), ang = cv2.fitEllipse(pts)
        asp = min(MA, ma) / (max(MA, ma) + 1e-6)
        return [asp, np.sin(np.radians(2*ang)), np.cos(np.radians(2*ang))]
    except Exception:
        return [1.0, 0.0, 1.0]

def ext(lms, w, h):
    base = rich_16d_from_lms(lms, w, h)
    if base is None: return None
    Le = ellipse_feats(lms, _LEFT_IRIS, w, h); Re = ellipse_feats(lms, _RIGHT_IRIS, w, h)
    return np.concatenate([base, Le, Re]).astype(np.float32)

GOOD = ["20260716_130217", "20260717_165617", "20260717_174621"]
X22, Y = [], []
for sid in GOOD:
    d = load_raw_landmarks(ROOT/"logs"/f"session_{sid}_landmarks")
    for k in range(d["n"]):
        if not bool(d["has_target"][k]): continue
        w = int(d["img_w"][k]); h = int(d["img_h"][k])
        f = ext(d["landmarks"][k], w, h)
        if f is None or not np.isfinite(f).all(): continue
        X22.append(f); Y.append(d["target"][k])
X22 = np.array(X22, np.float32); Y = np.array(Y, np.float32); X16 = X22[:, :16]

def H(): return HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=800)
def loo(X, Y):
    uniq, ids = np.unique(np.round(Y,4), axis=0, return_inverse=True)
    P,G,YD = [],[],[]
    for p in np.unique(ids):
        te,tr = ids==p, ids!=p
        if tr.sum()<30: continue
        sc = StandardScaler().fit(X[tr]); a,b = sc.transform(X[tr]), sc.transform(X[te])
        mx = H().fit(a, Y[tr,0]); my = H().fit(a, Y[tr,1])
        P.append(np.column_stack([mx.predict(b), my.predict(b)])); G.append(Y[te]); YD.append(np.abs(np.degrees(X22[te,5])))
    P,G,YD = np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    return np.median(e), binstats(e,YD)

t0 = time.time()
log(f"\n---\n## 実験7: 虹彩楕円フィット追加(16D→22D)  全体|0-10 10-20 20-30 30+")
log(f"データ: {len(X22)}フレーム(生ランドマークから再計算)")
med, bv = loo(X16, Y); log(f"  16D baseline    : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
med, bv = loo(X22, Y); log(f"  22D(+虹彩楕円)   : {med:5.2f} | " + " ".join(f"{v:5.2f}" for v in bv))
# 楕円特徴のみ追加した効果を30+で見る
log(f"→ 22Dが16Dより横向き30+で良ければ虹彩楕円が有効。悪化なら16Dで打ち止め。")
log(f"\n(実験7 完了 {(time.time()-t0)/60:.1f}分)")
