"""目の重み探索: 横向きで「手前の目(カメラに近い側)」を使うと横向きが改善するか。
次元は増やさない(むしろ減らす)・過学習を避けるため点ごとLOO・姿勢bin別。合算多姿勢データ。

特徴セット(全て Huber回帰):
  both7D  : [Lx,Ly,Rx,Ry,pitch,yaw,dist]        現状H1のベース相当
  left5D  : [Lx,Ly,pitch,yaw,dist]              左目のみ
  right5D : [Rx,Ry,pitch,yaw,dist]              右目のみ
  front5D : yaw>0→右目, yaw<0→左目 の手前目 + [pitch,yaw,dist]   ★横向きで手前目
  weighted: 両目を yaw で連続重み付けブレンドした虹彩2D + [pitch,yaw,dist]
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SCREEN = np.array([30.9,17.4]); BINS=[(0,10),(10,20),(20,30),(30,90)]
ROOT = Path(__file__).parent.parent / "logs"
GOOD = ["20260716_130217","20260717_165617","20260717_174621"]

def euc(P,G): return np.sqrt(((P-G)[:,0]*SCREEN[0])**2+((P-G)[:,1]*SCREEN[1])**2)

def load(sid):
    d=np.load(ROOT/f"session_{sid}_rich16d.npz"); m=d["has_target"].astype(bool)
    return d["X"][m][:,:7], d["y_norm"][m]

def feat_transform(X, kind):
    yaw = X[:,5]
    pyd = X[:,4:7]
    if kind=="both7D":  return X[:,:7]
    if kind=="left5D":  return np.hstack([X[:,[0,1]], pyd])
    if kind=="right5D": return np.hstack([X[:,[2,3]], pyd])
    if kind=="front5D":
        # yaw>0で右目[2,3], それ以外で左目[0,1](手前側)
        iris = np.where((yaw>0)[:,None], X[:,[2,3]], X[:,[0,1]])
        return np.hstack([iris, pyd])
    if kind=="weighted":
        # yawで左右を連続ブレンド(手前目に大きい重み)。sigmoid的
        wr = 1/(1+np.exp(-yaw*6))          # yaw>0で右に寄る
        iris = np.column_stack([wr*X[:,2]+(1-wr)*X[:,0], wr*X[:,3]+(1-wr)*X[:,1]])
        return np.hstack([iris, pyd])
    raise ValueError(kind)

def fit_pred(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr)
    hx=HuberRegressor(max_iter=800).fit(sc.transform(Xtr),ytr[:,0])
    hy=HuberRegressor(max_iter=800).fit(sc.transform(Xtr),ytr[:,1])
    return np.column_stack([hx.predict(sc.transform(Xte)),hy.predict(sc.transform(Xte))])

def evaluate(Xraw,y,kind):
    uniq,ids=np.unique(np.round(y,4),axis=0,return_inverse=True)
    F=feat_transform(Xraw,kind)
    P,G,YD=[],[],[]
    for p in np.unique(ids):
        te,tr=ids==p,ids!=p
        if tr.sum()<20: continue
        P.append(fit_pred(F[tr],y[tr],F[te])); G.append(y[te]); YD.append(np.abs(np.degrees(Xraw[te][:,5])))
    P,G,YD=np.vstack(P),np.vstack(G),np.concatenate(YD); e=euc(P,G)
    bv=[np.median(e[(YD>=lo)&(YD<hi)]) if ((YD>=lo)&(YD<hi)).sum() else np.nan for lo,hi in BINS]
    return np.median(e),bv

Xs,ys=[],[]
for sid in GOOD:
    try: X,y=load(sid); Xs.append(X); ys.append(y)
    except FileNotFoundError: pass
Xa,ya=np.vstack(Xs),np.vstack(ys)
print(f"合算 {len(Xa)}フレーム  点ごとLOO 姿勢bin別 median Euc(cm)")
print(f"{'特徴':10s} {'frame':>6s}  |y|0-10 |y|10-20 |y|20-30 |y|30+")
for kind in ["both7D","left5D","right5D","front5D","weighted"]:
    fr,bv=evaluate(Xa,ya,kind)
    print(f"{kind:10s} {fr:6.2f}  "+" ".join(f"{v:5.2f}" for v in bv))
print("\n★front5D/weighted が both7D より 30+ で良ければ『手前目を使う』が有効")
