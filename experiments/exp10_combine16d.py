"""追加実験(ユーザ指摘対応): MPII(16D)とユーザデータの「併用」を検証。
exp9で抽出したMPII 16D で汎用モデルを作り、ユーザデータに: A)そのまま B)少数アフィン適応 で適用。
C)ユーザ個人16D(LOO)と比較。7Dで前に破綻した併用が16Dでどうか。cache/mpii_16d_ck.npz が必要(exp9生成)。
"""
import sys, time, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
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

# MPII 16D 汎用モデル
d = np.load(ROOT/"cache"/"mpii_16d_ck.npz")
Xm, ym = d["X"], d["y"]
sc = StandardScaler().fit(Xm)
gmx = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:,0])
gmy = HuberRegressor(max_iter=800).fit(sc.transform(Xm), ym[:,1])
def gen(X): return np.column_stack([gmx.predict(sc.transform(X)), gmy.predict(sc.transform(X))])
def affine(bt, yt, be):
    A = np.hstack([bt, np.ones((len(bt),1))]); W,*_ = np.linalg.lstsq(A, yt, rcond=None)
    return np.hstack([be, np.ones((len(be),1))]) @ W

t0 = time.time()
log(f"\n---\n## 実験10(併用検証): MPII 16D汎用 → ユーザデータ  全体|0-10 10-20 20-30 30+")
log(f"MPII 16D汎用モデル: {len(Xm)}フレーム/15人")
rng = np.random.RandomState(0)
for f in sorted(glob.glob(str(ROOT/"logs"/"session_*_rich16d.npz"))):
    dd = np.load(f); m = dd["has_target"].astype(bool)
    Xu, yu = dd["X"][m][:, :16], dd["y_norm"][m]
    if len(Xu) < 300: continue
    yd = np.abs(np.degrees(Xu[:,5]))
    name = Path(f).name.replace("session_","").replace("_rich16d.npz","")
    # A: 汎用そのまま
    a = euc(gen(Xu), yu)
    # B: 汎用+50点アフィン(点分割: 先頭50点で適応, 残りで評価)
    base = gen(Xu); idx = rng.permutation(len(Xu)); cal, ev = idx[:50], idx[50:]
    b = euc(affine(base[cal], yu[cal], base[ev]), yu[ev]); ydb = yd[ev]
    # C: 個人16D Huber(点ごとLOO)
    uniq, ids = np.unique(np.round(yu,4), axis=0, return_inverse=True)
    Pc, Gc, YDc = [], [], []
    for p in np.unique(ids):
        te, tr = ids==p, ids!=p
        if tr.sum()<30: continue
        s2 = StandardScaler().fit(Xu[tr])
        mx = HuberRegressor(max_iter=800).fit(s2.transform(Xu[tr]), yu[tr,0])
        my = HuberRegressor(max_iter=800).fit(s2.transform(Xu[tr]), yu[tr,1])
        Pc.append(np.column_stack([mx.predict(s2.transform(Xu[te])), my.predict(s2.transform(Xu[te]))])); Gc.append(yu[te]); YDc.append(yd[te])
    Pc,Gc,YDc = np.vstack(Pc),np.vstack(Gc),np.concatenate(YDc); c = euc(Pc,Gc)
    log(f"[{name}]")
    log(f"  A:MPII16D汎用そのまま : {np.median(a):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(a, yd)))
    log(f"  B:MPII16D汎用+50点適応: {np.median(b):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(b, ydb)))
    log(f"  C:個人16D(LOO)       : {np.median(c):5.2f} | " + " ".join(f"{v:5.2f}" for v in binstats(c, YDc)))
log(f"→ B(汎用+適応)がC(個人)に匹敵/凌駕すれば、MPII16D併用が有効=1人データ依存を脱せる。")
log(f"\n(実験10 完了 {(time.time()-t0)/60:.1f}分)")
