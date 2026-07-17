"""
【決定版】豊富特徴14D の効果を エンドツーエンドで検証。
両抽出(extract_rich_features.py と extract_rich_test.py)完了後に実行。

比較(全て正面キャリブ→bin別 median cm):
  - 現行ローカル(2D iris affine)          … ベースライン
  - 7D  hybrid (7D global + 7D local)
  - 14D hybrid (rich global + rich local)  … 本命

グローバルは big rich cache(split=0)で学習。X[:,:7]=7D, X=14D。
ローカル/評価は rich_test_cache(subj_id付き)で被験者別に実施。

Usage:
    .venv/Scripts/python.exe benchmarks/rich_hybrid_eval.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import euclidean_cm, POSE_BINS

ROOT = Path(__file__).parent.parent
BIG_RICH  = ROOT / "cache" / "rich_features_cache.npz"
TEST_RICH = ROOT / "cache" / "rich_test_cache.npz"


def make_mlp(seed=0):
    return make_pipeline(StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128,64,32), activation="relu", alpha=1e-4,
                     batch_size=512, learning_rate_init=1e-3, max_iter=300,
                     early_stopping=True, n_iter_no_change=15, random_state=seed))


def feat_2d(X):
    return np.column_stack([(X[:,0]+X[:,2])/2, (X[:,1]+X[:,3])/2])


def pose_mag(X):
    return np.sqrt(np.degrees(X[:,4])**2 + np.degrees(X[:,5])**2)


def hybrid_eval(Xt, yct, subj, gmodel, n_dims, tau=6.0, alpha=10.0, cal_ratio=0.10):
    """rich_test上で 正面キャリブ→bin別。gmodelはn_dims次元入力。"""
    mag = pose_mag(Xt)
    gp_full = gmodel.predict(Xt[:, :n_dims])
    per_bin = {b: [] for b in POSE_BINS}; all_euc = []
    for sid in np.unique(subj):
        m = subj == sid
        Xs, yc, ms, gp = Xt[m], yct[m], mag[m], gp_full[m]
        order = np.argsort(ms); n_cal = max(8, int(np.ceil(cal_ratio*len(Xs))))
        idx_cal, idx_rest = order[:n_cal], order[n_cal:]
        if len(idx_rest) < 10: continue
        sc = StandardScaler(); Xc = sc.fit_transform(Xs[idx_cal, :n_dims])
        Xe = sc.transform(Xs[idx_rest, :n_dims])
        r = Ridge(alpha=alpha).fit(Xc, yc[idx_cal]); local = r.predict(Xe)
        m_cal = np.percentile(ms[idx_cal], 90); me = ms[idx_rest]
        w = np.exp(-np.maximum(0.0, me-m_cal)/tau)[:, None]
        pred = w*local + (1-w)*gp[idx_rest]
        euc = euclidean_cm(pred, yc[idx_rest]); all_euc.append(euc)
        for lo, hi in POSE_BINS:
            bm = (me>=lo)&(me<hi)
            if bm.sum()>=5: per_bin[(lo,hi)].append(np.median(euc[bm]))
    rep = {b:(float(np.median(per_bin[b])) if per_bin[b] else None) for b in POSE_BINS}
    rep["all"] = float(np.median(np.concatenate(all_euc)))
    return rep


def baseline_local2d(Xt, yct, subj, cal_ratio=0.10):
    mag = pose_mag(Xt); per_bin={b:[] for b in POSE_BINS}; all_euc=[]
    for sid in np.unique(subj):
        m = subj==sid; Xs,yc,ms = Xt[m],yct[m],mag[m]
        order=np.argsort(ms); n_cal=max(8,int(np.ceil(cal_ratio*len(Xs))))
        idx_cal,idx_rest=order[:n_cal],order[n_cal:]
        if len(idx_rest)<10: continue
        F=feat_2d(Xs); D=np.column_stack([F[idx_cal],np.ones(n_cal)])
        A,*_=np.linalg.lstsq(D,yc[idx_cal],rcond=None)
        pred=np.column_stack([F[idx_rest],np.ones(len(idx_rest))])@A
        euc=euclidean_cm(pred,yc[idx_rest]); all_euc.append(euc)
        me=ms[idx_rest]
        for lo,hi in POSE_BINS:
            bm=(me>=lo)&(me<hi)
            if bm.sum()>=5: per_bin[(lo,hi)].append(np.median(euc[bm]))
    rep={b:(float(np.median(per_bin[b])) if per_bin[b] else None) for b in POSE_BINS}
    rep["all"]=float(np.median(np.concatenate(all_euc))); return rep


def prow(label, rep):
    s=f"  {label:<22}"
    for b in POSE_BINS:
        v=rep[b]; s+=f" {v:>6.2f}" if v is not None else "   -  "
    print(s+f"  | {rep['all']:.3f}")


def main():
    if not BIG_RICH.exists() or not TEST_RICH.exists():
        print(f"[待機] 必要キャッシュ未生成:")
        print(f"  big rich : {'OK' if BIG_RICH.exists() else '未'} {BIG_RICH.name}")
        print(f"  test rich: {'OK' if TEST_RICH.exists() else '未'} {TEST_RICH.name}")
        print(f"  → extract_rich_features.py と extract_rich_test.py を先に完了させる")
        return
    t0=time.time()
    db=np.load(str(BIG_RICH)); Xb,ycb,scb=db["X"],db["y_cm"],db["split_code"]
    tr=scb==0
    print(f"[Train] {tr.sum()} frames, dim={Xb.shape[1]}")
    nd = Xb.shape[1]   # 実次元(16D)
    g7   = make_mlp().fit(Xb[tr,:7], ycb[tr]); print(f"  7D global fit ({time.time()-t0:.0f}s)")
    grich= make_mlp().fit(Xb[tr],    ycb[tr]); print(f"  {nd}D global fit ({time.time()-t0:.0f}s)")

    dt=np.load(str(TEST_RICH)); Xt,yct,subj=dt["X"],dt["y_cm"],dt["subj_id"]
    print(f"[Test] {len(Xt)} frames, {len(np.unique(subj))} subjects\n")

    hdr=f"  {'method':<22}"
    for lo,hi in POSE_BINS: hdr+=f" {lo:>2}-{hi:<3}"
    print("="*92); print(hdr+"  |  all"); print("  "+"-"*88)
    prow("現行ローカル2D", baseline_local2d(Xt,yct,subj))
    prow("7D hybrid",      hybrid_eval(Xt,yct,subj,g7,7))
    prow(f"{nd}D rich hybrid",hybrid_eval(Xt,yct,subj,grich,nd))
    print("  "+"-"*88)
    print(f"\n[{time.time()-t0:.0f}s] 14D richが7Dより横向きbinで低ければ豊富特徴の勝ち")


if __name__ == "__main__":
    main()
