"""MPIIFaceGaze 15人で「汎用性」を評価する。True Gaze の核心テスト。

3段階:
  (1) person-independent (他人・キャリブ無し)  … 純粋な汎用性。1人を除いて14人で学習→除いた人を予測。
  (2) 汎用ベース + 個人少数キャリブ(適応)       … 「誰でもそこそこ + キャリブで最高」のバランス。
  (3) 個人フルキャリブ (上限の目安)             … その人のデータだけで学習。

指標: 画面正規化[0,1]の Euclidean 誤差。cm近似は MPII ノートPC を平均約 30×19cm と仮定して併記。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
CM = np.array([30.0, 19.0])  # MPII ノートPC 平均画面(近似)

def load():
    path = ROOT / "cache" / "mpii_7d.npz"
    if "--data" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--data") + 1])
    d = np.load(path)
    return d["X"], d["y"], d["pid"]

def euc_norm(P, G): return np.linalg.norm(P - G, axis=1)
def euc_cm(P, G):   return np.linalg.norm((P - G) * CM, axis=1)

def fit(Xtr, ytr, kind="huber"):
    sc = StandardScaler().fit(Xtr)
    R = (lambda: HuberRegressor(max_iter=500)) if kind == "huber" else (lambda: Ridge(1.0))
    mx = R().fit(sc.transform(Xtr), ytr[:, 0]); my = R().fit(sc.transform(Xtr), ytr[:, 1])
    return lambda Xte: np.column_stack([mx.predict(sc.transform(Xte)), my.predict(sc.transform(Xte))])

def affine_adapt(pred_base_tr, ytr, pred_base_te):
    # 汎用予測を個人の少数点で線形補正(2x3アフィン)
    A = np.hstack([pred_base_tr, np.ones((len(pred_base_tr), 1))])
    W, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.hstack([pred_base_te, np.ones((len(pred_base_te), 1))]) @ W

def main():
    X, y, pid = load()
    persons = sorted(set(pid.tolist()))
    print(f"MPII 15人 {len(X)}フレーム / 被験者{len(persons)}人\n")

    # (1) person-independent
    print("=== (1) person-independent (他人・キャリブ無し) ===")
    errs_ni = []
    base_preds = {}
    for p in persons:
        te = pid == p; tr = ~te
        f = fit(X[tr], y[tr])
        pr = f(X[te]); base_preds[p] = pr
        e = euc_cm(pr, y[te])
        errs_ni.append(np.median(e))
    print(f"  median誤差(cm近似) 全人平均={np.mean(errs_ni):.2f}  中央={np.median(errs_ni):.2f}  "
          f"最良={np.min(errs_ni):.2f} 最悪={np.max(errs_ni):.2f}")

    # (2) 汎用ベース + 少数キャリブ適応
    print("\n=== (2) 汎用ベース + 個人Nキャリブ点で適応 ===")
    rng = np.random.RandomState(0)
    for ncal in [9, 25, 50]:
        errs = []
        for p in persons:
            te = pid == p
            idx = np.where(te)[0]
            if len(idx) < ncal + 20: continue
            cal = rng.choice(idx, ncal, replace=False)
            evalmask = np.setdiff1d(idx, cal)
            # 汎用ベース(他14人)予測
            tr = pid != p
            f = fit(X[tr], y[tr])
            pb_cal = f(X[cal]); pb_ev = f(X[evalmask])
            pr = affine_adapt(pb_cal, y[cal], pb_ev)
            errs.append(np.median(euc_cm(pr, y[evalmask])))
        print(f"  {ncal:2d}点適応: median誤差(cm近似) 平均={np.mean(errs):.2f}  中央={np.median(errs):.2f}")

    # (3) 個人フル(上限目安)
    print("\n=== (3) 個人フルキャリブ(その人のデータで5-fold, 上限目安) ===")
    from sklearn.model_selection import KFold
    errs_full = []
    for p in persons:
        idx = np.where(pid == p)[0]
        if len(idx) < 50: continue
        Xp, yp = X[idx], y[idx]
        kf = KFold(5, shuffle=True, random_state=0); pr = np.zeros_like(yp)
        for tr, te in kf.split(Xp):
            f = fit(Xp[tr], yp[tr]); pr[te] = f(Xp[te])
        errs_full.append(np.median(euc_cm(pr, yp)))
    print(f"  median誤差(cm近似) 平均={np.mean(errs_full):.2f}  中央={np.median(errs_full):.2f}")

    print("\n→ (1)が汎用性の生値、(2)が実用(少しキャリブ)、(3)が個人上限。")
    print("  (2)が(1)から大きく下がれば『汎用ベース+軽いキャリブ』戦略が有効。")

if __name__ == "__main__":
    main()
