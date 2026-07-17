"""ユーザ実webカメラ録画(生ランドマーク由来16D)で実精度を測る。

入力: logs/session_<id>_rich16d.npz (reprocess_raw_landmarks.py の出力)
正解付きフレーム(9点キャリブ中)だけを使い、**点ごとのleave-one-point-out**で評価する。
同一点内のフレームは強く相関するので、フレーム単位LOOだと精度を過大評価する
（アプリHUDのloo_euc_cmはフレーム単位＝甘い）。

比較:
  A) 7D  + ローカルRidge   （現行アプリ相当）
  B) 16D + ローカルRidge   （特徴だけ増やす）
  C) 16D + グローバルMLP prior → cm→正規化のアフィン個人補正（勝ち筋の座標系ブリッジ）
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
from sklearn.linear_model import Ridge

SCREEN_CM_W, SCREEN_CM_H = 30.9, 17.4
CALIB_DISCARD = 1.0   # 各点の最初1秒は破棄(アプリと同条件)


def euc_cm(pred, gt):
    d = pred - gt
    return np.sqrt((d[:, 0] * SCREEN_CM_W) ** 2 + (d[:, 1] * SCREEN_CM_H) ** 2)


def report(name, err):
    print(f"  {name:34s} median={np.median(err):6.3f}  mean={np.mean(err):6.3f}  "
          f"p90={np.percentile(err, 90):6.3f}  (n={len(err)})")


def group_by_point(y):
    """正解座標のユニーク値で点IDを振る。"""
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    return uniq, ids


def loo_point(X, y, ids, fit_predict):
    preds, gts = [], []
    for p in np.unique(ids):
        te = ids == p
        tr = ~te
        if tr.sum() < 10:
            continue
        preds.append(fit_predict(X[tr], y[tr], X[te]))
        gts.append(y[te])
    return np.vstack(preds), np.vstack(gts)


def ridge_fp(alpha=1.0):
    def f(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr)
        return m.predict((Xte - mu) / sd)
    return f


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        cands = sorted(Path("logs").glob("session_*_rich16d.npz"))
        if not cands:
            print("no rich16d npz found"); return
        path = str(cands[-1])
    d = np.load(path)
    X, y, ht, ts = d["X"], d["y_norm"], d["has_target"], d["time_s"]

    m = ht.copy()
    X, y, ts = X[m], y[m], ts[m]
    print(f"[data] {path}")
    print(f"  正解付きフレーム: {len(X)}")

    uniq, ids = group_by_point(y)
    print(f"  キャリブ点数: {len(uniq)}")
    for p in np.unique(ids):
        print(f"    point{p} target=({uniq[p][0]:.2f},{uniq[p][1]:.2f}) frames={int((ids==p).sum())}")

    # 各点の最初1秒を破棄(アプリと同条件: 視線が到達する前の遷移を除く)
    keep = np.zeros(len(X), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        t0 = ts[sel].min()
        keep[sel[ts[sel] - t0 >= CALIB_DISCARD]] = True
    X, y, ids = X[keep], y[keep], ids[keep]
    print(f"  破棄後: {len(X)} フレーム\n")

    print("[点ごと leave-one-point-out] Euc(cm) 低いほど良い")
    P, G = loo_point(X[:, :7], y, ids, ridge_fp())
    report("A) 7D  + ローカルRidge", euc_cm(P, G))
    P, G = loo_point(X, y, ids, ridge_fp())
    report("B) 16D + ローカルRidge", euc_cm(P, G))

    # 参考: 常に画面中央を返すだけのベースライン
    P = np.tile([0.5, 0.5], (len(y), 1))
    report("baseline) 常に画面中央", euc_cm(P, y))

    # C) グローバルMLP prior + アフィン個人補正
    mp = Path("cache/global_mlp_16d.joblib")
    if not mp.exists():
        print("\n  [skip] cache/global_mlp_16d.joblib が無い → C)未評価")
        return
    import joblib
    obj = joblib.load(mp)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    scaler = obj.get("scaler") if isinstance(obj, dict) else None
    Xs = scaler.transform(X) if scaler is not None else X
    prior_cm = model.predict(Xs)          # GazeCapture cm 空間
    print(f"\n[global MLP] prior_cm range x[{prior_cm[:,0].min():.1f},{prior_cm[:,0].max():.1f}] "
          f"y[{prior_cm[:,1].min():.1f},{prior_cm[:,1].max():.1f}]")

    def affine_fp(Xtr, ytr, Xte):
        # Xtr/Xte は prior_cm。cm → 画面正規化 のアフィン(個人キャリブ相当)
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        W, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return np.hstack([Xte, np.ones((len(Xte), 1))]) @ W

    P, G = loo_point(prior_cm, y, ids, affine_fp)
    report("C) 16D globalMLP + アフィン補正", euc_cm(P, G))

    # C2) prior_cm を16Dに足してローカルRidge(ハイブリッド近似)
    Xh = np.hstack([X, prior_cm])
    P, G = loo_point(Xh, y, ids, ridge_fp())
    report("C2) 16D+prior_cm + ローカルRidge", euc_cm(P, G))


if __name__ == "__main__":
    main()
