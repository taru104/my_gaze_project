"""アプリのローカルキャリブが plain Ridge の3倍悪い原因を切り分ける。

疑い1: モデルが過学習している（TargetedPolyCalibration の Y^2, Y*pitch 項）
疑い2: 入力特徴が貧しい（アプリは [X_feat, Y_feat, pitch] の2特徴のみ。
       7D は [Lx,Ly,Rx,Ry,pitch,yaw,dist] で両目を別々に持ち yaw/dist もある）

同一セッションの
  - CSV      : アプリが実際に使った X_feat/Y_feat/pitch
  - rich16d  : 生ランドマークから再計算した 16D
を、どちらも**アプリと同一指標**(点ごとLOO→点内中央値→軸別平均→cm合成)で測る。

Usage: .venv/Scripts/python.exe benchmarks/diagnose_local_calib.py <session_id>
"""
import sys, csv
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import TargetedPolyCalibration, PolyRidgeCalibration, AffineCalibration

SCREEN_CM_W, SCREEN_CM_H = 30.9, 17.4
CALIB_DISCARD = 1.0


def app_metric(pred_by_point):
    """{point: (pred_med(2,), target(2,))} → アプリと同一の loo_euc_cm"""
    ex = np.mean([abs(p[0] - t[0]) for p, t in pred_by_point.values()])
    ey = np.mean([abs(p[1] - t[1]) for p, t in pred_by_point.values()])
    return float(np.sqrt((ex * SCREEN_CM_W) ** 2 + (ey * SCREEN_CM_H) ** 2))


def load_csv(sid):
    p = Path("logs") / f"session_{sid}.csv"
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["calib_target_x"] and r["X_feat"] and r["pitch_deg"]]
    X = np.array([[float(r["X_feat"]), float(r["Y_feat"]),
                   np.radians(float(r["pitch_deg"]))] for r in rows])
    y = np.array([[float(r["calib_target_x"]), float(r["calib_target_y"])] for r in rows])
    t = np.array([float(r["time_s"]) for r in rows])
    return X, y, t


def discard_early(y, t):
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    keep = np.zeros(len(y), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        keep[sel[t[sel] - t[sel].min() >= CALIB_DISCARD]] = True
    return uniq, ids, keep


def loo_app_model(make, X, y, uniq, ids):
    """アプリのキャリブclass(add/fit/predict I/F)を点ごとLOOで評価。"""
    out = {}
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        m = make()
        for f, tg in zip(X[tr], y[tr]):
            m.add(float(f[0]), float(f[1]), float(tg[0]), float(tg[1]),
                  1.0, pitch_rad=float(f[2]))
        m.fit()
        preds = np.array([m.predict(float(f[0]), float(f[1]), float(f[2])) for f in X[te]])
        out[p] = (np.median(preds, axis=0), uniq[p])
    return out


def loo_ridge(X, y, uniq, ids, alpha=1.0):
    from sklearn.linear_model import Ridge
    out = {}
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        m = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, y[tr])
        preds = m.predict((X[te] - mu) / sd)
        out[p] = (np.median(preds, axis=0), uniq[p])
    return out


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"

    print("=" * 68)
    print("アプリが実際に使った特徴 [X_feat, Y_feat, pitch] での比較")
    print("=" * 68)
    Xc, yc, tc = load_csv(sid)
    uniq, ids, keep = discard_early(yc, tc)
    Xc, yc, ids = Xc[keep], yc[keep], ids[keep]
    print(f"  {len(Xc)} フレーム / {len(uniq)} 点\n")

    r = loo_app_model(TargetedPolyCalibration, Xc, yc, uniq, ids)
    print(f"  アプリ本番 TargetedPolyCalibration      : {app_metric(r):6.3f} cm")
    r = loo_app_model(PolyRidgeCalibration, Xc, yc, uniq, ids)
    print(f"  PolyRidgeCalibration (2次10param)       : {app_metric(r):6.3f} cm")
    r = loo_app_model(AffineCalibration, Xc, yc, uniq, ids)
    print(f"  AffineCalibration (線形・最小)          : {app_metric(r):6.3f} cm")
    r = loo_ridge(Xc[:, :2], yc, uniq, ids)
    print(f"  plain Ridge [X_feat,Y_feat] のみ        : {app_metric(r):6.3f} cm")
    r = loo_ridge(Xc, yc, uniq, ids)
    print(f"  plain Ridge [X_feat,Y_feat,pitch]       : {app_metric(r):6.3f} cm")

    print()
    print("=" * 68)
    print("生ランドマーク由来の特徴での比較（同一セッション・同一指標）")
    print("=" * 68)
    d = np.load(f"logs/session_{sid}_rich16d.npz")
    Xr, yr, ht, tr_ = d["X"], d["y_norm"], d["has_target"], d["time_s"]
    m = ht.copy(); Xr, yr, tr_ = Xr[m], yr[m], tr_[m]
    uniq2, ids2, keep2 = discard_early(yr, tr_)
    Xr, yr, ids2 = Xr[keep2], yr[keep2], ids2[keep2]
    print(f"  {len(Xr)} フレーム / {len(uniq2)} 点\n")

    sets = [
        ("7D [Lx,Ly,Rx,Ry,pitch,yaw,dist]", Xr[:, :7]),
        ("16D (全部)",                        Xr),
        ("両目の虹彩4D [Lx,Ly,Rx,Ry] のみ",   Xr[:, :4]),
        ("片目2D [Lx,Ly] のみ",               Xr[:, :2]),
        ("4D虹彩 + pitch,yaw",                Xr[:, [0, 1, 2, 3, 4, 5]]),
    ]
    for name, Xa in sets:
        r = loo_ridge(Xa, yr, uniq2, ids2)
        print(f"  plain Ridge {name:34s}: {app_metric(r):6.3f} cm")


if __name__ == "__main__":
    main()
