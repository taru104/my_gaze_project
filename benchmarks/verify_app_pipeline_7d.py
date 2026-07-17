"""実アプリのクラス CalibrationPipeline を7D特徴で駆動し、
アプリHUDの loo_euc_cm がオフライン予測(3.12cm)を再現するか検証する。

入力は今日のセッションの生ランドマーク由来7D(=features.extract がライブで作る物と同一数式)。
これが通れば「features 7D化 + calibration差し替え」がエンドツーエンドで正しいと確認できる。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import CalibrationPipeline

SCREEN_CM_W, SCREEN_CM_H = 30.9, 17.4
CALIB_DISCARD = 1.0


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    d = np.load(f"logs/session_{sid}_rich16d.npz")
    X, y, ht, ts = d["X"][:, :7], d["y_norm"], d["has_target"], d["time_s"]
    m = ht.copy(); X, y, ts = X[m], y[m], ts[m]

    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    keep = np.zeros(len(X), bool)
    for p in np.unique(ids):
        sel = np.where(ids == p)[0]
        keep[sel[ts[sel] - ts[sel].min() >= CALIB_DISCARD]] = True
    X, y, ids = X[keep], y[keep], ids[keep]
    print(f"[data] session_{sid}: {len(X)} フレーム / {len(uniq)} 点 / 7D\n")

    # アプリと同じ流れ: collect_point を全フレーム → finalize → 内部 _compute_loo
    pipe = CalibrationPipeline()
    for feat, tgt, i in zip(X, y, ids):
        # pitch/yaw は feat[4],feat[5] に含まれるが、ref算出用に別途渡す(アプリと同じ)
        pipe.collect_point(feat, tgt, weight=1.0,
                           pitch_rad=float(feat[4]), yaw_rad=float(feat[5]))
    pipe.finalize()

    print("=== アプリ CalibrationPipeline の内部指標(HUDに出る値) ===")
    ex = pipe.loo_euc_x * SCREEN_CM_W
    ey = pipe.loo_euc_y * SCREEN_CM_H
    loo_cm = float(np.sqrt(ex ** 2 + ey ** 2))
    print(f"  loo_euc_cm         = {loo_cm:.3f} cm   (旧アプリ実測 9.128cm から改善したか?)")
    print(f"  loo_mgae_deg       = {pipe.loo_mgae:.3f}")
    print(f"  train_rmse(norm)   = {pipe.train_rmse:.4f}")
    print(f"  n_samples          = {pipe.n_samples}")

    # 独立検算: 点ごとLOO を外から回して median Euclidean も出す
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    P, G = [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        sc = StandardScaler().fit(X[tr])
        rx = Ridge(1.0).fit(sc.transform(X[tr]), y[tr, 0])
        ry = Ridge(1.0).fit(sc.transform(X[tr]), y[tr, 1])
        pr = np.column_stack([rx.predict(sc.transform(X[te])), ry.predict(sc.transform(X[te]))])
        P.append(pr); G.append(y[te])
    P, G = np.vstack(P), np.vstack(G)
    e = np.sqrt(((P - G)[:, 0] * SCREEN_CM_W) ** 2 + ((P - G)[:, 1] * SCREEN_CM_H) ** 2)
    print(f"\n=== 独立検算(点ごとLOO, フレーム単位) ===")
    print(f"  median Euclidean   = {np.median(e):.3f} cm")
    print(f"  mean               = {np.mean(e):.3f} cm")

    ok = loo_cm < 5.0
    print(f"\n{'[OK]' if ok else '[FAIL]'} アプリ経路の7Dキャリブが "
          f"{'5cm未満で機能' if ok else '想定外に悪い'}（旧9.13cm）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
