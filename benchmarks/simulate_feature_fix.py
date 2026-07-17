"""「features.py を直したらアプリは本当に良くなるのか」を既存データで事前検証する。

前回の 3.12cm は plain Ridge on 7D＝**アプリのパイプラインではない**。
ここでは **アプリの本物のクラス**(TargetedPolyCalibration / CalibrationPipeline)を
そのまま使い、**入力特徴だけ**を差し替えて、直した後に何cmになるかを予測する。

修正案:
  案A (最小改修): 2D I/F を維持。基準点だけ画像中心→目頭・目尻中点に変える。
        X_feat = (L_n[0] + R_n[0]) / 2,  Y_feat = (L_n[1] + R_n[1]) / 2
        → calibration.py / estimator.py を一切触らずに済む。
  案B (7D化):     両目を別々に + yaw + dist。パイプラインのベクトル化が必要。

指標はアプリHUDの loo_euc_cm と同一定義（点ごとLOO→点内中央値→軸別平均→cm合成）。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from extract_rich_features import (
    _FACE_3D_MODEL, _FACE_2D_IDX, _LEFT_IRIS, _RIGHT_IRIS,
    _L_IN, _L_OUT, _R_IN, _R_OUT, _DIST, _geo_normalize,
)
from raw_landmark_logger import load_raw_landmarks
from calibration import TargetedPolyCalibration
from diagnose_local_calib import app_metric, discard_early, loo_ridge, loo_app_model


def build(lms, w, h):
    """同一フレームから アプリ方式2D / 案A 2D / 案B 7D を同時に作る。"""
    def P(i):
        return np.array([lms[i][0] * w, lms[i][1] * h])
    L_px = np.mean([P(i) for i in _LEFT_IRIS], axis=0)
    R_px = np.mean([P(i) for i in _RIGHT_IRIS], axis=0)

    f = float(w)
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    face_2d = np.array([[lms[i][0] * w, lms[i][1] * h] for i in _FACE_2D_IDX], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) * np.pi / 180.0
    yaw = float(angles[1]) * np.pi / 180.0
    if pitch > np.pi / 2: pitch = np.pi - pitch
    elif pitch < -np.pi / 2: pitch = -np.pi - pitch

    L_d = 2.0 * np.mean([np.linalg.norm(P(i) - L_px) for i in _LEFT_IRIS[1:]])
    R_d = 2.0 * np.mean([np.linalg.norm(P(i) - R_px) for i in _RIGHT_IRIS[1:]])
    if L_d < 2.0 or R_d < 2.0:
        return None
    cx, cy = w / 2.0, h / 2.0
    now = np.array([((L_px[0] - cx) / L_d + (R_px[0] - cx) / R_d) / 2.0,
                    ((L_px[1] - cy) / L_d + (R_px[1] - cy) / R_d) / 2.0, pitch])

    L_n = _geo_normalize(L_px, P(_L_IN), P(_L_OUT))
    R_n = _geo_normalize(R_px, P(_R_IN), P(_R_OUT))
    fixA = np.array([(L_n[0] + R_n[0]) / 2.0, (L_n[1] + R_n[1]) / 2.0, pitch])
    dist = float(np.linalg.norm(L_px - R_px) / w)
    fixB = np.array([L_n[0], L_n[1], R_n[0], R_n[1], pitch, yaw, dist])
    return now, fixA, fixB


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    d = load_raw_landmarks(f"logs/session_{sid}_landmarks")
    NOW, A, B, Y, T = [], [], [], [], []
    for k in range(d["n"]):
        if not bool(d["has_target"][k]):
            continue
        r = build(d["landmarks"][k], int(d["img_w"][k]), int(d["img_h"][k]))
        if r is None or not all(np.isfinite(x).all() for x in r):
            continue
        NOW.append(r[0]); A.append(r[1]); B.append(r[2])
        Y.append(d["target"][k]); T.append(float(d["time_s"][k]))
    NOW, A, B, Y, T = map(np.array, (NOW, A, B, Y, T))
    uniq, ids, keep = discard_early(Y, T)
    NOW, A, B, Y, ids = NOW[keep], A[keep], B[keep], Y[keep], ids[keep]
    print(f"[data] session_{sid}: {len(Y)} フレーム / {len(uniq)} 点")
    print("  ※ 同一フレームから3方式を同時に計算＝特徴以外の条件は完全に同一\n")

    print("=" * 70)
    print("アプリの本物のモデル TargetedPolyCalibration に入れた場合")
    print("=" * 70)
    r = loo_app_model(TargetedPolyCalibration, NOW, Y, uniq, ids)
    base = app_metric(r)
    print(f"  現状（画像中心基準 2D）                : {base:6.3f} cm")
    r = loo_app_model(TargetedPolyCalibration, A, Y, uniq, ids)
    a = app_metric(r)
    print(f"  案A 最小改修（目頭基準 2D・両目平均）   : {a:6.3f} cm   ({base/a:.2f}倍改善)")

    print()
    print("=" * 70)
    print("案B 7D化（モデルは plain Ridge。パイプラインのベクトル化が必要）")
    print("=" * 70)
    r = loo_ridge(B, Y, uniq, ids)
    b = app_metric(r)
    print(f"  案B 7D + plain Ridge                  : {b:6.3f} cm   ({base/b:.2f}倍改善)")
    r = loo_ridge(A, Y, uniq, ids)
    print(f"  参考: 案A特徴 + plain Ridge            : {app_metric(r):6.3f} cm")

    print()
    print("=" * 70)
    print("案A の頑健性チェック: 頭部姿勢 |yaw| 別 (median Euclidean, 点ごとLOO)")
    print("=" * 70)
    SCREEN = np.array([30.9, 17.4])
    for nm, F, use_app in [("現状", NOW, True), ("案A", A, True), ("案B(Ridge)", B, False)]:
        P, G, YW = [], [], []
        for p in np.unique(ids):
            te, tr = ids == p, ids != p
            if use_app:
                m = TargetedPolyCalibration()
                for f_, tg in zip(F[tr], Y[tr]):
                    m.add(float(f_[0]), float(f_[1]), float(tg[0]), float(tg[1]), 1.0,
                          pitch_rad=float(f_[2]))
                m.fit()
                pr = np.array([m.predict(float(f_[0]), float(f_[1]), float(f_[2])) for f_ in F[te]])
            else:
                from sklearn.linear_model import Ridge
                mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
                mm = Ridge(alpha=1.0).fit((F[tr] - mu) / sd, Y[tr])
                pr = mm.predict((F[te] - mu) / sd)
            P.append(pr); G.append(Y[te]); YW.append(B[te][:, 5])
        P, G, YW = np.vstack(P), np.vstack(G), np.concatenate(YW)
        e = np.linalg.norm((P - G) * SCREEN, axis=1)
        yd = np.abs(np.degrees(YW))
        s = "  %-11s" % nm
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 90)]:
            k = (yd >= lo) & (yd < hi)
            s += f"  |yaw|{lo}-{hi}: {np.median(e[k]):5.2f}" if k.sum() else ""
        print(s)


if __name__ == "__main__":
    main()
