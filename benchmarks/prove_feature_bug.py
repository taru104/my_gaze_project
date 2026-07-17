"""アプリの X_feat/Y_feat が「視線」ではなく「顔が画面内のどこに居るか」を測っている証明。

アプリ (features.py:302):
    X_feat = (iris_x_画像座標 - 画像中心x) / 虹彩直径
  → 虹彩直径で割るので"距離"不変にはなっている。だが基準点が**画像中心**なので、
    顔を平行移動しただけで視線が1mmも動かなくても X_feat は大きく動く。

7D/16D (_geo_normalize):
    虹彩を**目頭・目尻の中点**基準にして、目の軸で回転、目幅で正規化
  → 「眼球の中で虹彩がどこにあるか」＝本当の視線信号。頭の平行移動に不変。

同じ生ランドマークから両方を計算して直接対決させる（特徴以外の条件は完全同一）。
"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from extract_rich_features import _LEFT_IRIS, _RIGHT_IRIS, _L_IN, _L_OUT, _R_IN, _R_OUT, _geo_normalize
from raw_landmark_logger import load_raw_landmarks
from diagnose_local_calib import app_metric, discard_early, loo_ridge

_NOSE = 1  # 鼻先: 顔の画面内位置の代理


def feats_from_lms(lms, w, h):
    def P(i):
        return np.array([lms[i][0] * w, lms[i][1] * h])
    L_px = np.mean([P(i) for i in _LEFT_IRIS], axis=0)
    R_px = np.mean([P(i) for i in _RIGHT_IRIS], axis=0)

    # --- アプリ方式: 画像中心基準 / 虹彩直径 ---
    cx, cy = w / 2.0, h / 2.0
    L_d = 2.0 * np.mean([np.linalg.norm(P(i) - L_px) for i in _LEFT_IRIS[1:]])
    R_d = 2.0 * np.mean([np.linalg.norm(P(i) - R_px) for i in _RIGHT_IRIS[1:]])
    if L_d < 2.0 or R_d < 2.0:
        return None
    app = np.array([((L_px[0] - cx) / L_d + (R_px[0] - cx) / R_d) / 2.0,
                    ((L_px[1] - cy) / L_d + (R_px[1] - cy) / R_d) / 2.0])

    # --- 7D方式: 目頭・目尻基準 ---
    L_n = _geo_normalize(L_px, P(_L_IN), P(_L_OUT))
    R_n = _geo_normalize(R_px, P(_R_IN), P(_R_OUT))
    geo = np.array([L_n[0], L_n[1], R_n[0], R_n[1]])

    nose = P(_NOSE) / np.array([w, h])   # 顔の画面内位置
    return app, geo, nose


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "20260716_130217"
    d = load_raw_landmarks(f"logs/session_{sid}_landmarks")
    n = d["n"]
    APP, GEO, NOSE, Y, T = [], [], [], [], []
    for k in range(n):
        if not bool(d["has_target"][k]):
            continue
        r = feats_from_lms(d["landmarks"][k], int(d["img_w"][k]), int(d["img_h"][k]))
        if r is None:
            continue
        app, geo, nose = r
        if not (np.isfinite(app).all() and np.isfinite(geo).all()):
            continue
        APP.append(app); GEO.append(geo); NOSE.append(nose)
        Y.append(d["target"][k]); T.append(float(d["time_s"][k]))
    APP, GEO, NOSE = np.array(APP), np.array(GEO), np.array(NOSE)
    Y, T = np.array(Y), np.array(T)

    uniq, ids, keep = discard_early(Y, T)
    APP, GEO, NOSE, Y, ids = APP[keep], GEO[keep], NOSE[keep], Y[keep], ids[keep]
    print(f"[data] {len(Y)} フレーム / {len(uniq)} 点  (同一生ランドマークから両特徴を計算)\n")

    print("=" * 66)
    print("証拠1: アプリ特徴は『視線』より『顔の画面内位置』と強く相関する")
    print("=" * 66)
    print("        相関 |r|      対 顔の画面内位置(鼻先)   対 正解の視線ターゲット")
    for nm, F in [("アプリ X_feat", APP[:, 0]), ("7D風 Lx(目頭基準)", GEO[:, 0])]:
        r_head = abs(np.corrcoef(F, NOSE[:, 0])[0, 1])
        r_gaze = abs(np.corrcoef(F, Y[:, 0])[0, 1])
        print(f"  {nm:22s}      {r_head:.3f}                   {r_gaze:.3f}")
    for nm, F in [("アプリ Y_feat", APP[:, 1]), ("7D風 Ly(目頭基準)", GEO[:, 1])]:
        r_head = abs(np.corrcoef(F, NOSE[:, 1])[0, 1])
        r_gaze = abs(np.corrcoef(F, Y[:, 1])[0, 1])
        print(f"  {nm:22s}      {r_head:.3f}                   {r_gaze:.3f}")

    print()
    print("=" * 66)
    print("証拠2: 特徴だけ差し替えた直接対決（モデルは同じ plain Ridge・同一指標）")
    print("=" * 66)
    r = loo_ridge(APP, Y, uniq, ids)
    print(f"  アプリ方式 2D (画像中心基準)          : {app_metric(r):6.3f} cm")
    r = loo_ridge(GEO, Y, uniq, ids)
    print(f"  目頭基準 4D (_geo_normalize)         : {app_metric(r):6.3f} cm")
    r = loo_ridge(np.hstack([APP, NOSE]), Y, uniq, ids)
    print(f"  アプリ方式 + 顔位置を明示的に与える    : {app_metric(r):6.3f} cm")
    print("   ↑ 顔位置を渡すと改善するなら、アプリ特徴が顔位置に汚染されている決定的証拠")


if __name__ == "__main__":
    main()
