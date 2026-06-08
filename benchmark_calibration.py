"""
キャリブレーション手法の精度比較ベンチマーク。

AffineCalibration vs PolyRidgeCalibration を LOO-CV で比較する。

合成データの設計根拠:
  X_feat = ax*(tx-0.5) + bx*(ty-0.5)^2 + cx*(tx-0.5)*(ty-0.5) + noise
  Y_feat = ay*(ty-0.5) + cy*pitch + dy*(ty-0.5)^2 + ey*(ty-0.5)*pitch + noise

  Y方向の非線形性 (dy, ey) はまぶた遮蔽による iris_diam 変動を模擬。
  pitch汚染 (cy) は過去セッションで観測した r=+0.37 の相関に対応。
"""
import numpy as np
from collections import defaultdict
from calibration import AffineCalibration, PolyRidgeCalibration, CALIB_POINTS_9

SCREEN_CM_W = 30.9
SCREEN_CM_H = 17.4
N_SAMPLES   = 120       # サンプル数/点 (2秒 x 60fps)
N_TRIALS    = 30        # 異なる乱数シードで繰り返す回数
SIGMA_FEAT  = 0.010     # 特徴量ノイズ (正規化単位)
BASE_PITCH  = -0.08     # キャリブ時の平均ピッチ [rad] (約 -5 deg)

# === 合成マッピング係数 (真値) ===
AX  =  1.25   # X_feat → tx  線形スケール
BX  = -0.20   # (ty-0.5)^2  (わずかな樽型歪み)
CX  = -0.15   # (tx-0.5)*(ty-0.5) クロス項

AY  =  1.10   # Y_feat → ty  線形スケール
CY  =  0.40   # pitch → ty  (pitch汚染, r≈0.37 に対応)
DY  =  0.60   # (ty-0.5)^2  (まぶた遮蔽による上方注視の非線形性)
EY  =  0.35   # (ty-0.5)*pitch 交差項


def screen_to_feat(tx, ty, pitch):
    """真の逆マッピング: 画面座標 + pitch → 特徴量"""
    dx, dy = tx - 0.5, ty - 0.5
    X_feat = AX * dx + BX * dy**2 + CX * dx * dy
    Y_feat = AY * dy + CY * pitch + DY * dy**2 + EY * dy * pitch
    return X_feat, Y_feat


def run_loo(calib_factory, samples_by_key):
    """LOO-CV を実行して (euc_x_cm, euc_y_cm, euc_cm) を返す。"""
    ex_list, ey_list = [], []
    keys = list(samples_by_key.keys())
    for held_key in keys:
        af = calib_factory()
        for k, samples in samples_by_key.items():
            if k == held_key:
                continue
            for X, Y, p, tx, ty, w in samples:
                af.add(X, Y, tx, ty, weight=w, pitch_rad=p)
        raw_buf = af._raw if hasattr(af, '_raw') else af._design
        if len(raw_buf) < 5:
            continue
        af.fit()

        preds = np.array([af.predict(X, Y, p)
                          for X, Y, p, tx, ty, w in samples_by_key[held_key]])
        pred_med  = np.median(preds, axis=0)
        target_pt = np.array(held_key)
        err = pred_med - target_pt
        ex_list.append(abs(err[0]))
        ey_list.append(abs(err[1]))

    euc_x_cm = float(np.mean(ex_list)) * SCREEN_CM_W
    euc_y_cm = float(np.mean(ey_list)) * SCREEN_CM_H
    euc_cm   = float(np.sqrt(euc_x_cm**2 + euc_y_cm**2))
    return euc_x_cm, euc_y_cm, euc_cm


def run_train_mgae(calib_factory, samples_by_key):
    """全サンプルで訓練し train_MGAE を返す。"""
    af = calib_factory()
    all_samples = []
    for samples in samples_by_key.values():
        for X, Y, p, tx, ty, w in samples:
            af.add(X, Y, tx, ty, weight=w, pitch_rad=p)
            all_samples.append((X, Y, p, tx, ty))
    af.fit()

    angles = []
    for X, Y, p, tx, ty in all_samples:
        pred = af.predict(X, Y, p)
        def v3(pt): return np.array([pt[0]-0.5, pt[1]-0.5, 1.0])
        v1, v2 = v3(pred), v3(np.array([tx, ty]))
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
        angles.append(float(np.degrees(np.arccos(np.clip(cos, -1+1e-7, 1-1e-7)))))
    return float(np.mean(angles))


results_aff  = []   # (ex, ey, euc)
results_poly = []
train_mgae_aff  = []
train_mgae_poly = []

print("Running benchmark...")
for trial in range(N_TRIALS):
    rng = np.random.RandomState(trial)
    samples_by_key = defaultdict(list)

    for point in CALIB_POINTS_9:
        tx, ty = float(point[0]), float(point[1])
        key    = (round(tx, 4), round(ty, 4))
        pitch_base = BASE_PITCH + rng.normal(0, 0.02)

        for i in range(N_SAMPLES):
            p_i    = pitch_base + rng.normal(0, 0.01)
            X0, Y0 = screen_to_feat(tx, ty, p_i)
            X_n    = X0 + rng.normal(0, SIGMA_FEAT)
            Y_n    = Y0 + rng.normal(0, SIGMA_FEAT)
            w      = min(float(i) / (N_SAMPLES * 0.5), 1.0)
            samples_by_key[key].append((X_n, Y_n, p_i, tx, ty, w))

    ex_a, ey_a, ec_a = run_loo(AffineCalibration,    samples_by_key)
    ex_p, ey_p, ec_p = run_loo(PolyRidgeCalibration, samples_by_key)
    results_aff.append((ex_a, ey_a, ec_a))
    results_poly.append((ex_p, ey_p, ec_p))

    ma = run_train_mgae(AffineCalibration,    samples_by_key)
    mp = run_train_mgae(PolyRidgeCalibration, samples_by_key)
    train_mgae_aff.append(ma)
    train_mgae_poly.append(mp)

ra = np.array(results_aff)
rp = np.array(results_poly)
ma = np.array(train_mgae_aff)
mp = np.array(train_mgae_poly)

sep = "=" * 60
print()
print(sep)
print("  Benchmark Result  (N=%d trials, %d samples/point)" % (N_TRIALS, N_SAMPLES))
print("  Synthetic data: realistic nonlinear mapping + noise")
print(sep)
print()
print("  --- LOO-CV Error (generalization) ---")
print("                     X[cm]   Y[cm]  Euc[cm]")
print("  Affine   mean :   %5.2f   %5.2f   %5.2f" % (ra[:,0].mean(), ra[:,1].mean(), ra[:,2].mean()))
print("           std  :   %5.2f   %5.2f   %5.2f" % (ra[:,0].std(),  ra[:,1].std(),  ra[:,2].std()))
print("  PolyRidge mean:   %5.2f   %5.2f   %5.2f" % (rp[:,0].mean(), rp[:,1].mean(), rp[:,2].mean()))
print("           std  :   %5.2f   %5.2f   %5.2f" % (rp[:,0].std(),  rp[:,1].std(),  rp[:,2].std()))
print()
imp_x   = (ra[:,0].mean() - rp[:,0].mean()) / ra[:,0].mean() * 100
imp_y   = (ra[:,1].mean() - rp[:,1].mean()) / ra[:,1].mean() * 100
imp_euc = (ra[:,2].mean() - rp[:,2].mean()) / ra[:,2].mean() * 100
print("  Improvement:      X=%+.0f%%  Y=%+.0f%%  Euc=%+.0f%%" % (imp_x, imp_y, imp_euc))
print()
print("  --- Train MGAE (fitting quality, lower=better fit) ---")
print("  Affine    mean: %5.2f deg  std: %.2f" % (ma.mean(), ma.std()))
print("  PolyRidge mean: %5.2f deg  std: %.2f" % (mp.mean(), mp.std()))
print()
print("  [NOTE] Synthetic data includes:")
print("    - Y^2 nonlinearity (eyelid occlusion effect)")
print("    - pitch x Y interaction (observed r=+0.37 in logs)")
print("    - Feature noise sigma=%.3f (normalized units)" % SIGMA_FEAT)
print(sep)
