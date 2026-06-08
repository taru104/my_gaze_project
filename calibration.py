"""
キャリブレーションモジュール。
- AffineCalibration  : 2×3アフィン変換 (lstsq) -- [X_mm, Y_mm, 1] → [x_norm, y_norm]
- DynamicCalibration : タップ履歴ベースの動的自動補正
- CalibrationPipeline: 9点キャリブレーション + 動的補正の統合
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── 9点キャリブレーション座標（正規化 [0,1]）────────────────────────────────
CALIB_POINTS_9 = np.array([
    [0.5, 0.5],   # 中央
    [0.1, 0.1],   # 左上
    [0.9, 0.1],   # 右上
    [0.1, 0.9],   # 左下
    [0.9, 0.9],   # 右下
    [0.5, 0.1],   # 上中央
    [0.1, 0.5],   # 左中央
    [0.9, 0.5],   # 右中央
    [0.5, 0.9],   # 下中央
], dtype=np.float32)

CALIB_POINTS_5 = np.array([
    [0.5, 0.5],
    [0.1, 0.1],
    [0.9, 0.1],
    [0.1, 0.9],
    [0.9, 0.9],
], dtype=np.float32)


class AffineCalibration:
    """
    重み付き最小二乗法による 3×2 アフィン変換。
    入力: [X_feat, Y_feat, pitch_rad]  出力: [x_norm, y_norm]
    設計行列 (N,4) = [X_feat, Y_feat, pitch_rad, 1] を lstsq で解く。
    pitch_rad を加えることで頭部仰俯角による Y 推定汚染を補正する。
    """

    def __init__(self):
        self._A:       Optional[np.ndarray] = None  # (4, 2)
        self._design:  List[List[float]]    = []
        self._targets: List[List[float]]    = []
        self._weights: List[float]          = []

    def add(self, X_mm: float, Y_mm: float,
            x_norm: float, y_norm: float,
            weight: float = 1.0,
            pitch_rad: float = 0.0) -> None:
        self._design.append([X_mm, Y_mm, pitch_rad, 1.0])
        self._targets.append([x_norm, y_norm])
        self._weights.append(max(0.0, weight))

    def fit(self) -> None:
        D = np.array(self._design,  dtype=np.float64)   # (N, 4): [X, Y, pitch, 1]
        T = np.array(self._targets, dtype=np.float64)   # (N, 2)
        W = np.sqrt(np.array(self._weights, dtype=np.float64))[:, None]

        Dw = D * W
        Tw = T * W

        # X出力: pitch を含まない [X_feat, Y_feat, 1] で解く
        Ax, _, _, _ = np.linalg.lstsq(Dw[:, [0, 1, 3]], Tw[:, 0:1], rcond=None)  # (3,1)

        # Y出力: pitch を含む [X_feat, Y_feat, pitch_rad, 1] で解く
        Ay, _, _, _ = np.linalg.lstsq(Dw,               Tw[:, 1:2], rcond=None)  # (4,1)

        # (4, 2) 行列に組み立て。X列の pitch 係数(index 2)は 0 のまま
        self._A = np.zeros((4, 2), dtype=np.float64)
        self._A[[0, 1, 3], 0] = Ax[:, 0]   # X: X_feat, Y_feat, bias
        self._A[:, 1]         = Ay[:, 0]   # Y: X_feat, Y_feat, pitch, bias

    def predict(self, X_mm: float, Y_mm: float, pitch_rad: float = 0.0) -> np.ndarray:
        if self._A is None:
            return np.array([0.5, 0.5], dtype=np.float32)
        return (np.array([X_mm, Y_mm, pitch_rad, 1.0], dtype=np.float64) @ self._A).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._A is not None

    def reset(self) -> None:
        self._A = None
        self._design.clear()
        self._targets.clear()
        self._weights.clear()


class TargetedPolyCalibration:
    """
    X方向: Ridge([X_feat, Y_feat])                         → 2特徴+intercept = 3パラメータ
    Y方向: Ridge([X_feat, Y_feat, pitch, Y_feat^2, Y*pitch])→ 5特徴+intercept = 6パラメータ

    根拠:
      X は水平注視に対してほぼ線形。Y^2 や X*Y は不要。
      Y にのみ非線形項が必要: まぶた遮蔽→Y^2、r=0.37のpitch汚染→Y*pitch。
      LOO(8点訓練)でもX:3、Y:6と十分過決定。PolyRidge(10パラメータ)より安定。
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._ridge_x  = Ridge(alpha=alpha, fit_intercept=True)
        self._ridge_y  = Ridge(alpha=alpha, fit_intercept=True)
        self._is_fitted = False
        self._raw:     List[List[float]] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def _fx(self, X: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.column_stack([X, Y])

    def _fy(self, X: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.column_stack([X, Y, p, Y**2, Y * p])

    def add(self, X_mm: float, Y_mm: float,
            x_norm: float, y_norm: float,
            weight: float = 1.0,
            pitch_rad: float = 0.0) -> None:
        self._raw.append([X_mm, Y_mm, pitch_rad])
        self._targets.append([x_norm, y_norm])
        self._weights.append(max(0.0, weight))

    def fit(self) -> None:
        raw = np.array(self._raw,     dtype=np.float64)
        T   = np.array(self._targets, dtype=np.float64)
        W   = np.array(self._weights, dtype=np.float64)
        X, Y, p = raw[:, 0], raw[:, 1], raw[:, 2]
        Fx = self._scaler_x.fit_transform(self._fx(X, Y, p))
        Fy = self._scaler_y.fit_transform(self._fy(X, Y, p))
        self._ridge_x.fit(Fx, T[:, 0], sample_weight=W)
        self._ridge_y.fit(Fy, T[:, 1], sample_weight=W)
        self._is_fitted = True

    def predict(self, X_mm: float, Y_mm: float, pitch_rad: float = 0.0) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        X = np.array([[X_mm]]); Y = np.array([[Y_mm]]); p = np.array([[pitch_rad]])
        Fx = self._scaler_x.transform(self._fx(X, Y, p))
        Fy = self._scaler_y.transform(self._fy(X, Y, p))
        return np.array([float(self._ridge_x.predict(Fx)[0]),
                         float(self._ridge_y.predict(Fy)[0])], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._is_fitted = False
        self._raw.clear()
        self._targets.clear()
        self._weights.clear()


class PolyRidgeCalibration:
    """
    2次多項式展開 + StandardScaler + Ridge 回帰によるキャリブレーション。
    入力: [X_feat, Y_feat, pitch_rad]  出力: [x_norm, y_norm]

    Pipeline: PolynomialFeatures(degree=2) → StandardScaler → Ridge(alpha=1.0)
    StandardScaler が必須: Ridgeはスケール依存のため正規化なしでは
    多項式項(X², XY, Y²)のスケール差でRidgeが正常に機能しない。
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import Ridge
        self._pipe_x = Pipeline([
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge',  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        self._pipe_y = Pipeline([
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge',  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        self._is_fitted = False
        self._raw:     List[List[float]] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def add(self, X_mm: float, Y_mm: float,
            x_norm: float, y_norm: float,
            weight: float = 1.0,
            pitch_rad: float = 0.0) -> None:
        self._raw.append([X_mm, Y_mm, pitch_rad])
        self._targets.append([x_norm, y_norm])
        self._weights.append(max(0.0, weight))

    def fit(self) -> None:
        X_raw = np.array(self._raw,     dtype=np.float64)
        T     = np.array(self._targets, dtype=np.float64)
        W     = np.array(self._weights, dtype=np.float64)
        self._pipe_x.fit(X_raw, T[:, 0], ridge__sample_weight=W)
        self._pipe_y.fit(X_raw, T[:, 1], ridge__sample_weight=W)
        self._is_fitted = True

    def predict(self, X_mm: float, Y_mm: float, pitch_rad: float = 0.0) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        X_raw = np.array([[X_mm, Y_mm, pitch_rad]], dtype=np.float64)
        x = float(self._pipe_x.predict(X_raw)[0])
        y = float(self._pipe_y.predict(X_raw)[0])
        return np.array([x, y], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._is_fitted = False
        self._raw.clear()
        self._targets.clear()
        self._weights.clear()


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HistEntry:
    screen_gt: np.ndarray
    predicted: np.ndarray
    head_vec:  np.ndarray   # [pitch, yaw] 正規化済み


class DynamicCalibration:
    """
    ユーザーの操作履歴（タップ等）を使った動的自動補正。

    補正式:
        G_C = G_E + (Σ λ_i * h_i * dG_i) / Σ λ_i * h_i
        λ_i = 1 / ||dG_i||          距離逆数重み
        h_i = dot(H_current, H_i)   頭部姿勢コサイン類似度（負は0でクランプ）
    """

    def __init__(self, max_history: int = 16, min_err: float = 1e-6):
        self.max_history = max_history
        self.min_err     = min_err
        self._hist: List[_HistEntry] = []

    def add(self, screen_gt: np.ndarray, predicted: np.ndarray, head_vec: np.ndarray) -> None:
        norm  = np.linalg.norm(head_vec) + 1e-8
        entry = _HistEntry(
            screen_gt=screen_gt.astype(np.float32),
            predicted=predicted.astype(np.float32),
            head_vec=(head_vec / norm).astype(np.float32),
        )
        self._hist.append(entry)
        if len(self._hist) > self.max_history:
            self._hist.pop(0)

    def correct(self, predicted: np.ndarray, current_head: np.ndarray) -> np.ndarray:
        if not self._hist:
            return predicted.copy()
        h_cur = current_head / (np.linalg.norm(current_head) + 1e-8)
        num = np.zeros(2, dtype=np.float64)
        den = 0.0
        for e in self._hist:
            dG  = e.screen_gt - e.predicted
            err = np.linalg.norm(dG)
            if err < self.min_err:
                continue
            h_sim        = max(0.0, float(np.dot(h_cur, e.head_vec)))
            spatial_dist = np.linalg.norm(predicted - e.screen_gt)
            w            = h_sim / (spatial_dist + 1e-4)
            num  += w * dG
            den  += w
        if den < 1e-10:
            return predicted.copy()
        return (predicted + num / den).astype(np.float32)

    def __len__(self) -> int:
        return len(self._hist)


# ─────────────────────────────────────────────────────────────────────────────

class CalibrationPipeline:
    """
    9点キャリブレーション + 動的補正 の統合パイプライン。

    使い方:
        1. collect_point(gaze_2d, target, weight, pitch_rad, yaw_rad) を各注視点中に呼ぶ
        2. finalize() でポリリッジ回帰をフィット
        3. predict(gaze_2d, head_vec) で補正済み画面座標を得る
        4. record_interaction(gt, gaze_2d, head_vec) でタップ履歴を蓄積
    """

    HEAD_POSE_TOLERANCE_RAD: float = np.radians(20.0)

    def __init__(self):
        self.poly_ridge = TargetedPolyCalibration()
        self.dynamic    = DynamicCalibration()
        self._samples:       List[Tuple] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples:   List[float] = []
        self.train_mgae: Optional[float] = None
        self.train_rmse: Optional[float] = None
        self.loo_mgae:   Optional[float] = None
        self.loo_euc:    Optional[float] = None
        self.loo_euc_x:  Optional[float] = None
        self.loo_euc_y:  Optional[float] = None
        self._ref_pitch: float = 0.0
        self._ref_yaw:   float = 0.0

    def collect_point(
        self,
        gaze_2d:   np.ndarray,
        target:    np.ndarray,
        weight:    float = 1.0,
        pitch_rad: Optional[float] = None,
        yaw_rad:   Optional[float] = None,
    ) -> None:
        pitch = float(pitch_rad) if pitch_rad is not None else 0.0
        self._samples.append((gaze_2d.copy(), target.astype(np.float32).copy(), float(weight), pitch))
        if pitch_rad is not None:
            self._pitch_samples.append(float(pitch_rad))
        if yaw_rad is not None:
            self._yaw_samples.append(float(yaw_rad))

    def finalize(self) -> None:
        if len(self._samples) < 5:
            raise ValueError(f"キャリブレーションサンプルが不足: {len(self._samples)} < 5")

        for gaze_2d, target, weight, pitch in self._samples:
            self.poly_ridge.add(
                float(gaze_2d[0]), float(gaze_2d[1]),
                float(target[0]),  float(target[1]),
                weight,
                pitch_rad=pitch,
            )
        self.poly_ridge.fit()

        if self._pitch_samples:
            self._ref_pitch = float(np.mean(self._pitch_samples))
            self._ref_yaw   = float(np.mean(self._yaw_samples))

        y_raw  = np.stack([s[1] for s in self._samples])
        y_pred = np.array([
            self.poly_ridge.predict(float(s[0][0]), float(s[0][1]), s[3])
            for s in self._samples
        ])
        self.train_rmse = float(np.sqrt(np.mean((y_pred - y_raw) ** 2)))
        self.train_mgae = float(self._compute_mgae(y_pred, y_raw))

        self.loo_mgae, self.loo_euc, self.loo_euc_x, self.loo_euc_y = self._compute_loo()

    def head_pose_ok(self, pitch: float, yaw: float) -> bool:
        if not self.is_calibrated:
            return True
        tol = self.HEAD_POSE_TOLERANCE_RAD
        return (abs(pitch - self._ref_pitch) <= tol and
                abs(yaw   - self._ref_yaw)   <= tol)

    def head_pose_delta_deg(self, pitch: float, yaw: float) -> Tuple[float, float]:
        return float(np.degrees(pitch - self._ref_pitch)), float(np.degrees(yaw - self._ref_yaw))

    def predict(
        self,
        gaze_2d:     np.ndarray,
        head_vec:    Optional[np.ndarray] = None,
        use_dynamic: bool = True,
    ) -> np.ndarray:
        if not self.poly_ridge.is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        pitch_rad = float(head_vec[0]) if head_vec is not None else 0.0
        pred = self.poly_ridge.predict(float(gaze_2d[0]), float(gaze_2d[1]), pitch_rad)
        if use_dynamic and head_vec is not None and len(self.dynamic) > 0:
            pred = self.dynamic.correct(pred, head_vec)
        return pred

    def record_interaction(
        self,
        screen_gt: np.ndarray,
        gaze_2d:   np.ndarray,
        head_vec:  np.ndarray,
    ) -> None:
        if self.poly_ridge.is_fitted:
            pitch_rad = float(head_vec[0]) if head_vec is not None else 0.0
            predicted = self.poly_ridge.predict(float(gaze_2d[0]), float(gaze_2d[1]), pitch_rad)
        else:
            predicted = np.array([0.5, 0.5], dtype=np.float32)
        self.dynamic.add(screen_gt, predicted, head_vec)

    def _compute_loo(self) -> Tuple[Optional[float], ...]:
        """
        Leave-One-Out Cross-Validation。
        9点のうち1点を除いて残り8点でフィット → 除いた点を予測 → 誤差計算。
        """
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for s in self._samples:
            key = (round(float(s[1][0]), 4), round(float(s[1][1]), 4))
            groups[key].append(s)

        if len(groups) < 3:
            return None, None, None, None

        mgae_list = []
        euc_list  = []
        ex_list   = []
        ey_list   = []

        for held_key, held_samples in groups.items():
            af = TargetedPolyCalibration()
            for key, samples in groups.items():
                if key == held_key:
                    continue
                for gaze_2d, target, weight, pitch in samples:
                    af.add(float(gaze_2d[0]), float(gaze_2d[1]),
                           float(target[0]),  float(target[1]),
                           weight, pitch_rad=pitch)
            if len(af._raw) < 5:
                continue
            af.fit()

            preds = np.array([
                af.predict(float(s[0][0]), float(s[0][1]), s[3])
                for s in held_samples
            ])
            pred_med  = np.median(preds, axis=0)
            target_pt = np.array(held_key)
            err       = pred_med - target_pt

            euc_list.append(float(np.linalg.norm(err)))
            ex_list.append(float(abs(err[0])))
            ey_list.append(float(abs(err[1])))

            def to3d(p): return np.array([p[0]-0.5, p[1]-0.5, 1.0])
            v1, v2  = to3d(pred_med), to3d(target_pt)
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            mgae_list.append(float(np.degrees(np.arccos(np.clip(cos_sim, -1+1e-7, 1-1e-7)))))

        if not mgae_list:
            return None, None, None, None
        return (float(np.mean(mgae_list)), float(np.mean(euc_list)),
                float(np.mean(ex_list)),   float(np.mean(ey_list)))

    @staticmethod
    def _compute_mgae(pred: np.ndarray, target: np.ndarray) -> float:
        def to3d(p): return np.array([p[0] - 0.5, p[1] - 0.5, 1.0])
        angles = []
        for p, t in zip(pred, target):
            v1, v2 = to3d(p), to3d(t)
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos_sim, -1 + 1e-7, 1 - 1e-7))))
        return float(np.mean(angles))

    def reset(self) -> None:
        self._samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        self.poly_ridge = TargetedPolyCalibration()
        self.dynamic    = DynamicCalibration()
        self.train_mgae = None
        self.train_rmse = None
        self.loo_mgae   = None
        self.loo_euc    = None
        self.loo_euc_x  = None
        self.loo_euc_y  = None
        self._ref_pitch = 0.0
        self._ref_yaw   = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self.poly_ridge.is_fitted

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def kappa_offset_norm(self) -> np.ndarray:
        if not self.is_calibrated:
            return np.zeros(2, dtype=np.float32)
        rx = self.poly_ridge._ridge_x.intercept_
        ry = self.poly_ridge._ridge_y.intercept_
        return np.array([float(rx), float(ry)], dtype=np.float32)
