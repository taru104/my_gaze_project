"""
時間的ノイズ除去フィルタモジュール。
- IQRFilter        : 四分位範囲ベースの外れ値除去
- EMAFilter        : 指数移動平均スムージング
- KalmanFilter2D   : 2次元カルマンフィルタ（状態: [x, y, vx, vy]）
- OneEuroFilter    : 適応型ローパスフィルタ（スカラー）
- OneEuroFilter2D  : 2次元視線座標用 One Euro Filter
"""

import numpy as np
from collections import deque
from typing import Optional


class IQRFilter:
    """
    直近フレームバッファから IQR を計算し外れ値を検出する。
    設計書: m=0.6, [Q1 - 0.6*IQR, Q3 + 0.6*IQR] を外れ値閾値とする。
    """

    def __init__(self, window_size: int = 30, m: float = 0.6, min_samples: int = 10):
        self.m = m
        self.min_samples = min_samples
        self._buf: deque = deque(maxlen=window_size)

    def is_outlier(self, x: np.ndarray) -> bool:
        """外れ値であれば True を返す。バッファが蓄積されるまでは外れ値扱いしない。"""
        if len(self._buf) < self.min_samples:
            self._buf.append(x.copy())
            return False

        buf = np.array(self._buf)           # (N, D)
        q1 = np.percentile(buf, 25, axis=0)
        q3 = np.percentile(buf, 75, axis=0)
        iqr = q3 - q1

        lower = q1 - self.m * iqr
        upper = q3 + self.m * iqr

        outlier = bool(np.any((x < lower) | (x > upper)))
        if not outlier:
            self._buf.append(x.copy())
        return outlier

    def seed(self, samples: list) -> None:
        """バッファを既知の正常サンプルで事前充填する（キャリブ後の初期化用）。"""
        for s in samples:
            self._buf.append(np.asarray(s, dtype=np.float32).copy())

    def reset(self):
        self._buf.clear()


class EMAFilter:
    """
    指数移動平均フィルタ。
    alpha = 2 / (window + 1) でウィンドウサイズと一対一対応。
    """

    def __init__(self, window: int = 5):
        assert 1 <= window <= 100, "window must be in [1, 100]"
        self.alpha = 2.0 / (window + 1)
        self._val: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if self._val is None:
            self._val = x.copy()
        else:
            self._val = self.alpha * x + (1.0 - self.alpha) * self._val
        return self._val.copy()

    def reset(self):
        self._val = None


class KalmanFilter2D:
    """
    2D 視線座標追跡用カルマンフィルタ。
    状態ベクトル: [x, y, vx, vy]
    観測ベクトル: [x, y]
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
        dt: float = 1.0,
    ):
        # 状態遷移行列 F (等速モデル)
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ],
        ], dtype=np.float64)

        # 観測行列 H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # プロセスノイズ共分散 Q
        self.Q = np.eye(4, dtype=np.float64) * process_noise

        # 観測ノイズ共分散 R
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # 初期推定誤差共分散
        self.P = np.eye(4, dtype=np.float64)
        self.x = np.zeros(4, dtype=np.float64)
        self._initialized = False

    def update(self, z: np.ndarray) -> np.ndarray:
        """観測値 z: (2,) で更新し、平滑化された位置 (2,) を返す。"""
        z = z.astype(np.float64)

        if not self._initialized:
            self.x[:2] = z
            self._initialized = True
            return z.copy()

        # ── Predict ──
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ──
        S = self.H @ P_pred @ self.H.T + self.R       # 残差共分散
        K = P_pred @ self.H.T @ np.linalg.inv(S)      # カルマンゲイン

        innovation = z - self.H @ x_pred
        self.x = x_pred + K @ innovation
        I_KH = np.eye(4) - K @ self.H
        # Joseph form for numerical stability
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        return self.x[:2].copy()

    def reset(self):
        self.P = np.eye(4, dtype=np.float64)
        self.x = np.zeros(4, dtype=np.float64)
        self._initialized = False


class OneEuroFilter:
    """
    Adaptive low-pass filter (Casiez et al. 2012).
    Slow signal → heavy smoothing; fast signal → pass-through (saccade).
    """

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: float, dt: float = 1.0 / 30.0) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x

    def reset(self):
        self._x = None
        self._dx = 0.0


class OneEuroFilter2D:
    """One Euro Filter applied independently to x and y gaze coordinates."""

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def update(self, z: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        x = self._fx.update(float(z[0]), dt)
        y = self._fy.update(float(z[1]), dt)
        return np.array([x, y], dtype=np.float32)

    def reset(self):
        self._fx.reset()
        self._fy.reset()


class OneEuroFilterND:
    """
    One Euro Filter applied independently to each of N dimensions.

    Use cases:
      HeadFilter  : dim=6, min_cutoff=0.1,  beta=0.01  (rvec+tvec, slow)
      EyeFilter   : dim=6, min_cutoff=1.5,  beta=0.30  (iris pos+diam, fast)
    """

    def __init__(
        self,
        dim: int,
        min_cutoff: float = 1.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self._filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(dim)]
        self.dim = dim

    def update(self, x: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        return np.array([f.update(float(x[i]), dt) for i, f in enumerate(self._filters)],
                        dtype=np.float32)

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
