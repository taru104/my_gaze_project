"""
視線推定エンドツーエンドパイプライン。
MediaPipe 486D特徴抽出 → 瞬き検出 → 頭部姿勢ゲート → キャリブレーション → One Euro Filter
"""

import time as _time
import numpy as np
from typing import Optional, Tuple

from features    import GazeFeatureExtractor
from filters     import KalmanFilter2D, OneEuroFilter2D
from calibration import CalibrationPipeline


class GazeEstimator:
    """
    リアルタイム視線推定パイプライン（486D頭部姿勢不変特徴量）。

    処理フロー:
        frame
          └─ GazeFeatureExtractor  → 486D features + blink + pitch/yaw
               └─ 瞬き検出 / 頭部姿勢ゲート
                    └─ CalibrationPipeline → 2D screen coord
                         └─ OneEuroFilter2D → smoothed gaze
    """

    def __init__(
        self,
        use_kalman:        bool  = False,
        process_noise:     float = 1e-3,
        measurement_noise: float = 1e-4,
    ):
        self._extractor = GazeFeatureExtractor()

        if use_kalman:
            self._smoother = KalmanFilter2D(process_noise, measurement_noise)
        else:
            # ユーザは追従より「震えない(平滑)」を優先(操作感)。beta を低く保つ。
            # min_cutoff も下げて静止時をさらに平滑化(遅延は許容されている)。
            self._smoother = OneEuroFilter2D(min_cutoff=1.0, beta=0.05)
        self._use_kalman = use_kalman
        self._last_proc_time: float = 0.0

        self.calibration = CalibrationPipeline()

        self._last_features:  Optional[np.ndarray] = None
        self._last_raw_pred:  Optional[np.ndarray] = None
        self._last_pitch_rad: float = 0.0
        self._last_yaw_rad:   float = 0.0
        self._current_features: Optional[np.ndarray] = None
        self._current_debug:    Optional[dict]        = None

    # ──── メイン処理 ──────────────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        フレームを受け取り、視線推定座標を返す。

        Returns:
            gaze  : np.ndarray (2,) in [0,1]^2、または検出失敗/瞬き/頭部ゲート時 None
            debug : dict
        """
        features, debug = self._extractor.extract(frame)
        self._current_features = features
        self._current_debug    = debug

        if features is None:
            return None, None

        pitch_rad = debug.get('pitch_rad', 0.0)
        yaw_rad   = debug.get('yaw_rad',   0.0)

        head_ok = self.calibration.head_pose_ok(pitch_rad, yaw_rad)

        if debug is not None:
            dp, dy = self.calibration.head_pose_delta_deg(pitch_rad, yaw_rad)
            debug['head_ok']          = head_ok
            debug['head_delta_pitch'] = dp
            debug['head_delta_yaw']   = dy

        # head_ok は HUD 用の助言フラグ。姿勢が外れても推定は止めない
        # (止めるより劣化した値を出す方がマシ: |yaw|30°超でも5.8cm＜中央固定12.4cm)
        self._last_features  = features
        self._last_pitch_rad = pitch_rad
        self._last_yaw_rad   = yaw_rad
        head_vec             = np.array([pitch_rad, yaw_rad])

        # dt for One Euro Filter
        now = _time.time()
        dt  = now - self._last_proc_time if self._last_proc_time > 0 else 1.0 / 30.0
        dt  = float(np.clip(dt, 1.0 / 120.0, 0.5))
        self._last_proc_time = now

        raw_pred = self.calibration.predict(features, head_vec)
        self._last_raw_pred = raw_pred

        raw_clipped = np.clip(raw_pred, -0.05, 1.05)

        if self._use_kalman:
            smoothed = self._smoother.update(raw_clipped)
        else:
            smoothed = self._smoother.update(raw_clipped, dt)

        gaze = np.clip(smoothed, 0.0, 1.0).astype(np.float32)

        if debug is not None:
            debug.update({
                'features':   features,
                'raw_pred':   raw_pred,
                'gaze':       gaze,
                'calibrated': self.calibration.is_calibrated,
            })

        return gaze, debug

    # ──── キャリブレーション関連 ──────────────────────────────────────────

    def collect_calibration(self, target_normalized: np.ndarray, weight: float = 1.0) -> bool:
        """キャリブレーション中に1フレームのサンプルを追加。"""
        if self._current_features is not None:
            pitch_rad = self._current_debug.get('pitch_rad') if self._current_debug else None
            yaw_rad   = self._current_debug.get('yaw_rad')   if self._current_debug else None
            self.calibration.collect_point(
                self._current_features, target_normalized, weight,
                pitch_rad=pitch_rad, yaw_rad=yaw_rad,
            )
            return True
        return False

    def finalize_calibration(self) -> None:
        """収集したサンプルでリッジ回帰をフィット。"""
        self.calibration.finalize()
        self._smoother.reset()

    def record_tap(self, screen_gt_normalized: np.ndarray) -> None:
        """ユーザーのタップ/クリック座標を動的キャリブレーションに記録。"""
        if self._last_features is None:
            return
        head_vec = np.array([self._last_pitch_rad, self._last_yaw_rad])
        self.calibration.record_interaction(
            screen_gt_normalized,
            self._last_features,
            head_vec,
        )

    def reset_calibration(self) -> None:
        self.calibration.reset()
        self._smoother.reset()
        self._last_features  = None
        self._last_raw_pred  = None
        self._last_pitch_rad = 0.0
        self._last_yaw_rad   = 0.0
        self._current_features = None
        self._current_debug    = None

    # ──── プロパティ ──────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.is_calibrated

    @property
    def last_features(self) -> Optional[np.ndarray]:
        return self._last_features

    @property
    def face_detected(self) -> bool:
        return self._current_features is not None

    def close(self) -> None:
        self._extractor.close()
