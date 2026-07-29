"""アピアランス版 推定器（実験用）。GazeEstimator と同じ公開I/Fで、16D幾何に目パッチPCA16を足す。
現mainの estimator.py は変更しない。main_appearance.py がこれを使う。
"""
import time as _time
import numpy as np
from typing import Optional, Tuple

from features import GazeFeatureExtractor
from filters import OneEuroFilter2D
from appearance import AppearancePipeline, eye_patch
from rich16d import rich_16d_from_lms, lms_to_array


class AppearanceEstimator:
    """16D幾何 + 目パッチ(48x32 CLAHE)PCA16 の視線推定。I/F は GazeEstimator と一致。"""

    def __init__(self):
        self._extractor = GazeFeatureExtractor()
        self._smoother = OneEuroFilter2D(min_cutoff=1.0, beta=0.05)
        self.calibration = AppearancePipeline(n_pca=16)

        self._last_feat16: Optional[np.ndarray] = None
        self._last_patch:  Optional[np.ndarray] = None
        self._last_pitch_rad = 0.0
        self._last_yaw_rad = 0.0
        self._cur_feat16: Optional[np.ndarray] = None
        self._cur_patch:  Optional[np.ndarray] = None
        self._cur_debug:  Optional[dict] = None
        self._last_proc_time = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        feat, debug = self._extractor.extract(frame)
        if feat is None:
            self._cur_feat16 = self._cur_patch = self._cur_debug = None
            return None, None

        # 16D を確保(config.MODE='7d' でも壊れないよう保険で16Dを再構成)
        feat = np.asarray(feat, np.float32)
        if feat.shape[0] == 16:
            feat16 = feat
        else:
            feat16 = rich_16d_from_lms(lms_to_array(debug['landmarks']),
                                       frame.shape[1], frame.shape[0]).astype(np.float32)

        patch = eye_patch(frame, debug['landmarks'])   # None のこともある(その場合は予測不可)

        self._cur_feat16, self._cur_patch, self._cur_debug = feat16, patch, debug

        pitch_rad = float(debug.get('pitch_rad', 0.0))
        yaw_rad   = float(debug.get('yaw_rad', 0.0))
        head_ok = self.calibration.head_pose_ok(pitch_rad, yaw_rad)
        dp, dy = self.calibration.head_pose_delta_deg(pitch_rad, yaw_rad)
        debug['head_ok'] = head_ok
        debug['head_delta_pitch'] = dp
        debug['head_delta_yaw'] = dy

        self._last_feat16 = feat16
        self._last_patch = patch
        self._last_pitch_rad = pitch_rad
        self._last_yaw_rad = yaw_rad
        head_vec = np.array([pitch_rad, yaw_rad])

        now = _time.time()
        dt = now - self._last_proc_time if self._last_proc_time > 0 else 1.0 / 30.0
        dt = float(np.clip(dt, 1.0 / 120.0, 0.5))
        self._last_proc_time = now

        if patch is None:
            # パッチ抽出失敗フレームは平滑器を更新せず前回値を返す(まれ)
            debug.update({'features': feat16, 'raw_pred': None,
                          'calibrated': self.calibration.is_calibrated})
            return None, debug

        raw_pred = self.calibration.predict(feat16, patch, head_vec)
        raw_clipped = np.clip(raw_pred, -0.05, 1.05)
        smoothed = self._smoother.update(raw_clipped, dt)
        gaze = np.clip(smoothed, 0.0, 1.0).astype(np.float32)

        debug.update({'features': feat16, 'raw_pred': raw_pred,
                      'gaze': gaze, 'calibrated': self.calibration.is_calibrated})
        return gaze, debug

    def collect_calibration(self, target_normalized: np.ndarray, weight: float = 1.0) -> bool:
        if self._cur_feat16 is not None and self._cur_patch is not None:
            pitch = self._cur_debug.get('pitch_rad') if self._cur_debug else None
            yaw   = self._cur_debug.get('yaw_rad') if self._cur_debug else None
            self.calibration.collect_point(self._cur_feat16, self._cur_patch,
                                           target_normalized, weight,
                                           pitch_rad=pitch, yaw_rad=yaw)
            return True
        return False

    def finalize_calibration(self) -> None:
        self.calibration.finalize()
        self._smoother.reset()

    def record_tap(self, screen_gt_normalized: np.ndarray) -> None:
        if self._last_feat16 is None or self._last_patch is None:
            return
        head_vec = np.array([self._last_pitch_rad, self._last_yaw_rad])
        self.calibration.record_interaction(screen_gt_normalized, self._last_feat16,
                                            self._last_patch, head_vec)

    def reset_calibration(self) -> None:
        self.calibration.reset()
        self._smoother.reset()
        self._last_feat16 = self._last_patch = None
        self._last_pitch_rad = self._last_yaw_rad = 0.0
        self._cur_feat16 = self._cur_patch = self._cur_debug = None

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.is_calibrated

    @property
    def face_detected(self) -> bool:
        return self._cur_feat16 is not None

    def close(self) -> None:
        self._extractor.close()
