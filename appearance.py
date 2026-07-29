"""画像アピアランス版キャリブ（実験用・main_appearance.py が使う）。
研究(REPORT6 exp52-61)の確定チャンピオン=16D + 目パッチ(48x32,CLAHE)PCA16 + Huber を実機に載せる。
現mainのファイル(features/estimator/calibration/config)は一切変更しない。ここは新規追加のみ。

- eye_patch(): ライブフレーム＋MediaPipeランドマークから、目頭・目尻で相似正規化した48x32 CLAHEパッチ(2眼=3072D)。
- AppearancePipeline: CalibrationPipeline と同じ公開I/F。16D幾何にパッチPCA16を足して Huber 回帰。
  過学習警戒: パッチは必ずPCA16に圧縮(生3072Dは絶対に直接使わない)。
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple

from calibration import DynamicCalibration   # タップ動的補正は既存を流用

PW, PH = 48, 32
_EYES = [(33, 133), (362, 263)]              # (outer, inner) 右目 / 左目
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
N_PCA = 16
PATCH_DIM = 2 * PW * PH


def _sim_transform(p0, p1, q0, q1):
    dp = p1 - p0; dq = q1 - q0
    s = np.hypot(*dq) / (np.hypot(*dp) + 1e-9)
    a = np.arctan2(dq[1], dq[0]) - np.arctan2(dp[1], dp[0])
    c, sn = np.cos(a) * s, np.sin(a) * s
    R = np.array([[c, -sn], [sn, c]]); t = q0 - R @ p0
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], np.float32)


def _one_eye(gray, lms, w, h, oi, ii):
    O = np.array([lms[oi].x * w, lms[oi].y * h])
    I = np.array([lms[ii].x * w, lms[ii].y * h])
    M = _sim_transform(O, I, np.array([PW * 0.15, PH * 0.5]), np.array([PW * 0.85, PH * 0.5]))
    p = cv2.warpAffine(gray, M, (PW, PH), flags=cv2.INTER_AREA).astype(np.float32)
    return ((p - p.mean()) / (p.std() + 1e-6)).ravel()


def eye_patch(frame_bgr: np.ndarray, lms) -> Optional[np.ndarray]:
    """BGRフレーム + MediaPipeランドマーク列 → 48x32 CLAHE 目パッチ(2眼連結, 3072D)。失敗時 None。"""
    try:
        h, w = frame_bgr.shape[:2]
        clahe = _CLAHE.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        patch = np.concatenate([_one_eye(clahe, lms, w, h, oi, ii) for oi, ii in _EYES])
        if not np.isfinite(patch).all():
            return None
        return patch.astype(np.float32)
    except Exception:
        return None


class AppearancePipeline:
    """16D幾何 + 目パッチPCA16 → 2D画面座標。CalibrationPipeline と同じ公開I/F。"""

    HEAD_POSE_TOLERANCE_RAD: float = np.radians(20.0)

    def __init__(self, n_pca: int = N_PCA):
        self._npca = n_pca
        self._f16:   List[np.ndarray] = []
        self._patch: List[np.ndarray] = []
        self._targets: List[List[float]] = []
        self._weights: List[float] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples:   List[float] = []
        self.dynamic = DynamicCalibration()
        self._pca = self._scaler = self._hx = self._hy = None
        self._is_fitted = False
        self.train_mgae: Optional[float] = None
        self.train_rmse: Optional[float] = None
        self.loo_mgae = self.loo_euc = self.loo_euc_x = self.loo_euc_y = None
        self._ref_pitch = 0.0
        self._ref_yaw = 0.0

    # ── 収集 ──
    def collect_point(self, feat16, patch, target, weight=1.0, pitch_rad=None, yaw_rad=None):
        if patch is None:
            return
        self._f16.append(np.asarray(feat16, np.float64).ravel())
        self._patch.append(np.asarray(patch, np.float64).ravel())
        self._targets.append([float(target[0]), float(target[1])])
        self._weights.append(max(0.0, float(weight)))
        if pitch_rad is not None:
            self._pitch_samples.append(float(pitch_rad))
        if yaw_rad is not None:
            self._yaw_samples.append(float(yaw_rad))

    # ── 学習コア(finalize と LOO で共有) ──
    def _fit_models(self, F16, P, T, W):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import HuberRegressor, Ridge
        npca = int(min(self._npca, max(1, len(F16) - 1), P.shape[1]))
        pca = PCA(n_components=npca).fit(P)
        Aug = np.hstack([F16, pca.transform(P)])
        scaler = StandardScaler().fit(Aug)
        Z = scaler.transform(Aug)
        Wc = np.clip(W, 1e-3, None)
        ms = []
        for i in range(2):
            try:
                m = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(Z, T[:, i], sample_weight=Wc)
            except Exception:
                m = Ridge(alpha=1.0).fit(Z, T[:, i], sample_weight=Wc)
            ms.append(m)
        return pca, scaler, ms[0], ms[1]

    @staticmethod
    def _pred_models(pca, scaler, hx, hy, F16, P):
        Z = scaler.transform(np.hstack([F16, pca.transform(P)]))
        return np.column_stack([hx.predict(Z), hy.predict(Z)])

    def finalize(self):
        if len(self._f16) < 5:
            raise ValueError(f"キャリブサンプル不足: {len(self._f16)} < 5")
        F16 = np.array(self._f16); P = np.array(self._patch)
        T = np.array(self._targets); W = np.array(self._weights)
        self._pca, self._scaler, self._hx, self._hy = self._fit_models(F16, P, T, W)
        self._is_fitted = True
        if self._pitch_samples:
            self._ref_pitch = float(np.mean(self._pitch_samples))
            self._ref_yaw = float(np.mean(self._yaw_samples))
        pred = self._pred_models(self._pca, self._scaler, self._hx, self._hy, F16, P)
        self.train_rmse = float(np.sqrt(np.mean((pred - T) ** 2)))
        self.train_mgae = float(self._mgae(pred, T))
        self._compute_loo(F16, P, T, W)

    # ── 予測 ──
    def predict(self, feat16, patch, head_vec=None, use_dynamic=True):
        if not self._is_fitted or patch is None:
            return np.array([0.5, 0.5], dtype=np.float32)
        F16 = np.asarray(feat16, np.float64).reshape(1, -1)
        P = np.asarray(patch, np.float64).reshape(1, -1)
        pred = self._pred_models(self._pca, self._scaler, self._hx, self._hy, F16, P)[0].astype(np.float32)
        if use_dynamic and head_vec is not None and len(self.dynamic) > 0:
            pred = self.dynamic.correct(pred, head_vec)
        return pred

    def record_interaction(self, screen_gt, feat16, patch, head_vec):
        if self._is_fitted and patch is not None:
            predicted = self.predict(feat16, patch, use_dynamic=False)
        else:
            predicted = np.array([0.5, 0.5], dtype=np.float32)
        self.dynamic.add(screen_gt, predicted, head_vec)

    # ── 頭部姿勢の助言(推定は止めない) ──
    def head_pose_ok(self, pitch, yaw):
        if not self._is_fitted:
            return True
        tol = self.HEAD_POSE_TOLERANCE_RAD
        return abs(pitch - self._ref_pitch) <= tol and abs(yaw - self._ref_yaw) <= tol

    def head_pose_delta_deg(self, pitch, yaw):
        return float(np.degrees(pitch - self._ref_pitch)), float(np.degrees(yaw - self._ref_yaw))

    # ── LOO(点ごと leave-one-out, main HUD の cm 表示用) ──
    def _compute_loo(self, F16, P, T, W):
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(len(T)):
            groups[(round(float(T[i, 0]), 4), round(float(T[i, 1]), 4))].append(i)
        if len(groups) < 3:
            return
        euc, ex, ey, mg = [], [], [], []
        for held, hidx in groups.items():
            tr = []
            for key, idx in groups.items():
                if key == held:
                    continue
                step = max(1, len(idx) // 60)     # 各点~60フレームに間引き(高速化)
                tr.extend(idx[::step])
            if len(tr) < 5:
                continue
            tr = np.array(tr)
            try:
                pca, sc, hx, hy = self._fit_models(F16[tr], P[tr], T[tr], W[tr])
                preds = self._pred_models(pca, sc, hx, hy, F16[hidx], P[hidx])
            except Exception:
                continue
            pm = np.median(preds, axis=0)
            tgt = np.array(held)
            err = pm - tgt
            euc.append(float(np.linalg.norm(err))); ex.append(abs(float(err[0]))); ey.append(abs(float(err[1])))
            mg.append(self._angle(pm, tgt))
        if euc:
            self.loo_euc = float(np.mean(euc)); self.loo_euc_x = float(np.mean(ex))
            self.loo_euc_y = float(np.mean(ey)); self.loo_mgae = float(np.mean(mg))

    @staticmethod
    def _angle(p, t):
        v1 = np.array([p[0] - 0.5, p[1] - 0.5, 1.0]); v2 = np.array([t[0] - 0.5, t[1] - 0.5, 1.0])
        cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cs, -1 + 1e-7, 1 - 1e-7))))

    def _mgae(self, pred, tgt):
        return float(np.mean([self._angle(p, t) for p, t in zip(pred, tgt)]))

    def reset(self):
        self._f16.clear(); self._patch.clear(); self._targets.clear(); self._weights.clear()
        self._pitch_samples.clear(); self._yaw_samples.clear()
        self.dynamic = DynamicCalibration()
        self._pca = self._scaler = self._hx = self._hy = None
        self._is_fitted = False
        self.train_mgae = self.train_rmse = None
        self.loo_mgae = self.loo_euc = self.loo_euc_x = self.loo_euc_y = None
        self._ref_pitch = self._ref_yaw = 0.0

    @property
    def is_calibrated(self):
        return self._is_fitted

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def n_samples(self):
        return len(self._f16)
