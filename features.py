"""
MediaPipe FaceLandmarker + 虹彩深度推定による幾何学的視線推定。
- gaze_2d = [X_feat, Y_feat] (無次元・距離不変): X_feat = (iris_x-cx)/iris_diam_px
- HeadFilter (低周波): rvec/tvec スムージング → 安定した頭部姿勢 (min_cutoff=0.3)
- 虹彩はフィルタなし (estimator の OneEuroFilter2D が最終スムージング担当)
後方互換: build_7d_features(), extract_7d_from_image() は変更なし
"""

import numpy as np
import cv2
import time as _time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import os
from collections import deque
from typing import Optional, Tuple

from filters import OneEuroFilterND

# ─── 解剖学的ランドマークインデックス ─────────────────────────────────────────
LEFT_INNER_CANTHUS  = 362
RIGHT_INNER_CANTHUS = 133
MID_EYES            = 168
NOSE_BASE           = 2

LEFT_IRIS_EDGES  = [468, 469, 470, 471, 472]   # center + 4 edges
RIGHT_IRIS_EDGES = [473, 474, 475, 476, 477]

LEFT_IRIS_EDGE4  = [469, 470, 471, 472]         # edges only (for diameter)
RIGHT_IRIS_EDGE4 = [474, 475, 476, 477]

LEFT_IRIS_H_EDGES  = [469, 471]   # horizontal only (right, left) — eyelid-free
RIGHT_IRIS_H_EDGES = [474, 476]   # horizontal only (right, left) — eyelid-free

FEATURE_DIM = 7

# EAR瞬き検出
_L_INNER = 133; _L_OUTER = 33;  _L_TOP = 159; _L_BOT = 145
_R_INNER = 362; _R_OUTER = 263; _R_TOP = 386; _R_BOT = 374

_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

# 解剖学的mm単位 6点顔モデル (Y+=上, Z+=顔正面, 鼻尖を原点)
_FACE_3D_MODEL = np.array([
    [ 0.0,    0.0,    0.0  ],  # idx=1   鼻尖
    [ 0.0,  -63.6,  -12.5 ],  # idx=152 顎先
    [-43.3,  32.7,  -26.0 ],  # idx=33  左目外眼角
    [ 43.3,  32.7,  -26.0 ],  # idx=263 右目外眼角
    [-28.9, -28.9,  -24.1 ],  # idx=61  左口角
    [ 28.9, -28.9,  -24.1 ],  # idx=291 右口角
], dtype=np.float64)

_FACE_2D_IDX = [1, 152, 33, 263, 61, 291]
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# 眼球中心オフセット（顔ローカル座標、mm）
_EYE_L_LOCAL = np.array([-31.5,  32.7, -26.0], dtype=np.float64)
_EYE_R_LOCAL = np.array([ 31.5,  32.7, -26.0], dtype=np.float64)

VIRTUAL_SCREEN_Z_MM = 0.0

# 虹彩物理直径 (mm) – 成人平均値。この定数を使うと焦点距離がキャンセルされる。
IRIS_DIAMETER_MM = 11.7


def _geo_normalize(pupil: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    vec    = outer - inner
    length = np.linalg.norm(vec) + 1e-8
    center = (inner + outer) / 2.0
    rel    = pupil - center
    ang    = np.arctan2(vec[1], vec[0])
    ca, sa = np.cos(-ang), np.sin(-ang)
    rot    = np.array([[ca, -sa], [sa, ca]])
    return (rot @ rel) / length


def build_7d_features(
    lms,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """解剖学的7次元特徴量 [Lx,Ly,Rx,Ry,Pitch,Yaw,dist] を抽出する。"""
    if len(lms) < 478:
        return None

    f = float(img_w)
    cam = np.array([[f, 0, img_w / 2.0],
                    [0, f, img_h / 2.0],
                    [0, 0, 1.0        ]], dtype=np.float64)

    def iris_center_px(indices):
        pts = np.array([[lms[i].x * img_w, lms[i].y * img_h] for i in indices])
        return pts.mean(axis=0)

    L_px = iris_center_px(LEFT_IRIS_EDGES)
    R_px = iris_center_px(RIGHT_IRIS_EDGES)

    face_2d = np.array([[lms[i].x * img_w, lms[i].y * img_h] for i in _FACE_2D_IDX],
                        dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) * np.pi / 180.0
    yaw   = float(angles[1]) * np.pi / 180.0
    roll  = float(angles[2]) * np.pi / 180.0

    if pitch > np.pi / 2:
        pitch = np.pi - pitch
    elif pitch < -np.pi / 2:
        pitch = -np.pi - pitch

    l_inner = np.array([lms[RIGHT_INNER_CANTHUS].x * img_w, lms[RIGHT_INNER_CANTHUS].y * img_h])
    l_outer = np.array([lms[33].x * img_w,                  lms[33].y * img_h              ])
    r_inner = np.array([lms[LEFT_INNER_CANTHUS].x * img_w,  lms[LEFT_INNER_CANTHUS].y * img_h])
    r_outer = np.array([lms[263].x * img_w,                 lms[263].y * img_h              ])
    L_n = _geo_normalize(L_px, l_inner, l_outer)
    R_n = _geo_normalize(R_px, r_inner, r_outer)

    dist = float(np.linalg.norm(L_px - R_px) / img_w)

    features = np.array([
        L_n[0], L_n[1],
        R_n[0], R_n[1],
        pitch, yaw,
        dist,
    ], dtype=np.float32)

    return features, yaw, pitch, roll


def build_invariant_features(lms, img_w: int = 640, img_h: int = 480):
    """後方互換エイリアス。"""
    return build_7d_features(lms, img_w, img_h)


def extract_7d_from_image(bgr_img: np.ndarray, landmarker) -> Optional[np.ndarray]:
    """バッチ処理用: 1枚のBGR画像から7D特徴量を抽出する (IMAGE mode landmarker)。"""
    h, w = bgr_img.shape[:2]
    rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    out = build_7d_features(lms, w, h)
    if out is None:
        return None
    return out[0]


def _iris_diam_px(lms, edge_indices: list, img_w: int, img_h: int) -> float:
    """虹彩エッジ4点から最大直径（ピクセル）を計算する。"""
    pts = np.array([[lms[i].x * img_w, lms[i].y * img_h] for i in edge_indices])
    max_d = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            max_d = max(max_d, float(np.linalg.norm(pts[i] - pts[j])))
    return max_d


def _iris_h_diam_px(lms, h_indices: list, img_w: int, img_h: int) -> float:
    """虹彩の水平直径（左右エッジのみ）。まぶた遮蔽を受けない。"""
    p0 = np.array([lms[h_indices[0]].x * img_w, lms[h_indices[0]].y * img_h])
    p1 = np.array([lms[h_indices[1]].x * img_w, lms[h_indices[1]].y * img_h])
    return float(np.linalg.norm(p0 - p1))


class GazeFeatureExtractor:
    """
    虹彩深度推定 + solvePnP 頭部姿勢 による幾何学的視線推定（VIDEO mode）。

    アーキテクチャ:
      HeadFilter (OneEuroFilterND, 低周波):
        rvec+tvec (6D) をスムージング → 安定した pitch/yaw ゲート
      gaze_2d = [X_feat, Y_feat] (無次元・距離不変):
        X_feat = (iris_x - cx) / iris_diam_px
        虹彩変位と虹彩直径がどちらも 1/Z に比例 → Z がキャンセル
    """

    def __init__(self, ear_history_len: int = 50):
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = _time.time()
        self._last_ts_ms = -1
        self._ear_hist: deque = deque(maxlen=ear_history_len)

        # 頭部姿勢フィルタ (rvec 3D + tvec 3D = 6D)
        self._head_filter = OneEuroFilterND(6, min_cutoff=0.3, beta=0.01)

    def extract(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        フレームから gaze_2d = [X_feat, Y_feat] を抽出する。
        無次元・距離不変: X_feat = (iris_x - cx) / iris_diam_px

        Returns:
            gaze_2d : np.ndarray (2,) or None
            debug   : dict
        """
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        prev_ts_ms       = self._last_ts_ms
        ts_ms            = int((_time.time() - self._start_time) * 1000)
        ts_ms            = max(ts_ms, prev_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result           = self._landmarker.detect_for_video(mp_img, ts_ms)

        dt_s = float(np.clip((ts_ms - prev_ts_ms) / 1000.0, 1.0/120.0, 0.5)) \
               if prev_ts_ms >= 0 else 1.0 / 30.0

        if not result.face_landmarks:
            return None, None
        lms = result.face_landmarks[0]
        if len(lms) < 478:
            return None, None

        f_px = float(w)
        cx, cy = w / 2.0, h / 2.0
        cam = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1.0]], dtype=np.float64)

        # ── solvePnP for head pose (HeadFilter 適用) ─────────────────────────
        face_2d = np.array([[lms[i].x * w, lms[i].y * h] for i in _FACE_2D_IDX],
                            dtype=np.float64)
        ok, rvec_raw, tvec_raw = cv2.solvePnP(
            _FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None

        head_raw    = np.concatenate([rvec_raw.flatten(), tvec_raw.flatten()])
        head_smooth = self._head_filter.update(head_raw, dt_s)
        rvec_s      = head_smooth[:3].reshape(3, 1)

        R_mat, _ = cv2.Rodrigues(rvec_s)
        angles, *_ = cv2.RQDecomp3x3(R_mat)
        pitch = float(angles[0]) * np.pi / 180.0
        yaw   = float(angles[1]) * np.pi / 180.0
        roll  = float(angles[2]) * np.pi / 180.0
        if pitch > np.pi / 2:
            pitch = np.pi - pitch
        elif pitch < -np.pi / 2:
            pitch = -np.pi - pitch

        # ── 虹彩中心 + 直径 (EyeFilter 適用) ────────────────────────────────
        def iris_center(indices):
            pts = np.array([[lms[i].x * w, lms[i].y * h] for i in indices])
            return pts.mean(axis=0)

        L_c_raw = iris_center(LEFT_IRIS_EDGES)
        R_c_raw = iris_center(RIGHT_IRIS_EDGES)
        L_d_raw = _iris_diam_px(lms, LEFT_IRIS_EDGE4,  w, h)
        R_d_raw = _iris_diam_px(lms, RIGHT_IRIS_EDGE4, w, h)

        if L_d_raw < 2.0 or R_d_raw < 2.0:
            return None, None

        L_cx, L_cy = float(L_c_raw[0]), float(L_c_raw[1])
        R_cx, R_cy = float(R_c_raw[0]), float(R_c_raw[1])
        L_diam     = max(L_d_raw, 2.0)
        R_diam     = max(R_d_raw, 2.0)

        # ── 距離不変の無次元特徴量 ──────────────────────────────────────────
        # X_feat = (iris_x - cx) / iris_diam_px
        # 虹彩変位も虹彩直径も 1/Z に比例するため比が距離をキャンセル
        X_feat = ((L_cx - cx) / L_diam + (R_cx - cx) / R_diam) / 2.0
        Y_feat = ((L_cy - cy) / L_diam + (R_cy - cy) / R_diam) / 2.0

        gaze_2d = np.array([X_feat, Y_feat], dtype=np.float32)

        Z_iris_L = f_px * IRIS_DIAMETER_MM / L_diam
        Z_iris_R = f_px * IRIS_DIAMETER_MM / R_diam

        # ── EAR 瞬き検出 ────────────────────────────────────────────────────
        def pt(i): return np.array([lms[i].x, lms[i].y])
        l_ear = (np.linalg.norm(pt(_L_TOP) - pt(_L_BOT)) /
                 (np.linalg.norm(pt(_L_OUTER) - pt(_L_INNER)) + 1e-9))
        r_ear = (np.linalg.norm(pt(_R_TOP) - pt(_R_BOT)) /
                 (np.linalg.norm(pt(_R_OUTER) - pt(_R_INNER)) + 1e-9))
        ear   = (l_ear + r_ear) / 2.0
        self._ear_hist.append(ear)
        thr   = float(np.mean(self._ear_hist)) * 0.8 if len(self._ear_hist) >= 15 else 0.2
        blink = ear < thr

        debug = {
            'pitch_rad':      pitch,
            'yaw_rad':        yaw,
            'roll_rad':       roll,
            'pitch_deg':      float(np.degrees(pitch)),
            'yaw_deg':        float(np.degrees(yaw)),
            'blink_detected': blink,
            'ear':            ear,
            'landmarks':      lms,
            'X_feat':         float(X_feat),
            'Y_feat':         float(Y_feat),
            'Z_iris_L_mm':    float(Z_iris_L),
            'Z_iris_R_mm':    float(Z_iris_R),
            'iris_diam_L_px': float(L_diam),
            'iris_diam_R_px': float(R_diam),
        }
        return gaze_2d, debug

    def close(self):
        self._landmarker.close()
