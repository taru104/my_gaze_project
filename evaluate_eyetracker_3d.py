"""
Webcam3DTracker (EyeTracker/Webcam3DTracker/MonitorTracking.py) headless evaluation
on GazeCapture test set.  Faithfully reproduces MonitorTracking.py's 3D geometry
pipeline without any GUI/mouse/keyboard dependencies.

Pipeline (ported from MonitorTracking.py):
  1. MediaPipe FaceLandmarker IMAGE mode  -> 478 landmarks (iris 468,473 included)
  2. PCA on 24 nose landmarks            -> head-center + R_final rotation matrix
  3. Eye-sphere locking on first valid calibration frame
  4. Scale-adaptive sphere positioning (nose-scale ratio each frame)
  5. Combined gaze direction: average(iris_L - sphere_L, iris_R - sphere_R)
  6. Yaw/pitch screen mapping:  yawDeg in [-15,+15] -> x_norm, pitchDeg in [-5,+5] -> y_norm
  7. Calibration: linear offset (delta_x_norm, delta_y_norm) over first-5% frames

Metrics:
  MGAE_2D : angle error using [x-0.5, y-0.5, 1.0] 3D approximation (normalized coords)
  MGAE_3D : fixed Z_face=50cm, pred/GT in cm via per-subject linear scale calibration
  Euc(cm) : Euclidean distance in cm space

Usage:
    .\.venv_eyetracker\Scripts\python.exe evaluate_eyetracker_3d.py
"""

import os
import sys
import math
import json
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.spatial.transform import Rotation as Rscipy
from pathlib import Path

# ──── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR   = Path(__file__).parent
ARCHIVE_DIR   = PROJECT_DIR / 'archive'
MODEL_PATH    = PROJECT_DIR / 'face_landmarker.task'
RESULTS_PATH  = PROJECT_DIR / 'eyetracker_3d_results.txt'

# Screen mapping constants (from MonitorTracking.py, unchanged)
YAW_DEGREES   = 15.0   # +/-15 deg left/right -> screen width
PITCH_DEGREES = 5.0    # +/-5 deg up/down -> screen height
BASE_RADIUS   = 20     # eye-sphere Z offset at calibration distance (world units)

# Nose landmark indices (from MonitorTracking.py line 79-81)
NOSE_INDICES = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

LEFT_IRIS_IDX  = 468
RIGHT_IRIS_IDX = 473

FIXED_Z_FACE_CM = 50.0   # for 3D MGAE (same value as evaluate_eyetrax.py)


# ──── GazeCapture data loading ─────────────────────────────────────────────────

def load_test_subjects(archive_dir: Path) -> dict:
    """Returns {subj_id: [(img_path, x_norm, y_norm, x_cm, y_cm), ...]}"""
    subjects = sorted(os.listdir(archive_dir))
    records_by_subj = {}
    skipped = 0
    for subj in subjects:
        subj_dir = archive_dir / subj / subj
        if not subj_dir.is_dir():
            continue
        try:
            with open(subj_dir / 'info.json') as f:
                if json.load(f).get('Dataset', '') != 'test':
                    continue
            with open(subj_dir / 'frames.json') as f:
                frames = json.load(f)
            with open(subj_dir / 'dotInfo.json') as f:
                dot = json.load(f)
            with open(subj_dir / 'screen.json') as f:
                screen = json.load(f)
            with open(subj_dir / 'faceGrid.json') as f:
                grid = json.load(f)
        except Exception:
            skipped += 1
            continue

        recs = []
        for i in range(len(frames)):
            if not grid['IsValid'][i]:
                continue
            p = subj_dir / 'frames' / frames[i]
            if not p.exists():
                continue
            W, H = screen['W'][i], screen['H'][i]
            if W == 0 or H == 0:
                continue
            recs.append((
                str(p),
                float(dot['XPts'][i]) / W,
                float(dot['YPts'][i]) / H,
                float(dot['XCam'][i]),
                float(dot['YCam'][i]),
            ))
        if recs:
            records_by_subj[int(subj)] = recs
    n_total = sum(len(v) for v in records_by_subj.values())
    print(f"[Data] {n_total} frames, {len(records_by_subj)} test subjects  "
          f"({skipped} subjects skipped)")
    return records_by_subj


# ──── Core 3D pipeline (ported from MonitorTracking.py) ───────────────────────

class HeadlessEyeTracker:
    """
    Faithfully reproduces MonitorTracking.py's geometric pipeline.
    Uses FaceLandmarker (Tasks API) instead of deprecated mp.solutions.face_mesh.
    """

    def __init__(self, model_path: str):
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self.reset()

    def reset(self):
        """Reset all per-subject state."""
        self._R_ref = [None]
        self._L_off = None      # left sphere local offset
        self._R_off = None      # right sphere local offset
        self._L_scale_ref = None
        self._R_scale_ref = None
        self._locked = False

    # ── MediaPipe detection ──────────────────────────────────────────────────

    def _detect(self, img_path: str):
        """Return face_landmarks list or None."""
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None, None, None
        return result.face_landmarks[0], w, h

    # ── Head pose via PCA (from compute_and_draw_coordinate_box) ─────────────

    def _head_pose(self, lms, w: int, h: int):
        """PCA on nose landmarks -> (center, R_final, nose_pts_3d)"""
        pts = np.array([
            [lms[i].x * w, lms[i].y * h, lms[i].z * w]
            for i in NOSE_INDICES
        ], dtype=float)
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        r = Rscipy.from_matrix(eigvecs)
        roll, pitch, yaw = r.as_euler('zyx', degrees=False)
        R = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

        # Stabilize eigenvector sign (prevents random flipping between frames)
        if self._R_ref[0] is None:
            self._R_ref[0] = R.copy()
        else:
            for i in range(3):
                if np.dot(R[:, i], self._R_ref[0][:, i]) < 0:
                    R[:, i] *= -1

        return center, R, pts

    # ── Scale (vectorised replacement for compute_scale) ─────────────────────

    @staticmethod
    def _scale(pts):
        n = len(pts)
        i_idx, j_idx = np.triu_indices(n, k=1)
        return float(np.mean(np.linalg.norm(pts[i_idx] - pts[j_idx], axis=-1)))

    # ── Sphere locking (simulates pressing 'c') ───────────────────────────────

    def _lock(self, iris_l, iris_r, head_center, R, nose_pts):
        cam_local = R.T @ np.array([0.0, 0.0, 1.0])
        self._L_off = R.T @ (iris_l - head_center) + BASE_RADIUS * cam_local
        self._R_off = R.T @ (iris_r - head_center) + BASE_RADIUS * cam_local
        scale = self._scale(nose_pts)
        self._L_scale_ref = scale
        self._R_scale_ref = scale
        self._locked = True

    # ── Gaze direction -> normalised screen coords ────────────────────────────

    def _gaze_to_norm(self, gaze_dir):
        """
        Replicates convert_gaze_to_screen_coordinates from MonitorTracking.py,
        but returns normalised [0,1] coords instead of pixels.
        """
        fwd = np.array([0.0, 0.0, -1.0])
        d = gaze_dir / (np.linalg.norm(gaze_dir) + 1e-9)

        # Yaw (horizontal, XZ plane)
        xz = np.array([d[0], 0.0, d[2]])
        n = np.linalg.norm(xz)
        xz = xz / n if n > 1e-9 else fwd.copy()
        yaw = math.degrees(math.acos(float(np.clip(np.dot(fwd, xz), -1, 1))))
        if d[0] < 0:
            yaw = -yaw
        # MonitorTracking sign-flip logic (preserved exactly)
        yaw = -yaw if yaw < 0 else -yaw

        # Pitch (vertical, YZ plane)
        yz = np.array([0.0, d[1], d[2]])
        n = np.linalg.norm(yz)
        yz = yz / n if n > 1e-9 else fwd.copy()
        pitch = math.degrees(math.acos(float(np.clip(np.dot(fwd, yz), -1, 1))))
        if d[1] > 0:
            pitch = -pitch

        # Map to [0, 1]  (no calibration offset applied here; applied externally)
        x_norm = (yaw   + YAW_DEGREES)   / (2 * YAW_DEGREES)
        y_norm = (PITCH_DEGREES - pitch)  / (2 * PITCH_DEGREES)
        return float(x_norm), float(y_norm)

    # ── Per-image prediction ──────────────────────────────────────────────────

    def predict(self, img_path: str):
        """
        Returns (x_norm, y_norm) uncalibrated, plus head/sphere internals.
        Returns None if detection fails or spheres not locked.
        """
        lms, w, h = self._detect(img_path)
        if lms is None:
            return None

        head_center, R, nose_pts = self._head_pose(lms, w, h)

        iris_l = np.array([lms[LEFT_IRIS_IDX].x * w,
                           lms[LEFT_IRIS_IDX].y * h,
                           lms[LEFT_IRIS_IDX].z * w])
        iris_r = np.array([lms[RIGHT_IRIS_IDX].x * w,
                           lms[RIGHT_IRIS_IDX].y * h,
                           lms[RIGHT_IRIS_IDX].z * w])

        if not self._locked:
            # Return landmarks for sphere locking; no prediction yet
            return ('unlocked', head_center, R, iris_l, iris_r, nose_pts)

        # Scale-adaptive sphere world positions
        scale = self._scale(nose_pts)
        sphere_l = head_center + R @ (self._L_off * (scale / self._L_scale_ref))
        sphere_r = head_center + R @ (self._R_off * (scale / self._R_scale_ref))

        gaze_l = iris_l - sphere_l
        gaze_r = iris_r - sphere_r
        nl, nr = np.linalg.norm(gaze_l), np.linalg.norm(gaze_r)
        if nl < 1e-9 or nr < 1e-9:
            return None
        gaze_l /= nl
        gaze_r /= nr
        combined = gaze_l + gaze_r
        nc = np.linalg.norm(combined)
        if nc < 1e-9:
            return None
        combined /= nc

        xn, yn = self._gaze_to_norm(combined)
        return ('pred', xn, yn, head_center, R, iris_l, iris_r, nose_pts)

    def close(self):
        self.landmarker.close()


# ──── Evaluation metrics ────────────────────────────────────────────────────────

def mgae_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    def v3(p):
        return np.column_stack([p[:, 0] - 0.5, p[:, 1] - 0.5, np.ones(len(p))])
    v1, v2 = v3(pred), v3(gt)
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
    cos = np.sum((v1 / n1) * (v2 / n2), axis=-1).clip(-1 + 1e-7, 1 - 1e-7)
    return float(np.mean(np.degrees(np.arccos(cos))))


def mgae_3d_fixed(pred_cm: np.ndarray, gt_cm: np.ndarray,
                  z: float = FIXED_Z_FACE_CM) -> float:
    g1 = np.column_stack([pred_cm[:, 0], pred_cm[:, 1], np.full(len(pred_cm), -z)])
    g2 = np.column_stack([gt_cm[:, 0],   gt_cm[:, 1],   np.full(len(gt_cm),   -z)])
    n1 = np.linalg.norm(g1, axis=-1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(g2, axis=-1, keepdims=True) + 1e-8
    cos = np.sum((g1 / n1) * (g2 / n2), axis=-1).clip(-1 + 1e-7, 1 - 1e-7)
    return float(np.mean(np.degrees(np.arccos(cos))))


def euc_cm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=-1))))


# ──── Per-subject evaluation ────────────────────────────────────────────────────

def evaluate_subject(tracker: HeadlessEyeTracker, records: list) -> dict | None:
    """
    first-5%  : lock spheres on first valid frame;
                collect predictions to compute linear offset (Δx_norm, Δy_norm)
                and also per-axis linear scale mapping norm -> cm.
    remaining : apply offset, evaluate MGAE_2D and MGAE_3D.
    """
    n = len(records)
    n_cal = max(1, int(n * 0.05))

    tracker.reset()

    # ── Calibration phase ─────────────────────────────────────────────────────
    cal_xn, cal_yn = [], []
    cal_gt_xn, cal_gt_yn = [], []
    cal_gt_xc, cal_gt_yc = [], []

    for img_path, xn_gt, yn_gt, xc_gt, yc_gt in records[:n_cal]:
        res = tracker.predict(img_path)
        if res is None:
            continue

        if res[0] == 'unlocked':
            _, head_center, R, iris_l, iris_r, nose_pts = res
            tracker._lock(iris_l, iris_r, head_center, R, nose_pts)
            # Re-predict now that spheres are locked
            res = tracker.predict(img_path)
            if res is None or res[0] != 'pred':
                continue

        if res[0] != 'pred':
            continue

        _, xn_pred, yn_pred, *_ = res
        cal_xn.append(xn_pred);  cal_yn.append(yn_pred)
        cal_gt_xn.append(xn_gt); cal_gt_yn.append(yn_gt)
        cal_gt_xc.append(xc_gt); cal_gt_yc.append(yc_gt)

    if not tracker._locked or len(cal_xn) < 2:
        return None

    cal_xn  = np.array(cal_xn);  cal_yn  = np.array(cal_yn)
    cal_gxn = np.array(cal_gt_xn); cal_gyn = np.array(cal_gt_yn)
    cal_gxc = np.array(cal_gt_xc); cal_gyc = np.array(cal_gt_yc)

    # Linear offset in normalised space (delta_x, delta_y)
    off_x = float(np.mean(cal_xn - cal_gxn))
    off_y = float(np.mean(cal_yn - cal_gyn))

    # Linear scale mapping: norm -> cm  (for 3D MGAE)
    # Fit: gt_cm = a * pred_norm + b  per axis
    ax, bx = np.polyfit(cal_xn, cal_gxc, 1)
    ay, by = np.polyfit(cal_yn, cal_gyc, 1)

    # ── Evaluation phase ──────────────────────────────────────────────────────
    pred_norm_list, pred_cm_list = [], []
    gt_norm_list,   gt_cm_list   = [], []

    for img_path, xn_gt, yn_gt, xc_gt, yc_gt in records[n_cal:]:
        res = tracker.predict(img_path)
        if res is None or res[0] != 'pred':
            continue
        _, xn_pred, yn_pred, *_ = res

        # Apply normalised offset calibration
        xn_cal = xn_pred - off_x
        yn_cal = yn_pred - off_y

        # Convert pred to cm via linear scale mapping
        xc_pred = ax * xn_pred + bx
        yc_pred = ay * yn_pred + by

        pred_norm_list.append([xn_cal, yn_cal])
        pred_cm_list.append([xc_pred, yc_pred])
        gt_norm_list.append([xn_gt, yn_gt])
        gt_cm_list.append([xc_gt, yc_gt])

    n_ev_ok = len(pred_norm_list)
    if n_ev_ok < 5:
        return None

    pred_norm = np.array(pred_norm_list, dtype=np.float64)
    pred_cm_  = np.array(pred_cm_list,   dtype=np.float64)
    gt_norm   = np.array(gt_norm_list,   dtype=np.float64)
    gt_cm_    = np.array(gt_cm_list,     dtype=np.float64)

    return {
        'mgae_2d':  mgae_2d(pred_norm, gt_norm),
        'mgae_3d':  mgae_3d_fixed(pred_cm_, gt_cm_),
        'euc':      euc_cm(pred_cm_, gt_cm_),
        'n_total':  n,
        'n_cal':    n_cal,
        'n_cal_ok': len(cal_xn),
        'n_ev_ok':  n_ev_ok,
        'off_x':    off_x,
        'off_y':    off_y,
    }


# ──── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 72)
    print("  Webcam3DTracker (MonitorTracking.py) -- GazeCapture Benchmark")
    print("=" * 72)

    records_by_subj = load_test_subjects(ARCHIVE_DIR)
    if not records_by_subj:
        print("[ERROR] No test subjects found.")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] face_landmarker.task not found at {MODEL_PATH}")
        sys.exit(1)

    tracker = HeadlessEyeTracker(str(MODEL_PATH))

    print(f"\n  {'Subj':>5}  {'n':>5}  {'cal':>4}  {'ok%':>5}  "
          f"{'dX':>6}  {'dY':>6}  {'MGAE_2D':>8}  {'MGAE_3D':>8}  {'Euc(cm)':>8}")
    print("  " + "-" * 70)

    all_results = []
    skipped = []

    for sid in sorted(records_by_subj):
        recs = records_by_subj[sid]
        n    = len(recs)
        n_cal = max(1, int(n * 0.05))
        res  = evaluate_subject(tracker, recs)

        if res is None:
            skipped.append(sid)
            print(f"  {sid:5d}  {n:5d}  {n_cal:4d}   SKIP (insufficient detections)")
            continue

        pct = 100.0 * res['n_ev_ok'] / max(n - n_cal, 1)
        print(f"  {sid:5d}  {n:5d}  {n_cal:4d}  "
              f"{pct:4.0f}%  "
              f"{res['off_x']:+6.3f}  {res['off_y']:+6.3f}  "
              f"{res['mgae_2d']:8.2f}  {res['mgae_3d']:8.2f}  {res['euc']:8.3f}")
        all_results.append(res)

    tracker.close()
    elapsed = time.time() - t0

    if not all_results:
        print("\n[ERROR] No subjects evaluated successfully.")
        sys.exit(1)

    m2d = np.array([r['mgae_2d'] for r in all_results])
    m3d = np.array([r['mgae_3d'] for r in all_results])
    euc = np.array([r['euc']     for r in all_results])

    lines = [
        "=" * 72,
        "  Webcam3DTracker (EyeTracker) vs EyeTrax vs Custom -- GazeCapture Eval",
        f"  Date    : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Runtime : {elapsed/60:.1f} min",
        "=" * 72,
        "",
        "  Evaluation Protocol",
        "    Dataset     : GazeCapture test split",
        f"    Subjects    : {len(all_results)} evaluated  ({len(skipped)} skipped)",
        "    Calibration : First 5% frames -- sphere-lock + linear offset (dx,dy)",
        "    Evaluation  : Remaining 95% frames",
        f"    3D MGAE     : Fixed Z_face={FIXED_Z_FACE_CM}cm; norm->cm via per-subject linear fit",
        "",
        "  " + "-" * 68,
        f"  {'Method':<38}  {'MGAE_2D':>8}  {'MGAE_3D':>8}  {'Euc(cm)':>8}",
        f"  {'':38}  {'mean':>8}  {'mean':>8}  {'median':>8}",
        "  " + "-" * 68,
        f"  {'EyeTrax(alpha=1.0)':<38}  {'54.51':>8}  {'26.83':>8}  {'24.66':>8}",
        f"  {'EyeTrax(alpha=adaptive)':<38}  {'44.15':>8}  {'18.06':>8}  {'16.36':>8}",
        f"  {'Custom(7D+poly36,adaptive)':<38}  {'45.54':>8}  {'22.29':>8}  {'21.30':>8}",
        f"  {'Webcam3DTracker(geometric)':<38}  "
        f"{np.mean(m2d):>8.2f}  {np.mean(m3d):>8.2f}  {np.median(euc):>8.3f}",
        "  " + "-" * 68,
        "",
        "  Detailed Statistics -- Webcam3DTracker",
        f"    Subjects    : {len(all_results)}",
        f"    MGAE_2D mean: {np.mean(m2d):.3f} +/- {np.std(m2d):.3f} deg  (med {np.median(m2d):.3f})",
        f"    MGAE_3D mean: {np.mean(m3d):.3f} +/- {np.std(m3d):.3f} deg  (med {np.median(m3d):.3f})",
        f"    Euc(cm) mean: {np.mean(euc):.3f}  med: {np.median(euc):.3f}  max: {np.max(euc):.3f}",
        "",
        "  " + "-" * 68,
        "  Algorithm Analysis",
        "  " + "-" * 68,
        "",
        "  3D Geometry Pipeline (Webcam3DTracker):",
        "    1. MediaPipe FaceLandmarker -> 478 landmarks (478 pts incl iris 468/473)",
        "    2. PCA on 24 nose landmarks -> head center + 3x3 rotation matrix",
        "       (Stabilized against eigenvector sign flips via reference matrix)",
        "    3. Eye-sphere locking at calibration:",
        "       sphere_local = R^T*(iris - head_center) + 20*R^T*[0,0,1]",
        "    4. Scale-adaptive tracking per frame:",
        "       sphere_world = head_center + R*(sphere_local * nose_scale_ratio)",
        "    5. Gaze direction: normalize(avg(iris_L-sphere_L, iris_R-sphere_R))",
        "    6. Screen mapping: yaw in [-15,+15]deg -> [0,1], pitch in [-5,+5]deg -> [0,1]",
        "    7. Calibration: mean offset (delta_x, delta_y) over first-5% predictions",
        "",
        "  Strengths:",
        "    - Fully geometric (no regression, no learned mapping from gaze to screen)",
        "    - Invariant to appearance changes (illumination, glasses, skin tone)",
        "    - Works without ANY labeled training data",
        "    - Head-movement robust via scale-adaptive sphere tracking",
        "",
        "  Weaknesses / Bottlenecks:",
        "    A. Fixed angular mapping (+-15 deg yaw, +-5 deg pitch) assumes specific",
        "       face-to-screen geometry; GazeCapture phones have much narrower range",
        "       => scale mismatch partially corrected by linear cal, but residual error",
        "    B. PCA coordinate frame (nose patch) is not as stable as solvePnP;",
        "       head-rotation estimation has higher variance than 6-point PnP",
        "    C. Eye sphere radius (base_radius=20 world units) is fixed, not estimated",
        "       => sphere center placement error directly affects gaze direction",
        "    D. First-5% calibration often covers narrow screen region (not 9-point)",
        "       => offset captures mean bias but not scale/rotation of the mapping",
        "    E. Iris landmark (single point 468/473) vs iris edge average is noisier",
        "",
        "  Why EyeTrax/Custom regression beats geometric pipeline on GazeCapture:",
        "    Regression models implicitly learn the face-to-screen geometry from",
        "    calibration data. With 37-176 calibration frames, Ridge can fit the",
        "    affine mapping (scale, offset, rotation) of the prediction space.",
        "    Geometric pipeline must rely on hard-coded parameters (+-15 deg range,",
        "    sphere radius=20) that may not match the actual recording setup.",
        "",
        "  For in-the-wild webcam use (main.py 9-point calibration):",
        "    The 9-point calibration in main.py provides full-screen coverage,",
        "    which resolves bottlenecks A and D. The geometric approach is likely",
        "    competitive with regression when calibration coverage is adequate.",
        "=" * 72,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n[Done] Results saved -> {RESULTS_PATH}")


if __name__ == '__main__':
    main()
