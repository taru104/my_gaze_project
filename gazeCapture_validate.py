"""
GazeCapture validation pipeline.

Phase 1 — Feature extraction (one-time, ~30-60 min):
    For each valid frame (faceGrid IsValid==1):
      load JPEG → MediaPipe IMAGE mode → 7D features
    Save to gazeCapture_features_cache.npz  (~15 MB, reused on re-runs)

Phase 2 — Model evaluation:
    A. Ridge regression  (our calibration pipeline at scale)
    B. GazeMLP           (7→36→16→16→2)
    Metrics: RMSE (normalized), MGAE (degrees), Euclidean error (cm)

Phase 3 — Save validation_results.txt + beep.

Usage:
    python gazeCapture_validate.py
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_DIR  = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from gazeCapture_dataset import GazeCaptureRawIndex, GazeCaptureFeatureDataset, SPLIT_CODE
from calibration import RidgeCalibration
from train       import compute_mgae, compute_rmse, GazeTrainer

ARCHIVE_DIR  = PROJECT_DIR / 'archive'
CACHE_PATH   = PROJECT_DIR / 'gazeCapture_features_cache.npz'
RESULTS_PATH = PROJECT_DIR / 'validation_results.txt'
MODEL_PATH   = PROJECT_DIR / 'face_landmarker.task'

# ─── Geometry constants (same as features.py) ────────────────────────────────

_FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0  ],
    [-225.0, 170.0, -135.0 ],
    [225.0,  170.0, -135.0 ],
    [-150.0,-150.0, -125.0 ],
    [150.0, -150.0, -125.0 ],
], dtype=np.float64)

_FACE_2D_IDX    = [1, 152, 33, 263, 61, 291]
_LEFT_IRIS      = [468, 469, 470, 471, 472]
_RIGHT_IRIS     = [473, 474, 475, 476, 477]
_L_EYE_INNER   = 133
_L_EYE_OUTER   = 33
_R_EYE_INNER   = 362
_R_EYE_OUTER   = 263
_DIST_COEFFS   = np.zeros((4, 1), dtype=np.float64)


def _geo_normalize(pupil, inner, outer):
    vec    = outer - inner
    length = np.linalg.norm(vec) + 1e-8
    center = (inner + outer) / 2.0
    rel    = pupil - center
    ang    = np.arctan2(vec[1], vec[0])
    ca, sa = np.cos(-ang), np.sin(-ang)
    rot    = np.array([[ca, -sa], [sa, ca]])
    return (rot @ rel) / length


def _rotate_to_portrait(bgr_img: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate image so that the face is upright (portrait coordinate frame)."""
    if orientation == 1:
        return bgr_img
    elif orientation == 2:
        return cv2.rotate(bgr_img, cv2.ROTATE_180)
    elif orientation == 3:   # LandscapeLeft → rotate CW
        return cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE)
    else:                    # orientation == 4, LandscapeRight → rotate CCW
        return cv2.rotate(bgr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def extract_7d(bgr_img: np.ndarray, landmarker, orientation: int = 1) -> np.ndarray:
    """
    Extract 7D features from a single BGR image using IMAGE-mode MediaPipe.
    Image is rotated to portrait orientation before landmark detection.
    Returns (7,) float32 or None if detection fails.
    """
    bgr_img = _rotate_to_portrait(bgr_img, orientation)
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    if len(lms) < 478:
        return None

    # Pinhole camera matrix (focal ≈ image width)
    f   = float(w)
    cam = np.array([[f, 0, w / 2.0],
                    [0, f, h / 2.0],
                    [0, 0, 1.0    ]], dtype=np.float64)

    # Iris centers
    def iris_center(indices):
        pts = np.array([[lms[i].x * w, lms[i].y * h] for i in indices])
        return pts.mean(axis=0)

    L_px = iris_center(_LEFT_IRIS)
    R_px = iris_center(_RIGHT_IRIS)

    # Head pose via solvePnP
    face_2d = np.array([[lms[i].x * w, lms[i].y * h] for i in _FACE_2D_IDX],
                        dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST_COEFFS,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) * np.pi / 180.0
    yaw   = float(angles[1]) * np.pi / 180.0
    # RQDecomp3x3 can return pitch near ±180° (flipped convention).
    # Normalise to (-π/2, π/2] so the feature is physically meaningful.
    if pitch > np.pi / 2:
        pitch = np.pi - pitch
    elif pitch < -np.pi / 2:
        pitch = -np.pi - pitch

    # Geometric normalization
    l_inner = np.array([lms[_L_EYE_INNER].x * w, lms[_L_EYE_INNER].y * h])
    l_outer = np.array([lms[_L_EYE_OUTER].x * w, lms[_L_EYE_OUTER].y * h])
    r_inner = np.array([lms[_R_EYE_INNER].x * w, lms[_R_EYE_INNER].y * h])
    r_outer = np.array([lms[_R_EYE_OUTER].x * w, lms[_R_EYE_OUTER].y * h])
    L_n = _geo_normalize(L_px, l_inner, l_outer)
    R_n = _geo_normalize(R_px, r_inner, r_outer)

    dist = float(np.linalg.norm(L_px - R_px) / w)

    return np.array([L_n[0], L_n[1], R_n[0], R_n[1], pitch, yaw, dist],
                    dtype=np.float32)


# ─── Phase 1: Feature extraction ────────────────────────────────────────────

CHECKPOINT_PATH = PROJECT_DIR / 'gazeCapture_features_checkpoint.npz'
CHECKPOINT_EVERY = 20_000  # save partial results every N successfully extracted frames


def _save_checkpoint(X_list, y_norm_list, y_cm_list, sc_list):
    np.savez_compressed(
        str(CHECKPOINT_PATH),
        X=np.array(X_list, dtype=np.float32),
        y_norm=np.array(y_norm_list, dtype=np.float32),
        y_cm=np.array(y_cm_list, dtype=np.float32),
        split_code=np.array(sc_list, dtype=np.int32),
    )


def run_extraction() -> None:
    """
    Iterate through all valid frames one at a time (no full-RAM load),
    extract 7D MediaPipe features, save to CACHE_PATH.
    Saves a checkpoint every CHECKPOINT_EVERY frames so crashes don't lose work.
    If a checkpoint exists, extraction resumes from the first un-processed frame.
    """
    index   = GazeCaptureRawIndex(str(ARCHIVE_DIR))
    records = index.records
    n_total = len(records)

    # Resume from checkpoint if available
    X_list, y_norm_list, y_cm_list, sc_list = [], [], [], []
    start_idx = 0
    if CHECKPOINT_PATH.exists():
        ck = np.load(str(CHECKPOINT_PATH))
        X_list       = list(ck['X'])
        y_norm_list  = list(ck['y_norm'])
        y_cm_list    = list(ck['y_cm'])
        sc_list      = list(ck['split_code'])
        # Checkpoint stores OK frames; we approximate resume index by assuming
        # ~93% detection rate (from index validity) — may re-process a few frames.
        start_idx = int(len(X_list) / 0.93)
        print(f"[Extract] Resuming from checkpoint: {len(X_list)} frames already extracted."
              f"  Starting at record ~{start_idx}")
    else:
        print(f"\n[Extract] Starting fresh. {n_total} frames to process.")

    print(f"  Model: {MODEL_PATH}")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    n_ok   = len(X_list)
    n_fail = 0
    t0     = time.time()
    next_ck = n_ok + CHECKPOINT_EVERY

    for i, record in enumerate(records[start_idx:], start=start_idx):
        img_path, xn, yn, xc, yc, sc = record[:6]
        ori = record[6] if len(record) > 6 else 1

        if i > start_idx and (i - start_idx) % 2000 == 0:
            elapsed = time.time() - t0
            fps = max((i - start_idx) / elapsed, 0.01)
            eta_min = (n_total - i) / fps / 60.0
            pct = 100.0 * i / n_total
            print(f"  [{pct:5.1f}%] {i}/{n_total}  ok={n_ok}  fail={n_fail}"
                  f"  {fps:.1f} fps  ETA {eta_min:.1f} min")

        img = cv2.imread(img_path)
        if img is None:
            n_fail += 1
            continue

        feats = extract_7d(img, landmarker, orientation=ori)
        if feats is None:
            n_fail += 1
            continue

        X_list.append(feats)
        y_norm_list.append([xn, yn])
        y_cm_list.append([xc, yc])
        sc_list.append(sc)
        n_ok += 1

        # Periodic checkpoint
        if n_ok >= next_ck:
            _save_checkpoint(X_list, y_norm_list, y_cm_list, sc_list)
            print(f"  [Checkpoint] {n_ok} frames saved → {CHECKPOINT_PATH}")
            next_ck = n_ok + CHECKPOINT_EVERY

    landmarker.close()

    X          = np.array(X_list,      dtype=np.float32)
    y_norm     = np.array(y_norm_list, dtype=np.float32)
    y_cm       = np.array(y_cm_list,   dtype=np.float32)
    split_code = np.array(sc_list,     dtype=np.int32)

    np.savez_compressed(str(CACHE_PATH),
                        X=X, y_norm=y_norm, y_cm=y_cm, split_code=split_code)

    elapsed = time.time() - t0
    print(f"\n[Extract] Done in {elapsed/60:.1f} min")
    print(f"  Frames: total={n_total}  ok={n_ok}  fail={n_fail}"
          f"  ({100*n_fail/n_total:.1f}% detection failure)")
    print(f"  Cache saved → {CACHE_PATH}  ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")


# ─── Metrics ────────────────────────────────────────────────────────────────

def euclidean_cm(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Euclidean distance in cm (standard GazeCapture metric)."""
    return float(np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=-1))))


# ─── Phase 2: Evaluation ────────────────────────────────────────────────────

def run_evaluation(cache_path: Path) -> dict:
    data = np.load(str(cache_path))
    X          = data['X']
    y_norm     = data['y_norm']
    y_cm       = data['y_cm']
    split_code = data['split_code']

    train_m = split_code == SPLIT_CODE['train']
    val_m   = split_code == SPLIT_CODE['val']
    test_m  = split_code == SPLIT_CODE['test']

    X_tr,  y_n_tr,  y_c_tr  = X[train_m], y_norm[train_m], y_cm[train_m]
    X_va,  y_n_va,  y_c_va  = X[val_m],   y_norm[val_m],   y_cm[val_m]
    X_te,  y_n_te,  y_c_te  = X[test_m],  y_norm[test_m],  y_cm[test_m]

    print(f"\n[Eval] train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    results = {}

    # ── A. Ridge regression (normalized targets) ────────────────────────────
    print("\n-- A. Ridge Regression (7D → StandardScaler → 36D → 2D norm) --")
    ridge_n = RidgeCalibration(lambda_reg=0.47)
    ridge_n.fit(X_tr, y_n_tr)
    pred_n  = ridge_n.predict(X_te)
    r_rmse  = compute_rmse(pred_n, y_n_te)
    r_mgae  = compute_mgae(pred_n, y_n_te)
    print(f"  RMSE (norm):   {r_rmse:.5f}")
    print(f"  MGAE (deg):    {r_mgae:.3f}")

    # Ridge on cm targets for Euclidean metric
    ridge_c = RidgeCalibration(lambda_reg=0.47)
    ridge_c.fit(X_tr, y_c_tr)
    pred_c  = ridge_c.predict(X_te)
    r_euc   = euclidean_cm(pred_c, y_c_te)
    print(f"  Euclidean (cm): {r_euc:.3f}")

    results['ridge'] = dict(rmse_norm=r_rmse, mgae_deg=r_mgae, euclidean_cm=r_euc)

    # ── B. GazeMLP training ─────────────────────────────────────────────────
    print("\n-- B. GazeMLP (7→36→16→16→2, gamma=1.0) --")
    trainer = GazeTrainer(gamma=1.0, hidden_dim=16)
    # Use val split (if large enough) otherwise 10% of train
    if len(X_va) > 200:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_n_tr, dtype=torch.float32),
        )
        val_ds   = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_n_va, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
    else:
        train_loader, val_loader = trainer.make_loaders(
            X_tr, y_n_tr, val_ratio=0.1, batch_size=512
        )

    trainer.train(train_loader, val_loader, epochs=80, lr=1e-3, patience=15)
    mlp_metrics = trainer.evaluate(X_te, y_n_te)
    m_rmse = mlp_metrics['rmse']
    m_mgae = mlp_metrics['mgae']
    print(f"  RMSE (norm): {m_rmse:.5f}")
    print(f"  MGAE (deg):  {m_mgae:.3f}")

    # Euclidean in cm: retrain MLP on cm targets
    trainer_cm = GazeTrainer(gamma=1.0, hidden_dim=16)
    tr_l_cm, va_l_cm = trainer_cm.make_loaders(X_tr, y_c_tr, val_ratio=0.1, batch_size=512)
    trainer_cm.train(tr_l_cm, va_l_cm, epochs=80, lr=1e-3, patience=15)
    import torch
    with torch.no_grad():
        Xt = torch.tensor(X_te, dtype=torch.float32).to(trainer_cm.device)
        pred_cm_mlp = trainer_cm.model(Xt).cpu().numpy()
    m_euc = euclidean_cm(pred_cm_mlp, y_c_te)
    print(f"  Euclidean (cm): {m_euc:.3f}")

    results['gazemlp'] = dict(rmse_norm=m_rmse, mgae_deg=m_mgae, euclidean_cm=m_euc)

    return results


# ─── Phase 3: Save results ───────────────────────────────────────────────────

def save_results(results: dict, elapsed_total: float, cache_path: Path) -> None:
    data = np.load(str(cache_path))
    sc   = data['split_code']
    n_tr = int((sc == SPLIT_CODE['train']).sum())
    n_va = int((sc == SPLIT_CODE['val']).sum())
    n_te = int((sc == SPLIT_CODE['test']).sum())

    lines = [
        "=" * 60,
        "GazeCapture Validation Results",
        f"Date       : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime    : {elapsed_total/60:.1f} min",
        "=" * 60,
        "",
        "Dataset",
        f"  Archive  : {ARCHIVE_DIR}",
        f"  Validity : faceGrid IsValid == 1 (face + eye detected)",
        f"  train    : {n_tr} frames",
        f"  val      : {n_va} frames",
        f"  test     : {n_te} frames",
        "",
        "Feature extraction",
        "  Tool     : MediaPipe FaceLandmarker (IMAGE mode)",
        "  Features : 7D [Lx Ly Rx Ry Pitch Yaw dist]",
        "  Iris     : 478-landmark model, landmarks 468-477",
        "  Pose     : solvePnP(6-point) → RQDecomp3x3",
        "  Normalization: geometric (eye-corner-based)",
        "",
        "─" * 60,
        "A. Ridge Regression  (lambda=0.47, StandardScaler + poly36)",
        "─" * 60,
        f"  RMSE  (normalized [0,1]) : {results['ridge']['rmse_norm']:.5f}",
        f"  MGAE  (degrees)          : {results['ridge']['mgae_deg']:.3f}",
        f"  Euclidean error (cm)     : {results['ridge']['euclidean_cm']:.3f}",
        "",
        "─" * 60,
        "B. GazeMLP  (7→poly36→FC16→FC16→2, gamma=1.0, 80 epochs)",
        "─" * 60,
        f"  RMSE  (normalized [0,1]) : {results['gazemlp']['rmse_norm']:.5f}",
        f"  MGAE  (degrees)          : {results['gazemlp']['mgae_deg']:.3f}",
        f"  Euclidean error (cm)     : {results['gazemlp']['euclidean_cm']:.3f}",
        "",
        "─" * 60,
        "Reference (GazeCapture paper, iTracker CNN)",
        "  Euclidean error (cm) : 2.53 cm (phone) / 3.02 cm (tablet)",
        "─" * 60,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_PATH.write_text(text, encoding='utf-8')
    print(f"\n[Done] Results saved → {RESULTS_PATH}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Phase 1: extract features (skip if cache exists)
    if CACHE_PATH.exists():
        print(f"[Cache] Found existing cache: {CACHE_PATH}"
              f"  ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)  — skipping extraction.")
    else:
        run_extraction()

    # Phase 2: evaluate
    results = run_evaluation(CACHE_PATH)

    # Phase 3: save
    save_results(results, time.time() - t0, CACHE_PATH)

    # Beep to notify user
    subprocess.run(
        ['powershell', '-Command', '[System.Console]::Beep(1000, 500)'],
        capture_output=True,
    )


if __name__ == '__main__':
    main()
