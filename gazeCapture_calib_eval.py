"""
GazeCapture per-subject calibration benchmark.

Real pipeline simulation:
  1. Take 5% of a subject's frames as calibration
  2. Fit TargetedPolyCalibration (same model as production)
     Features: X_feat = avg(L_n_x, R_n_x), Y_feat = avg(L_n_y, R_n_y), pitch
  3. Test on remaining 95%
  4. Report MGAE across subjects

Usage: python gazeCapture_calib_eval.py
Extracts ~N_FRAMES_CAP frames fresh (orientation-corrected).
"""
import sys, time, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from gazeCapture_dataset import GazeCaptureRawIndex
from gazeCapture_validate import extract_7d
from calibration import TargetedPolyCalibration

PROJECT_DIR   = Path(__file__).parent
ARCHIVE_DIR   = PROJECT_DIR / 'archive'
MODEL_PATH    = PROJECT_DIR / 'face_landmarker.task'

N_FRAMES_CAP  = 9000
CALIB_RATIO   = 0.05
MIN_CALIB     = 9      # need >= params for overdetermined system
MIN_TEST      = 30
SPLIT_CODE_TEST = 2


def mgae(preds, targets):
    angs = []
    for pr, tg in zip(preds, targets):
        v1 = np.array([pr[0]-0.5, pr[1]-0.5, 1.0])
        v2 = np.array([tg[0]-0.5, tg[1]-0.5, 1.0])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
        angs.append(float(np.degrees(np.arccos(np.clip(cos, -1+1e-7, 1-1e-7)))))
    return float(np.mean(angs)) if angs else 999.0


def feat_to_3d(feat7):
    """7D [Lx, Ly, Rx, Ry, pitch, yaw, dist] → 3D [X_feat, Y_feat, pitch]."""
    X_feat = (feat7[0] + feat7[2]) / 2.0
    Y_feat = (feat7[1] + feat7[3]) / 2.0
    pitch  = feat7[4]
    return X_feat, Y_feat, pitch


# ── 1. Build index ────────────────────────────────────────────────────────────
print("Building index...")
idx = GazeCaptureRawIndex(str(ARCHIVE_DIR))

# Group test records by subject directory
subj_records = defaultdict(list)
for r in idx.records:
    if r[5] != SPLIT_CODE_TEST:
        continue
    subj = str(Path(r[0]).parent.parent)
    subj_records[subj].append(r)

valid_subjs = {k: v for k, v in subj_records.items()
               if len(v) >= MIN_CALIB + MIN_TEST}
print(f"Valid test subjects: {len(valid_subjs)}  "
      f"(total frames: {sum(len(v) for v in valid_subjs.values())})")

# ── 2. Subsample to N_FRAMES_CAP total ───────────────────────────────────────
rng = np.random.RandomState(42)
subj_list   = sorted(valid_subjs.keys())
total_avail = sum(len(valid_subjs[s]) for s in subj_list)
selected    = {}
n_selected  = 0

for subj in subj_list:
    records = valid_subjs[subj]
    n_take  = max(MIN_CALIB + MIN_TEST,
                  round(len(records) * N_FRAMES_CAP / total_avail))
    n_take  = min(n_take, len(records))
    idxs    = rng.choice(len(records), n_take, replace=False)
    selected[subj] = [records[i] for i in sorted(idxs)]
    n_selected += n_take
    if n_selected >= N_FRAMES_CAP:
        break

print(f"Extracting {n_selected} frames from {len(selected)} subjects...")

# ── 3. Extract features ───────────────────────────────────────────────────────
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

t0 = time.time()
subj_data = {}
n_ok = n_fail = 0

for subj, records in selected.items():
    Xs, ys = [], []
    for r in records:
        img_path, xn, yn = r[0], r[1], r[2]
        ori = r[6] if len(r) > 6 else 1
        img = cv2.imread(img_path)
        if img is None:
            n_fail += 1
            continue
        feat7 = extract_7d(img, landmarker, orientation=ori)
        if feat7 is None:
            n_fail += 1
            continue
        Xs.append(feat_to_3d(feat7))
        ys.append([xn, yn])
        n_ok += 1

    if len(Xs) >= MIN_CALIB + MIN_TEST:
        subj_data[subj] = (np.array(Xs), np.array(ys))

landmarker.close()
elapsed = time.time() - t0
print(f"Done: {n_ok} ok, {n_fail} fail  ({n_ok/elapsed:.1f} fps, {elapsed:.0f}s)")

# ── 4. Per-subject calibration + evaluation ───────────────────────────────────
mgae_list  = []
no_cal_list = []  # baseline: predict mean of calib targets

for subj, (X, y) in subj_data.items():
    n = len(X)
    n_calib = max(MIN_CALIB, int(n * CALIB_RATIO))

    perm       = rng.permutation(n)
    calib_idx  = perm[:n_calib]
    test_idx   = perm[n_calib:]

    if len(test_idx) < MIN_TEST:
        continue

    # Fit TargetedPolyCalibration
    cal = TargetedPolyCalibration(alpha=1.0)
    for Xf, Yf, p in X[calib_idx]:
        idx_row = np.where((X == [Xf, Yf, p]).all(axis=1))[0]
        cal.add(Xf, Yf, 0.0, 0.0)   # dummy target
    # Re-add with correct targets
    cal = TargetedPolyCalibration(alpha=1.0)
    for (Xf, Yf, p), tgt in zip(X[calib_idx], y[calib_idx]):
        cal.add(float(Xf), float(Yf), float(tgt[0]), float(tgt[1]),
                weight=1.0, pitch_rad=float(p))
    cal.fit()

    preds = [cal.predict(float(Xf), float(Yf), float(p))
             for Xf, Yf, p in X[test_idx]]
    m = mgae(preds, y[test_idx])
    mgae_list.append(m)

    # No-calibration baseline: predict mean of calibration targets
    mean_pred = np.mean(y[calib_idx], axis=0)
    no_cal = mgae([mean_pred] * len(test_idx), y[test_idx])
    no_cal_list.append(no_cal)

sep = '=' * 55
print()
print(sep)
print('  GazeCapture Per-Subject Calibration Benchmark')
print(f'  Subjects: {len(mgae_list)}   Calib: {CALIB_RATIO:.0%} of frames')
print(sep)
print(f'  TargetedPoly MGAE : {np.mean(mgae_list):.2f} +- {np.std(mgae_list):.2f} deg')
print(f'  TargetedPoly median: {np.median(mgae_list):.2f} deg')
print(f'  No-calib baseline : {np.mean(no_cal_list):.2f} deg')
print(f'  Best subject      : {min(mgae_list):.2f} deg')
print(f'  Worst subject     : {max(mgae_list):.2f} deg')
print(sep)
