"""Quick import and API sanity check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=== Checking features.py ===")
from features import (build_7d_features, extract_7d_from_image,
                      GazeFeatureExtractor, FEATURE_DIM,
                      LEFT_IRIS_EDGES, RIGHT_IRIS_EDGES)
print(f"  FEATURE_DIM = {FEATURE_DIM}")
print(f"  LEFT_IRIS_EDGES  = {LEFT_IRIS_EDGES}")
print(f"  RIGHT_IRIS_EDGES = {RIGHT_IRIS_EDGES}")
print("  OK")

print("\n=== Checking calibration.py ===")
from calibration import (expand_features, RidgeCalibration,
                          CalibrationPipeline, CALIB_POINTS_9)
import numpy as np

r = RidgeCalibration()
print(f"  RidgeCalibration default lambda = {r.lambda_reg}")
assert abs(r.lambda_reg - 0.473) < 1e-9, f"Expected 0.473, got {r.lambda_reg}"

cp = CalibrationPipeline()
print(f"  CalibrationPipeline default lambda = {cp.ridge.lambda_reg}")
assert abs(cp.ridge.lambda_reg - 0.473) < 1e-9

x7 = np.random.randn(7).astype(np.float32)
psi = expand_features(x7)
print(f"  expand_features(7D) → shape {psi.shape}")
assert psi.shape == (36,), f"Expected (36,), got {psi.shape}"

X7 = np.random.randn(10, 7).astype(np.float32)
psi_batch = expand_features(X7)
print(f"  expand_features(10,7) → shape {psi_batch.shape}")
assert psi_batch.shape == (10, 36), f"Expected (10,36), got {psi_batch.shape}"
print("  OK")

print("\n=== Checking evaluate_sota.py imports ===")
from evaluate_sota import (compute_2d_mgae, compute_3d_mgae,
                            compute_rmse, euclidean_cm, LAMBDA_REG)
print(f"  LAMBDA_REG = {LAMBDA_REG}")
assert abs(LAMBDA_REG - 0.473) < 1e-9

pred = np.array([[0.5, 0.5], [0.3, 0.4]])
gt   = np.array([[0.5, 0.5], [0.6, 0.7]])
m2d  = compute_2d_mgae(pred, gt)
print(f"  compute_2d_mgae sample = {m2d:.4f}°")

pred_cm = np.array([[1.0, 2.0], [-1.0, 0.5]])
gt_cm   = np.array([[0.5, 1.0], [-0.5, 1.0]])
dist_f  = np.array([0.08, 0.10])
m3d     = compute_3d_mgae(pred_cm, gt_cm, dist_f)
print(f"  compute_3d_mgae sample = {m3d:.4f}°")
print("  OK")

print("\n=== All checks passed ===")
