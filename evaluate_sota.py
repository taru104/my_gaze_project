"""
SOTA評価: 7D解剖学的特徴量 + 被験者ごとの5%キャリブレーション + 3D MGAE。

手法 (Case A: few-shot cross-subject adaptation):
  テストセットの各被験者について:
    - 最初の5%フレーム → RidgeCalibration (poly36, λ=0.473)
    - 残り95%フレーム → 予測・評価
  全被験者の平均MGAE/RMSEを算出。

3D MGAE:
  - 予測cm座標からピンホールカメラモデルで3D視線ベクトルを生成
  - GT cm座標 (XCam, YCam) も同様に3D化
  - arccos(dot(g_pred, g_gt)) の平均

Usage:
    python evaluate_sota.py
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

PROJECT_DIR  = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from features            import extract_7d_from_image
from calibration         import RidgeCalibration
from gazeCapture_dataset import GazeCaptureRawIndex, SPLIT_CODE

ARCHIVE_DIR  = PROJECT_DIR / 'archive'
MODEL_PATH   = PROJECT_DIR / 'face_landmarker.task'
CACHE_PATH   = PROJECT_DIR / 'sota_7d_cache.npz'
RESULTS_PATH = PROJECT_DIR / 'sota_results.txt'

CHECKPOINT_PATH  = PROJECT_DIR / 'sota_7d_checkpoint.npz'
CHECKPOINT_EVERY = 3_000

LAMBDA_REG = 0.473   # SOTAハイパーパラメータ最適化値


# ─── 評価指標 ────────────────────────────────────────────────────────────────

def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def euclidean_cm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=-1))))


def compute_2d_mgae(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    """
    2D近似MGAE: 正規化座標 (x-0.5, y-0.5, 1.0) を3Dベクトルとみなす。
    CalibrationPipeline._compute_mgae と同一ロジック。
    """
    def to3d(p):
        return np.column_stack([p[:, 0] - 0.5, p[:, 1] - 0.5,
                                np.ones(len(p))])
    v1, v2 = to3d(pred_norm), to3d(gt_norm)
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
    cos_sim = np.sum((v1 / n1) * (v2 / n2), axis=-1).clip(-1 + 1e-7, 1 - 1e-7)
    return float(np.mean(np.degrees(np.arccos(cos_sim))))


def compute_3d_mgae(
    pred_cm:      np.ndarray,   # (N, 2) predicted XCam/YCam in cm
    gt_cm:        np.ndarray,   # (N, 2) GT XCam/YCam in cm
    dist_feature: np.ndarray,   # (N,)  pupil dist / img_width (7D feature[6])
    ipd_cm: float = 6.3,
) -> float:
    """
    3D MGAE計算（GazeCaptureの物理座標系）。

    モデル:
      OE = (0, 0, Z_face)  where Z_face = ipd_cm / dist  (ピンホール近似 f=img_w)
      P_target = (XCam, YCam, 0) [cm, カメラ中心からの画面上の座標]
      g = normalize(P_target - OE) = normalize((XCam, YCam, -Z_face))
      MGAE = mean arccos(dot(g_pred, g_gt))
    """
    Z_face = ipd_cm / (dist_feature + 1e-8)   # (N,) in cm

    def to3d(xy_cm):
        return np.column_stack([xy_cm[:, 0], xy_cm[:, 1], -Z_face])  # (N, 3)

    g_pred = to3d(pred_cm)
    g_gt   = to3d(gt_cm)

    n_pred = np.linalg.norm(g_pred, axis=-1, keepdims=True) + 1e-8
    n_gt   = np.linalg.norm(g_gt,   axis=-1, keepdims=True) + 1e-8
    cos_sim = np.sum((g_pred / n_pred) * (g_gt / n_gt), axis=-1).clip(-1 + 1e-7, 1 - 1e-7)
    return float(np.mean(np.degrees(np.arccos(cos_sim))))


def get_subject_id(img_path: str) -> int:
    return int(Path(img_path).parent.parent.name)


# ─── Phase 1: テスト被験者の7D特徴量抽出 ─────────────────────────────────────

def run_extraction() -> None:
    """テスト被験者の全フレームから7D特徴量を抽出し CACHE_PATH に保存。"""
    index        = GazeCaptureRawIndex(str(ARCHIVE_DIR))
    test_records = index.by_split('test')
    n_total      = len(test_records)
    n_subj       = len(set(get_subject_id(r[0]) for r in test_records))
    print(f"\n[Extract-7D] {n_total} test frames from {n_subj} subjects")

    X_list, yn_list, yc_list, sid_list = [], [], [], []
    start_idx = 0
    if CHECKPOINT_PATH.exists():
        ck       = np.load(str(CHECKPOINT_PATH))
        X_list   = list(ck['X'])
        yn_list  = list(ck['y_norm'])
        yc_list  = list(ck['y_cm'])
        sid_list = list(ck['subj_id'])
        start_idx = int(len(X_list) / 0.93)
        print(f"[Extract-7D] Resume from checkpoint: {len(X_list)} frames. Start ≈ {start_idx}")
    else:
        print(f"[Extract-7D] Starting fresh. Model: {MODEL_PATH}")

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

    for i, (img_path, xn, yn, xc, yc, _sc) in enumerate(
            test_records[start_idx:], start=start_idx):
        if i > start_idx and (i - start_idx) % 500 == 0:
            elapsed = time.time() - t0
            fps     = max((i - start_idx) / elapsed, 0.01)
            eta_min = (n_total - i) / fps / 60.0
            pct     = 100.0 * i / n_total
            print(f"  [{pct:5.1f}%] {i}/{n_total}  ok={n_ok}  fail={n_fail}"
                  f"  {fps:.1f} fps  ETA {eta_min:.1f} min")

        img = cv2.imread(img_path)
        if img is None:
            n_fail += 1
            continue

        feats = extract_7d_from_image(img, landmarker)
        if feats is None:
            n_fail += 1
            continue

        X_list.append(feats)
        yn_list.append([xn, yn])
        yc_list.append([xc, yc])
        sid_list.append(get_subject_id(img_path))
        n_ok += 1

        if n_ok >= next_ck:
            np.savez_compressed(str(CHECKPOINT_PATH),
                X=np.array(X_list,   dtype=np.float32),
                y_norm=np.array(yn_list, dtype=np.float32),
                y_cm=np.array(yc_list,   dtype=np.float32),
                subj_id=np.array(sid_list, dtype=np.int32),
            )
            print(f"  [Checkpoint] {n_ok} frames -> {CHECKPOINT_PATH}")
            next_ck = n_ok + CHECKPOINT_EVERY

    landmarker.close()

    elapsed = time.time() - t0
    print(f"\n[Extract-7D] Done in {elapsed/60:.1f} min")
    print(f"  Frames: total={n_total}  ok={n_ok}  fail={n_fail}"
          f"  ({100*n_fail/max(n_total,1):.1f}% detection failure)")

    np.savez_compressed(str(CACHE_PATH),
        X=np.array(X_list,    dtype=np.float32),
        y_norm=np.array(yn_list,  dtype=np.float32),
        y_cm=np.array(yc_list,    dtype=np.float32),
        subj_id=np.array(sid_list, dtype=np.int32),
    )
    print(f"  Cache saved -> {CACHE_PATH}  ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")


# ─── Phase 2: 被験者ごと5%キャリブ → 95%評価 ────────────────────────────────

def run_sota_evaluation(cache_path: Path) -> dict:
    data     = np.load(str(cache_path))
    X        = data['X']          # (N, 7)
    y_norm   = data['y_norm']     # (N, 2)
    y_cm     = data['y_cm']       # (N, 2)  XCam/YCam [cm from camera center]
    subj_ids = data['subj_id']    # (N,) int32

    unique_sids = np.unique(subj_ids)
    n_subjects  = len(unique_sids)
    print(f"\n[SOTA-Eval] {len(X)} frames across {n_subjects} test subjects")
    print(f"  Method  : first 5% of each subject -> Ridge calibration | remaining 95% -> eval")
    print(f"  Features: 7D anatomical [Lx Ly Rx Ry Pitch Yaw dist] + poly36 expansion")
    print(f"  Lambda  : {LAMBDA_REG}  (Ridge regularization)")
    print()

    all_mgae_2d = []
    all_mgae_3d = []
    all_rmse    = []
    all_euc     = []
    skipped     = 0

    for sid in unique_sids:
        mask   = subj_ids == sid
        X_s    = X[mask]
        yn_s   = y_norm[mask]
        yc_s   = y_cm[mask]
        n      = len(X_s)
        n_cal  = max(5, int(np.ceil(0.05 * n)))
        n_eval = n - n_cal

        if n_eval < 10:
            skipped += 1
            continue

        # 適応的λ: 少サンプルでは正則化を強める
        # 理由: poly36 (36次元) を n_cal サンプルで解くため
        # lambda_cal ≈ 5000/n_cal  → n_cal=37で135, n_cal=129で38.8
        lambda_cal = max(LAMBDA_REG, 5000.0 / n_cal)

        # キャリブレーション (5%): 正規化座標用 + cm座標用
        ridge_n = RidgeCalibration(lambda_reg=lambda_cal)
        ridge_n.fit(X_s[:n_cal], yn_s[:n_cal])

        ridge_c = RidgeCalibration(lambda_reg=lambda_cal)
        ridge_c.fit(X_s[:n_cal], yc_s[:n_cal])

        # 評価 (95%)
        pred_n = ridge_n.predict(X_s[n_cal:])   # normalized coords
        pred_c = ridge_c.predict(X_s[n_cal:])   # cm coords

        mgae_2d = compute_2d_mgae(pred_n, yn_s[n_cal:])
        dist_ev = X_s[n_cal:, 6]                # dist feature for depth estimation
        mgae_3d = compute_3d_mgae(pred_c, yc_s[n_cal:], dist_ev)
        rmse    = compute_rmse(pred_n, yn_s[n_cal:])
        euc     = euclidean_cm(pred_c, yc_s[n_cal:])

        all_mgae_2d.append(mgae_2d)
        all_mgae_3d.append(mgae_3d)
        all_rmse.append(rmse)
        all_euc.append(euc)

        print(f"  Subject {sid:5d}  n={n:5d}  cal={n_cal:3d}  lam={lambda_cal:6.1f}"
              f"  MGAE_2D={mgae_2d:6.2f}deg  MGAE_3D={mgae_3d:6.2f}deg"
              f"  Euc={euc:.3f}cm")

    print(f"\n  ({skipped} subjects skipped - too few frames)")

    results = {
        'n_subjects':        n_subjects - skipped,
        'mgae_2d_mean':      float(np.mean(all_mgae_2d)),
        'mgae_2d_med':       float(np.median(all_mgae_2d)),
        'mgae_2d_std':       float(np.std(all_mgae_2d)),
        'mgae_3d_mean':      float(np.mean(all_mgae_3d)),
        'mgae_3d_med':       float(np.median(all_mgae_3d)),
        'mgae_3d_std':       float(np.std(all_mgae_3d)),
        'rmse_norm':         float(np.mean(all_rmse)),
        'euclidean_cm_mean': float(np.mean(all_euc)),
        'euclidean_cm_med':  float(np.median(all_euc)),
    }
    return results


# ─── Phase 3: 結果保存 ───────────────────────────────────────────────────────

def save_results(results: dict, elapsed: float) -> None:
    sep  = "=" * 64
    hsep = "-" * 64
    lines = [
        sep,
        "SOTA Evaluation: 7D Anatomical Features + Few-Shot Calibration",
        f"Date       : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime    : {elapsed/60:.1f} min",
        sep,
        "",
        "Method",
        "  Features     : 7D [Lx Ly Rx Ry Pitch Yaw dist]",
        "                 solvePnP(6pt) head pose, geometric pupil normalization",
        f"  Calibration  : Ridge (lambda_base={LAMBDA_REG}, adaptive per subject,"
        "  StandardScaler + poly36)",
        "                 First 5% of each test subject's frames",
        "  Evaluation   : Remaining 95% of each test subject's frames",
        "  3D MGAE      : Pinhole model Z_face=6.3/dist_feature, screen at Z=0",
        "  Subjects     : GazeCapture test split",
        "",
        hsep,
        "Results (mean / median across all test subjects)",
        hsep,
        f"  Subjects evaluated : {results['n_subjects']}",
        f"  MGAE_2D mean (deg) : {results['mgae_2d_mean']:.3f}  +/- {results['mgae_2d_std']:.3f}",
        f"  MGAE_2D med  (deg) : {results['mgae_2d_med']:.3f}",
        f"  MGAE_3D mean (deg) : {results['mgae_3d_mean']:.3f}  +/- {results['mgae_3d_std']:.3f}",
        f"  MGAE_3D med  (deg) : {results['mgae_3d_med']:.3f}",
        f"  RMSE  (normalized) : {results['rmse_norm']:.5f}",
        f"  Euclidean mean(cm) : {results['euclidean_cm_mean']:.3f}",
        f"  Euclidean med (cm) : {results['euclidean_cm_med']:.3f}",
        "",
        hsep,
        "Reference",
        "  iTracker CNN (no calibration) : 2.53 cm (phone) / 3.02 cm (tablet)",
        "  486D pipeline (few-shot calib): MGAE=85.29 deg (overfitting, lam=100)",
        "  7D global Ridge (no calib)    : MGAE=16.79 deg, Euc=5.47 cm",
        hsep,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_PATH.write_text(text, encoding='utf-8')
    print(f"\n[Done] Results saved -> {RESULTS_PATH}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    if CACHE_PATH.exists():
        sz = os.path.getsize(CACHE_PATH) / 1e6
        print(f"[Cache] Found sota_7d_cache.npz ({sz:.1f} MB) - skipping extraction.")
    else:
        run_extraction()

    results = run_sota_evaluation(CACHE_PATH)
    save_results(results, time.time() - t0)

    subprocess.run(
        ['powershell', '-Command',
         '[System.Console]::Beep(1000, 300);'
         '[System.Threading.Thread]::Sleep(200);'
         '[System.Console]::Beep(1200, 400)'],
        capture_output=True,
    )


if __name__ == '__main__':
    main()
