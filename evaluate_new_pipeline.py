"""
solvePnP + AffineCalibration パイプライン -- GazeCapture ベンチマーク (v2)

変更点 (v1 → v2):
  - 特徴抽出: solvePnP 球体深度 → 虹彩直径ベース動的深度
    Z_iris = f * 11.7 / iris_diam_px  だが f がキャンセルされるため
    X_mm = 11.7 * (iris_x - cx) / iris_diam_px  (焦点距離仮定不要)
  - solvePnP を評価スクリプトから除外 (速度改善)
  - AffineCalibration はそのまま使用

比較対象 (prior results):
  EyeTrax(adaptive)      MGAE_3D=18.06 deg,  Euc(cm)=16.36 (median)
  Webcam3DTracker(PCA)   MGAE_3D= 9.94 deg,  Euc(cm)= 7.59 (median)
  solvePnP+Affine v1     MGAE_3D=14.61 deg,  Euc(cm)= 8.22 (median)

Usage:
    .\.venv\Scripts\python.exe evaluate_new_pipeline.py
"""

import os, sys, json, time
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, str(Path(__file__).parent))
from calibration import AffineCalibration

PROJECT_DIR  = Path(__file__).parent
ARCHIVE_DIR  = PROJECT_DIR / 'archive'
MODEL_PATH   = PROJECT_DIR / 'face_landmarker.task'
RESULTS_PATH = PROJECT_DIR / 'new_pipeline_results.txt'
FIXED_Z_FACE_CM = 50.0

IRIS_DIAMETER_MM = 11.7   # 成人平均虹彩直径 (mm)

# 虹彩エッジインデックス
_L_EDGE4 = [469, 470, 471, 472]
_R_EDGE4 = [474, 475, 476, 477]
_L_ALL   = [468, 469, 470, 471, 472]
_R_ALL   = [473, 474, 475, 476, 477]


# ──── Data loading ─────────────────────────────────────────────────────────────

def load_test_subjects(archive_dir: Path) -> dict:
    subjects = sorted(os.listdir(archive_dir))
    records = {}
    skipped = 0
    for subj in subjects:
        sdir = archive_dir / subj / subj
        if not sdir.is_dir():
            continue
        try:
            with open(sdir / 'info.json') as f:
                if json.load(f).get('Dataset', '') != 'test':
                    continue
            with open(sdir / 'frames.json') as f:
                frames = json.load(f)
            with open(sdir / 'dotInfo.json') as f:
                dot = json.load(f)
            with open(sdir / 'screen.json') as f:
                screen = json.load(f)
            with open(sdir / 'faceGrid.json') as f:
                grid = json.load(f)
        except Exception:
            skipped += 1
            continue
        recs = []
        for i in range(len(frames)):
            if not grid['IsValid'][i]:
                continue
            p = sdir / 'frames' / frames[i]
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
            records[int(subj)] = recs
    n_total = sum(len(v) for v in records.values())
    print(f"[Data] {n_total} frames, {len(records)} test subjects  "
          f"({skipped} subjects skipped)")
    return records


# ──── Feature extraction (iris depth, solvePnP-free) ──────────────────────────

def _iris_diam_px(lms, edge_indices, img_w, img_h) -> float:
    pts = np.array([[lms[i].x * img_w, lms[i].y * img_h] for i in edge_indices])
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = max(best, float(np.linalg.norm(pts[i] - pts[j])))
    return best


def extract_gaze_2d(img_path: str, landmarker) -> "np.ndarray | None":
    """
    虹彩直径から gaze_2d = [X_mm, Y_mm] を抽出。solvePnP 不要。
    X_mm = IRIS_DIAMETER_MM * (iris_x - cx) / iris_diam_px  (f がキャンセル)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    if len(lms) < 478:
        return None

    cx, cy = w / 2.0, h / 2.0

    def iris_center(indices):
        pts = np.array([[lms[i].x * w, lms[i].y * h] for i in indices])
        return pts.mean(axis=0)

    L_c = iris_center(_L_ALL)
    R_c = iris_center(_R_ALL)
    L_d = _iris_diam_px(lms, _L_EDGE4, w, h)
    R_d = _iris_diam_px(lms, _R_EDGE4, w, h)

    if L_d < 2.0 or R_d < 2.0:
        return None

    X_mm = (IRIS_DIAMETER_MM * (L_c[0] - cx) / L_d +
            IRIS_DIAMETER_MM * (R_c[0] - cx) / R_d) / 2.0
    Y_mm = (IRIS_DIAMETER_MM * (L_c[1] - cy) / L_d +
            IRIS_DIAMETER_MM * (R_c[1] - cy) / R_d) / 2.0

    return np.array([X_mm, Y_mm], dtype=np.float64)


# ──── Metrics ──────────────────────────────────────────────────────────────────

def mgae_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    def v3(p):
        return np.column_stack([p[:, 0]-0.5, p[:, 1]-0.5, np.ones(len(p))])
    v1, v2 = v3(pred), v3(gt)
    cos = (np.sum((v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)) *
                  (v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)), axis=-1)
           .clip(-1+1e-7, 1-1e-7))
    return float(np.mean(np.degrees(np.arccos(cos))))


def mgae_3d_fixed(pred_cm: np.ndarray, gt_cm: np.ndarray,
                  z: float = FIXED_Z_FACE_CM) -> float:
    g1 = np.column_stack([pred_cm[:,0], pred_cm[:,1], np.full(len(pred_cm), -z)])
    g2 = np.column_stack([gt_cm[:,0],   gt_cm[:,1],   np.full(len(gt_cm),   -z)])
    cos = (np.sum((g1 / (np.linalg.norm(g1, axis=-1, keepdims=True) + 1e-8)) *
                  (g2 / (np.linalg.norm(g2, axis=-1, keepdims=True) + 1e-8)), axis=-1)
           .clip(-1+1e-7, 1-1e-7))
    return float(np.mean(np.degrees(np.arccos(cos))))


def euc_cm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.median(np.sqrt(np.sum((pred - gt) ** 2, axis=-1))))


# ──── Per-subject evaluation ────────────────────────────────────────────────────

def evaluate_subject(records: list, landmarker) -> "dict | None":
    n     = len(records)
    n_cal = max(1, int(n * 0.05))

    gazes = []
    gt_xn, gt_yn, gt_xc, gt_yc = [], [], [], []
    for img_path, xn, yn, xc, yc in records:
        gazes.append(extract_gaze_2d(img_path, landmarker))
        gt_xn.append(xn); gt_yn.append(yn)
        gt_xc.append(xc); gt_yc.append(yc)

    gt_xn = np.array(gt_xn); gt_yn = np.array(gt_yn)
    gt_xc = np.array(gt_xc); gt_yc = np.array(gt_yc)

    # ── Calibration: AffineCalibration on first 5% ───────────────────────────
    af = AffineCalibration()
    cal_pred_xn, cal_pred_yn = [], []
    cal_gt_xc_list, cal_gt_yc_list = [], []

    for i in range(n_cal):
        g = gazes[i]
        if g is None:
            continue
        af.add(g[0], g[1], float(gt_xn[i]), float(gt_yn[i]), weight=1.0)
        cal_pred_xn.append(g[0]); cal_pred_yn.append(g[1])
        cal_gt_xc_list.append(gt_xc[i]); cal_gt_yc_list.append(gt_yc[i])

    n_cal_ok = len(cal_pred_xn)
    if n_cal_ok < 3:
        return None
    af.fit()

    # norm → cm の線形スケールを calibration データで学習
    cal_px = np.array(cal_pred_xn); cal_py = np.array(cal_pred_yn)
    cal_gxc = np.array(cal_gt_xc_list); cal_gyc = np.array(cal_gt_yc_list)

    cal_pred_norm = np.array([af.predict(x, y) for x, y in zip(cal_px, cal_py)])
    cpnx, cpny = cal_pred_norm[:, 0], cal_pred_norm[:, 1]

    if len(cpnx) >= 2 and np.std(cpnx) > 1e-6:
        ax, bx = np.polyfit(cpnx, cal_gxc, 1)
    else:
        ax, bx = 0.0, float(np.mean(cal_gxc)) if len(cal_gxc) else 0.0

    if len(cpny) >= 2 and np.std(cpny) > 1e-6:
        ay, by = np.polyfit(cpny, cal_gyc, 1)
    else:
        ay, by = 0.0, float(np.mean(cal_gyc)) if len(cal_gyc) else 0.0

    # polyfit スロープの異常値クランプ (外挿暴走防止)
    SLOPE_CLIP = 200.0
    ax = float(np.clip(ax, -SLOPE_CLIP, SLOPE_CLIP))
    ay = float(np.clip(ay, -SLOPE_CLIP, SLOPE_CLIP))

    # ── Evaluation ───────────────────────────────────────────────────────────
    pred_norm_list, pred_cm_list = [], []
    gt_norm_list,   gt_cm_list   = [], []

    for i in range(n_cal, n):
        g = gazes[i]
        if g is None:
            continue
        pn = af.predict(g[0], g[1])
        pc = np.array([ax * pn[0] + bx, ay * pn[1] + by])

        pred_norm_list.append(pn)
        pred_cm_list.append(pc)
        gt_norm_list.append([gt_xn[i], gt_yn[i]])
        gt_cm_list.append([gt_xc[i], gt_yc[i]])

    n_ev_ok = len(pred_norm_list)
    if n_ev_ok < 5:
        return None

    pred_norm = np.array(pred_norm_list)
    pred_cm   = np.array(pred_cm_list)
    gt_norm   = np.array(gt_norm_list)
    gt_cm_arr = np.array(gt_cm_list)
    ok_pct    = int(round(100.0 * (n_cal_ok + n_ev_ok) / n))

    return {
        'mgae_2d':  mgae_2d(pred_norm, gt_norm),
        'mgae_3d':  mgae_3d_fixed(pred_cm, gt_cm_arr),
        'euc':      euc_cm(pred_cm, gt_cm_arr),
        'n_total':  n,
        'n_cal':    n_cal,
        'n_cal_ok': n_cal_ok,
        'n_ev_ok':  n_ev_ok,
        'ok_pct':   ok_pct,
    }


# ──── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 72)
    print("  Iris Depth + AffineCalibration (v2) -- GazeCapture Benchmark")
    print("=" * 72)

    records_by_subj = load_test_subjects(ARCHIVE_DIR)
    if not records_by_subj:
        print("[ERROR] No test subjects found.")
        sys.exit(1)

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    print(f"\n  {'Subj':>5}  {'n':>5}  {'cal':>4}  {'ok%':>5}  "
          f"{'MGAE_2D':>8}  {'MGAE_3D':>8}  {'Euc(cm)':>8}")
    print("  " + "-" * 58)

    all_results = []
    skipped = []

    for sid in sorted(records_by_subj):
        recs  = records_by_subj[sid]
        n     = len(recs)
        n_cal = max(1, int(n * 0.05))
        res   = evaluate_subject(recs, landmarker)

        if res is None:
            skipped.append(sid)
            print(f"  {sid:5d}  {n:5d}  {n_cal:4d}   SKIP")
            sys.stdout.flush()
            continue

        all_results.append(res)
        print(f"  {sid:5d}  {res['n_total']:5d}  {res['n_cal']:4d}  "
              f"{res['ok_pct']:>4d}%  "
              f"{res['mgae_2d']:>8.2f}  {res['mgae_3d']:>8.2f}  {res['euc']:>8.3f}")
        sys.stdout.flush()

    landmarker.close()
    elapsed = time.time() - t0

    if not all_results:
        print("[ERROR] No valid subjects evaluated.")
        sys.exit(1)

    mgae2d_vals = [r['mgae_2d'] for r in all_results]
    mgae3d_vals = [r['mgae_3d'] for r in all_results]
    euc_vals    = [r['euc']     for r in all_results]

    sep  = "=" * 72
    hsep = "-" * 72
    lines = [
        sep,
        "  Iris Depth + AffineCalibration vs Previous Pipelines -- GazeCapture",
        f"  Date    : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Runtime : {elapsed/60:.1f} min",
        sep,
        "",
        "  Evaluation Protocol",
        "    Dataset     : GazeCapture test split",
        f"    Subjects    : {len(all_results)} evaluated  ({len(skipped)} skipped)",
        "    Calibration : First 5% frames -- AffineCalibration (2x3 lstsq)",
        "    Evaluation  : Remaining 95% frames",
        f"    3D MGAE     : Fixed Z_face={FIXED_Z_FACE_CM}cm; norm->cm via per-subject linear fit",
        "    Iris Depth  : X_mm = 11.7*(iris_x-cx)/iris_diam_px  (f cancels)",
        "",
        "  " + hsep,
        f"  {'Method':<42} {'MGAE_2D':>8}  {'MGAE_3D':>8}  {'Euc(cm)':>8}",
        f"  {'':42} {'mean':>8}  {'mean':>8}  {'median':>8}",
        "  " + hsep,
        f"  {'EyeTrax(alpha=1.0)':<42} {'54.51':>8}  {'26.83':>8}  {'24.66':>8}",
        f"  {'EyeTrax(alpha=adaptive)':<42} {'44.15':>8}  {'18.06':>8}  {'16.36':>8}",
        f"  {'Custom(7D+poly36,adaptive)':<42} {'45.54':>8}  {'22.29':>8}  {'21.30':>8}",
        f"  {'Webcam3DTracker(PCA sphere)':<42} {'44.55':>8}  {' 9.94':>8}  {' 7.589':>8}",
        f"  {'solvePnP+AffineCalib(v1)':<42} {'44.21':>8}  {'14.61':>8}  {' 8.218':>8}",
        f"  {'IrisDepth+AffineCalib(v2,NEW)':<42} {np.mean(mgae2d_vals):>8.2f}  "
        f"{np.mean(mgae3d_vals):>8.2f}  {np.median(euc_vals):>8.3f}",
        "  " + hsep,
        "",
        "  Detailed Statistics -- IrisDepth + AffineCalibration",
        f"    Subjects    : {len(all_results)}",
        f"    MGAE_2D mean: {np.mean(mgae2d_vals):.3f} +/- {np.std(mgae2d_vals):.3f} deg  "
        f"(med {np.median(mgae2d_vals):.3f})",
        f"    MGAE_3D mean: {np.mean(mgae3d_vals):.3f} +/- {np.std(mgae3d_vals):.3f} deg  "
        f"(med {np.median(mgae3d_vals):.3f})",
        f"    Euc(cm) mean: {np.mean(euc_vals):.3f}  med: {np.median(euc_vals):.3f}  "
        f"max: {np.max(euc_vals):.3f}",
        sep,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_PATH.write_text(text, encoding='utf-8')
    print(f"\n[Done] Results saved -> {RESULTS_PATH}")


if __name__ == '__main__':
    main()
