"""
実環境精度計測スクリプト。

プロトコル:
  Phase 1: 9点キャリブレーション（main.py と同じ CALIB_POINTS_9）
  Phase 2: 8点評価（キャリブ点と重ならない位置）
           各点 3秒注視 → 最初 1秒を安定待ち → 後半 2秒の gaze 平均を記録

出力:
  - 各評価点の予測位置・真値・誤差
  - 全体 MGAE（角度誤差）、Euclidean error（正規化・cm換算）
  - eval_results_YYYYMMDD_HHMMSS.txt に保存

Usage:
  python eval_accuracy.py                   # カメラ0
  python eval_accuracy.py 1                 # カメラ1
  python eval_accuracy.py 0 52.0 29.0       # カメラ0, 画面52cm×29cm
"""

import csv
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator   import GazeEstimator
from calibration import CALIB_POINTS_9

# ──── 評価点（3×3グリッドからずらした8点、キャリブ点と重ならない）────────────
EVAL_POINTS = np.array([
    [0.25, 0.25], [0.75, 0.25],
    [0.25, 0.75], [0.75, 0.75],
    [0.50, 0.17], [0.50, 0.83],
    [0.17, 0.50], [0.83, 0.50],
], dtype=np.float32)

# ──── UI 定数 ─────────────────────────────────────────────────────────────────
C_BG     = (20,  20,  20 )
C_CALIB  = (0,   255, 120)
C_EVAL   = (0,   180, 255)
C_DONE   = (80,  80,  80 )
C_TEXT   = (220, 220, 220)
C_WARN   = (60,  60,  200)
C_ERR    = (60,  80,  200)

CALIB_TOTAL   = 3.0
CALIB_DISCARD = 1.0
EVAL_TOTAL    = 3.0
EVAL_DISCARD  = 1.0


class EvalApp:
    def __init__(self, cam_id: int = 0, win_w: int = 1280, win_h: int = 720,
                 screen_cm_w: Optional[float] = None,
                 screen_cm_h: Optional[float] = None):
        self.win_w = win_w
        self.win_h = win_h
        self.screen_cm_w = screen_cm_w
        self.screen_cm_h = screen_cm_h

        self.estimator = GazeEstimator()
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {cam_id} not found.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._state       = 'calibrating'   # 'calibrating' | 'evaluating' | 'done'
        self._phase_idx   = 0
        self._phase_t     = time.time()

        self._gaze_buf: List[np.ndarray] = []   # 評価フレームのgaze蓄積
        self._eval_results: List[dict]   = []   # 各評価点の結果

        self._fps_t0  = time.time()
        self._fps_cnt = 0
        self._fps     = 0.0

    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow('Gaze Accuracy Evaluation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gaze Accuracy Evaluation', self.win_w, self.win_h)

        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        gaze: Optional[np.ndarray] = None

        while self._state != 'done':
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            new_gaze, debug = self.estimator.process_frame(frame)
            if new_gaze is not None:
                gaze = new_gaze

            self._update_fps()
            canvas[:] = C_BG
            self._draw_camera_preview(canvas, frame)

            if self._state == 'calibrating':
                self._step_calibration(new_gaze, canvas)
            elif self._state == 'evaluating':
                self._step_evaluation(new_gaze, gaze, canvas)

            self._draw_fps(canvas)
            cv2.imshow('Gaze Accuracy Evaluation', canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("[ABORT]")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.estimator.close()

        if self._eval_results:
            self._report()

    # ──── キャリブレーションフェーズ ────────────────────────────────────────

    def _step_calibration(self, new_gaze, canvas):
        w, h  = self.win_w, self.win_h
        n_pts = len(CALIB_POINTS_9)
        idx   = self._phase_idx
        elapsed = time.time() - self._phase_t

        # 完了判定
        if idx >= n_pts:
            try:
                self.estimator.finalize_calibration()
                print(f"[CAL] done  {self.estimator.calibration.n_samples} samples  "
                      f"MGAE={self.estimator.calibration.train_mgae:.2f}deg")
            except ValueError as e:
                print(f"[ERROR] {e}")
            self._state     = 'evaluating'
            self._phase_idx = 0
            self._phase_t   = time.time()
            self._gaze_buf  = []
            return

        # サンプル収集
        if elapsed >= CALIB_DISCARD and self.estimator.face_detected:
            target = CALIB_POINTS_9[idx]
            weight = min((elapsed - CALIB_DISCARD) / (CALIB_TOTAL - CALIB_DISCARD), 1.0)
            self.estimator.collect_calibration(target, weight)

        if elapsed >= CALIB_TOTAL:
            self._phase_idx += 1
            self._phase_t = time.time()
            return

        # 描画
        for i in range(n_pts):
            px = int(CALIB_POINTS_9[i][0] * w)
            py = int(CALIB_POINTS_9[i][1] * h)
            if i < idx:
                cv2.circle(canvas, (px, py), 10, C_DONE, -1)
            elif i == idx:
                r = int(18 + 6 * np.sin(time.time() * 6.0))
                cv2.circle(canvas, (px, py), r, C_CALIB, 2)
                cv2.circle(canvas, (px, py), 6, C_CALIB, -1)
                # プログレスバー
                bw = 120
                prog = min(elapsed / CALIB_TOTAL, 1.0)
                cv2.rectangle(canvas, (px-bw//2, py+28), (px+bw//2, py+40), (40,40,40), -1)
                cv2.rectangle(canvas, (px-bw//2, py+28),
                              (px-bw//2+int(bw*prog), py+40), C_CALIB, -1)
            else:
                cv2.circle(canvas, (px, py), 10, (60, 120, 60), 1)

        status = "Stabilizing..." if elapsed < CALIB_DISCARD else "Look at the dot"
        cv2.putText(canvas, f"CALIBRATION  {idx+1}/{n_pts}: {status}",
                    (w//2-260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT, 2)

    # ──── 評価フェーズ ──────────────────────────────────────────────────────

    def _step_evaluation(self, new_gaze, gaze, canvas):
        w, h    = self.win_w, self.win_h
        n_pts   = len(EVAL_POINTS)
        idx     = self._phase_idx
        elapsed = time.time() - self._phase_t

        if idx >= n_pts:
            self._state = 'done'
            return

        target = EVAL_POINTS[idx]
        tx = int(target[0] * w)
        ty = int(target[1] * h)

        # gaze 収集（安定後）
        if elapsed >= EVAL_DISCARD and new_gaze is not None:
            self._gaze_buf.append(new_gaze.copy())

        # 1点完了
        if elapsed >= EVAL_TOTAL:
            if len(self._gaze_buf) >= 5:
                buf = np.array(self._gaze_buf)
                pred_mean = buf.mean(axis=0)
                pred_med  = np.median(buf, axis=0)
                err_vec   = pred_med - target
                euc_norm  = float(np.linalg.norm(err_vec))

                # MGAE (angular, 画面座標を3Dレイに変換)
                def to3d(p):
                    return np.array([p[0]-0.5, p[1]-0.5, 1.0])
                v1 = to3d(pred_med)
                v2 = to3d(target)
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
                mgae = float(np.degrees(np.arccos(np.clip(cos_sim, -1+1e-7, 1-1e-7))))

                self._eval_results.append({
                    'target':    target.copy(),
                    'pred_mean': pred_mean,
                    'pred_med':  pred_med,
                    'err_vec':   err_vec,
                    'euc_norm':  euc_norm,
                    'mgae':      mgae,
                    'n_frames':  len(self._gaze_buf),
                    'std':       buf.std(axis=0),
                })
                print(f"[EVAL] pt{idx+1}/{n_pts}  target=({target[0]:.2f},{target[1]:.2f})  "
                      f"pred=({pred_med[0]:.3f},{pred_med[1]:.3f})  "
                      f"euc={euc_norm:.4f}  MGAE={mgae:.2f}deg")
            else:
                print(f"[EVAL] pt{idx+1} skipped (only {len(self._gaze_buf)} frames)")

            self._phase_idx += 1
            self._phase_t   = time.time()
            self._gaze_buf  = []
            return

        # 描画: 完了済み点
        for i, res in enumerate(self._eval_results):
            px = int(res['target'][0] * w)
            py = int(res['target'][1] * h)
            cv2.circle(canvas, (px, py), 8, C_DONE, -1)
            # 誤差矢印
            ex = int(res['pred_med'][0] * w)
            ey = int(res['pred_med'][1] * h)
            cv2.arrowedLine(canvas, (px, py), (ex, ey), (100, 80, 200), 2, tipLength=0.3)

        # 描画: 現在の評価点
        r = int(20 + 8 * np.sin(time.time() * 5.0))
        cv2.circle(canvas, (tx, ty), r, C_EVAL, 3)
        cv2.circle(canvas, (tx, ty), 6, C_EVAL, -1)

        bw = 140
        prog = min(elapsed / EVAL_TOTAL, 1.0)
        cv2.rectangle(canvas, (tx-bw//2, ty+30), (tx+bw//2, ty+42), (40,40,40), -1)
        cv2.rectangle(canvas, (tx-bw//2, ty+30),
                      (tx-bw//2+int(bw*prog), ty+42), C_EVAL, -1)

        # 現在の gaze を表示
        if gaze is not None:
            gx = int(gaze[0] * w)
            gy = int(gaze[1] * h)
            cv2.circle(canvas, (gx, gy), 10, (255, 140, 0), -1)

        status = "Stabilizing..." if elapsed < EVAL_DISCARD else f"Look at the dot  ({len(self._gaze_buf)} frames)"
        cv2.putText(canvas, f"EVALUATION  {idx+1}/{n_pts}: {status}",
                    (w//2-270, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT, 2)

    # ──── 結果レポート ───────────────────────────────────────────────────────

    def _report(self):
        results = self._eval_results
        if not results:
            print("[WARN] No results.")
            return

        eucs  = np.array([r['euc_norm'] for r in results])
        mgaes = np.array([r['mgae']     for r in results])

        lines = []
        lines.append("=" * 70)
        lines.append("  Gaze Accuracy Evaluation -- Real Environment")
        lines.append(f"  Date    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Points  : {len(results)} eval  (calibrated on {len(CALIB_POINTS_9)} pts)")
        if self.screen_cm_w and self.screen_cm_h:
            lines.append(f"  Screen  : {self.screen_cm_w:.1f} x {self.screen_cm_h:.1f} cm")
        lines.append("=" * 70)
        lines.append("")
        lines.append("  Per-point results:")
        lines.append(f"  {'#':>2}  {'target':>12}  {'pred(med)':>14}  {'err_x':>7}  {'err_y':>7}  {'euc':>7}  {'MGAE':>7}  {'n':>4}")
        lines.append("  " + "-" * 68)

        for i, r in enumerate(results):
            t  = r['target']
            p  = r['pred_med']
            ev = r['err_vec']
            euc_str  = f"{r['euc_norm']:.4f}"
            mgae_str = f"{r['mgae']:.2f}"

            # cm 換算
            if self.screen_cm_w and self.screen_cm_h:
                err_cm = np.sqrt((ev[0]*self.screen_cm_w)**2 + (ev[1]*self.screen_cm_h)**2)
                euc_str = f"{r['euc_norm']:.4f} ({err_cm:.2f}cm)"

            lines.append(f"  {i+1:>2}  ({t[0]:.2f},{t[1]:.2f})  "
                         f"({p[0]:.3f},{p[1]:.3f})  "
                         f"{ev[0]:>+7.4f}  {ev[1]:>+7.4f}  "
                         f"{euc_str:>14}  {mgae_str:>7}  {r['n_frames']:>4}")

        lines.append("")
        lines.append("  Summary (normalized [0,1] coordinates):")
        lines.append(f"    Euc error   : mean={eucs.mean():.4f}  med={np.median(eucs):.4f}  std={eucs.std():.4f}  max={eucs.max():.4f}")
        lines.append(f"    MGAE        : mean={mgaes.mean():.2f}deg  med={np.median(mgaes):.2f}deg")

        if self.screen_cm_w and self.screen_cm_h:
            eucs_cm = np.array([
                np.sqrt((r['err_vec'][0]*self.screen_cm_w)**2 +
                        (r['err_vec'][1]*self.screen_cm_h)**2)
                for r in results
            ])
            lines.append(f"    Euc error(cm): mean={eucs_cm.mean():.2f}  med={np.median(eucs_cm):.2f}  max={eucs_cm.max():.2f}")

        # X/Y 個別バイアス
        err_x = np.array([r['err_vec'][0] for r in results])
        err_y = np.array([r['err_vec'][1] for r in results])
        lines.append(f"    X bias      : {err_x.mean():+.4f}  (pos=right, neg=left)")
        lines.append(f"    Y bias      : {err_y.mean():+.4f}  (pos=down,  neg=up)")
        lines.append(f"    X std       : {err_x.std():.4f}")
        lines.append(f"    Y std       : {err_y.std():.4f}")
        lines.append("=" * 70)

        report = "\n".join(lines)
        print("\n" + report)

        # ファイル保存
        out_path = Path(__file__).parent.parent / 'results' / f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        out_path.write_text(report, encoding='utf-8')
        print(f"\n[SAVED] {out_path}")

    def _draw_camera_preview(self, canvas, frame):
        ph, pw = 120, 160
        small  = cv2.resize(frame, (pw, ph))
        canvas[8:8+ph, 8:8+pw] = small
        cv2.rectangle(canvas, (8, 8), (8+pw, 8+ph), (60,60,60), 1)

    def _draw_fps(self, canvas):
        cv2.putText(canvas, f"FPS:{self._fps:.0f}",
                    (self.win_w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1)

    def _update_fps(self):
        self._fps_cnt += 1
        if time.time() - self._fps_t0 >= 1.0:
            self._fps     = self._fps_cnt / (time.time() - self._fps_t0)
            self._fps_cnt = 0
            self._fps_t0  = time.time()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cam_id      = int(sys.argv[1])   if len(sys.argv) > 1 else 0
    screen_cm_w = float(sys.argv[2]) if len(sys.argv) > 2 else None
    screen_cm_h = float(sys.argv[3]) if len(sys.argv) > 3 else None

    print("=" * 50)
    print("  Gaze Accuracy Evaluation")
    print(f"  Camera : {cam_id}")
    if screen_cm_w:
        print(f"  Screen : {screen_cm_w} x {screen_cm_h} cm")
    else:
        print("  Screen : unknown (normalized coords only)")
        print("  Tip    : python eval_accuracy.py 0 <width_cm> <height_cm>")
    print("  Phase 1: 9-point calibration")
    print("  Phase 2: 8-point evaluation (3s each)")
    print("  Q/ESC  : abort")
    print("=" * 50)

    try:
        app = EvalApp(cam_id, screen_cm_w=screen_cm_w, screen_cm_h=screen_cm_h)
        app.run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
