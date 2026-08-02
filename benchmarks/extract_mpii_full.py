"""MPIIFaceGaze を「一度で全部」抽出する決定版キャッシュ。

これまでの mpii_7d / mpii_16d / mpii_app* は特徴だけを持ち、3D注釈を捨てていた。
そのため (a) cm 換算が固定30.0cm の粗い近似 (b) 角度誤差(度)が出せず文献と比較不能 だった。
本スクリプトは 1 フレームにつき以下を全部保存し、以降の実験を全てこのキャッシュ1本に統一する。

  X16    : rich16d と厳密同一の16D幾何特徴
  patch  : appearance.eye_patch と同一手順の 48x32 CLAHE 目パッチ(2眼) を uint8 で保存
           (z-score正規化は決定的なので復元可。float32保存比 1/4 の容量)
  y_norm : 画面正規化 [0,1]  / y_px : 画面ピクセル
  fc, gt : 3D顔中心・3D注視点(カメラ座標, mm)   → 視線ベクトル gt-fc = 角度誤差の正解
  hr, ht : 頭部姿勢 rvec/tvec
  scr    : そのフレームの人の画面 [width_mm, height_mm, width_px, height_px]
  pid / lidx : 被験者ID・注釈行番号(再現・突合用)

Usage:
    .venv/Scripts/python.exe benchmarks/extract_mpii_full.py --test     # p00 の40枚で検証
    .venv/Scripts/python.exe benchmarks/extract_mpii_full.py            # 全15人(要 run_in_background)
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import cv2
import scipy.io as sio
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from rich16d import rich_16d_from_lms, lms_to_array
from appearance import _sim_transform, _CLAHE, _EYES, PW, PH

MPII  = ROOT / "MPIIFaceGaze"
MODEL = ROOT / "face_landmarker.task"


def make_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def patch_u8(frame_bgr, lms):
    """appearance.eye_patch と同一の幾何・同一CLAHE。z-score前の uint8 パッチ(2眼連結)を返す。"""
    h, w = frame_bgr.shape[:2]
    gray = _CLAHE.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    out = []
    for oi, ii in _EYES:
        O = np.array([lms[oi].x * w, lms[oi].y * h])
        I = np.array([lms[ii].x * w, lms[ii].y * h])
        M = _sim_transform(O, I, np.array([PW * 0.15, PH * 0.5]),
                           np.array([PW * 0.85, PH * 0.5]))
        out.append(cv2.warpAffine(gray, M, (PW, PH), flags=cv2.INTER_AREA).ravel())
    return np.concatenate(out).astype(np.uint8)


def calib(pid):
    ss = sio.loadmat(str(MPII / pid / "Calibration" / "screenSize.mat"))
    mp_ = sio.loadmat(str(MPII / pid / "Calibration" / "monitorPose.mat"))
    g = lambda k: float(np.ravel(ss[k])[0])
    R = cv2.Rodrigues(np.ravel(mp_["rvects"] if "rvects" in mp_ else mp_["rvecs"]).astype(np.float64))[0]
    T = np.ravel(mp_["tvecs"]).astype(np.float64)
    return (g("width_mm"), g("height_mm"), g("width_px" if "width_px" in ss else "width_pixel"),
            g("height_pixel")), R, T


def px_to_cam(px, py, scr, R, T):
    """画面ピクセル → カメラ座標3D(mm)。monitorPose による平面変換。"""
    wmm, hmm, wpx, hpx = scr
    v = np.array([px / wpx * wmm, py / hpx * hmm, 0.0])
    return R @ v + T


def main():
    test = "--test" in sys.argv
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    out = ROOT / "cache" / f"mpii_full{tag}.npz"
    ck = ROOT / "cache" / f"mpii_full{tag}_ck.npz"

    keys = ["X16", "patch", "y_norm", "y_px", "fc", "gt", "hr", "ht", "scr", "pid", "lidx"]
    acc = {k: [] for k in keys}
    done = set()
    if not test and ck.exists():
        d = np.load(ck)
        for k in keys: acc[k] = list(d[k])
        done = set(np.asarray(d["pid"]).tolist())
        print(f"[Resume] {len(acc['X16'])}フレーム, 済み={sorted(done)}", flush=True)

    lm = make_landmarker()
    people = ["p00"] if test else [f"p{i:02d}" for i in range(15)]
    t0 = time.time()
    for pid in people:
        if pid in done:
            print(f"{pid}: skip(済)", flush=True); continue
        scr, R, T = calib(pid)
        lines = open(MPII / pid / f"{pid}.txt").read().strip().splitlines()
        if test: lines = lines[:40]
        elif limit: lines = lines[:limit]
        ok = 0
        chk = []           # px→3D 変換の健全性チェック(gt との差, mm)
        for li, ln in enumerate(lines):
            f = ln.split()
            img = MPII / pid / f[0]
            frame = cv2.imread(str(img))
            if frame is None: continue
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not res.face_landmarks: continue
            lms = res.face_landmarks[0]
            try:
                x16 = rich_16d_from_lms(lms_to_array(lms), frame.shape[1], frame.shape[0])
                pu8 = patch_u8(frame, lms)
            except Exception as e:
                if ok == 0 and li < 3: print(f"  [debug] {type(e).__name__}: {e}", flush=True)
                continue
            if x16 is None or not np.isfinite(x16).all(): continue
            gx, gy = float(f[1]), float(f[2])
            fc = np.array([float(v) for v in f[21:24]])
            gt = np.array([float(v) for v in f[24:27]])
            if len(chk) < 50:
                chk.append(np.linalg.norm(px_to_cam(gx, gy, scr, R, T) - gt))
            acc["X16"].append(np.asarray(x16, np.float32))
            acc["patch"].append(pu8)
            acc["y_norm"].append(np.array([gx / scr[2], gy / scr[3]], np.float32))
            acc["y_px"].append(np.array([gx, gy], np.float32))
            acc["fc"].append(fc.astype(np.float32)); acc["gt"].append(gt.astype(np.float32))
            acc["hr"].append(np.array([float(v) for v in f[15:18]], np.float32))
            acc["ht"].append(np.array([float(v) for v in f[18:21]], np.float32))
            acc["scr"].append(np.array(scr, np.float32))
            acc["pid"].append(pid); acc["lidx"].append(li)
            ok += 1
        d_mm = float(np.median(chk)) if chk else float("nan")
        print(f"{pid}: {ok}/{len(lines)}枚OK  px→3D誤差(中央値)={d_mm:.1f}mm  "
              f"累計{len(acc['X16'])}  {time.time()-t0:.0f}s", flush=True)
        if not test:
            np.savez_compressed(ck, **{k: np.asarray(acc[k]) for k in keys})
    if not test:
        np.savez_compressed(out, **{k: np.asarray(acc[k]) for k in keys})
        print(f"[DONE] {out} n={len(acc['X16'])}", flush=True)
    else:
        print("[TEST] 保存なし。上の px→3D誤差が数mm以内なら monitorPose の解釈は正しい。")


if __name__ == "__main__":
    main()
