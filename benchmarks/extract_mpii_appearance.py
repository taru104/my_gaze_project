"""MPII画像から「目領域アピアランス特徴」を抽出(16Dと同サンプルに揃える)。
目頭・目尻の2点で相似変換して目を正規化パッチ化(平行移動/回転/スケール不変)→グレースケール小画像。
「幾何16Dが捨てている画像情報が視線に効くか」をMPII客観で検証するため(exp52)。GPU不要。mainは触らない。

出力: cache/mpii_app.npz  X16(N,16), Xapp(N,2*PW*PH), y(N,2 正規化screen), pid(N)
Usage:
    .venv/Scripts/python.exe benchmarks/extract_mpii_appearance.py --test
    .venv/Scripts/python.exe benchmarks/extract_mpii_appearance.py --limit 400   # 各人400枚(run_in_background)
"""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import cv2
import scipy.io as sio
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
sys.path.insert(0, str(Path(__file__).parent.parent))
from rich16d import rich_16d_from_lms, lms_to_array

ROOT = Path(__file__).parent.parent
MPII = ROOT / "MPIIFaceGaze"
MODEL = ROOT / "face_landmarker.task"
PW, PH = 24, 16                       # 目パッチ解像度
# MediaPipe FaceMesh 目頭/目尻: 右目(33 outer,133 inner) 左目(362 inner,263 outer)
EYES = [(33, 133), (362, 263)]

def make_lm():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)

def screen_size(pid):
    ss = sio.loadmat(str(MPII / pid / "Calibration" / "screenSize.mat"))
    return float(ss["width_pixel"][0][0]), float(ss["height_pixel"][0][0])

def sim_transform(p0, p1, q0, q1):
    """2点対応 (p0->q0, p1->q1) の相似変換(平行移動+回転+等方スケール) 2x3."""
    dp = p1 - p0; dq = q1 - q0
    np_ = np.hypot(*dp) + 1e-9
    s = (np.hypot(*dq)) / np_
    ang = np.arctan2(dq[1], dq[0]) - np.arctan2(dp[1], dp[0])
    c, sn = np.cos(ang) * s, np.sin(ang) * s
    R = np.array([[c, -sn], [sn, c]])
    t = q0 - R @ p0
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], np.float32)

def eye_patch(gray, arr, w, h, oi, ii):
    O = np.array([arr[oi, 0] * w, arr[oi, 1] * h])   # outer corner
    I = np.array([arr[ii, 0] * w, arr[ii, 1] * h])   # inner corner
    # 目頭/目尻を固定位置へ: outer=(PW*0.15,PH*0.5), inner=(PW*0.85,PH*0.5)
    M = sim_transform(O, I, np.array([PW * 0.15, PH * 0.5]), np.array([PW * 0.85, PH * 0.5]))
    patch = cv2.warpAffine(gray, M, (PW, PH), flags=cv2.INTER_AREA)
    patch = patch.astype(np.float32)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)   # 照明正規化
    return patch.ravel()

def main():
    test = "--test" in sys.argv
    limit = 40 if test else None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    lm = make_lm()
    ckpath = ROOT / "cache" / "mpii_app_ck.npz"
    X16, XAP, Y, PID = [], [], [], []
    done = set()
    if not test and ckpath.exists():
        d = np.load(ckpath)
        X16 = list(d["X16"]); XAP = list(d["Xapp"]); Y = list(d["y"]); PID = list(d["pid"].tolist())
        done = set(PID)
        print(f"[Resume] {len(X16)}枚復元, 済:{sorted(done)}", flush=True)
    parts = ["p00"] if test else [f"p{i:02d}" for i in range(15)]
    t0 = time.time()
    for pid in parts:
        if pid in done:
            print(f"{pid}: skip", flush=True); continue
        try: wpx, hpx = screen_size(pid)
        except Exception as e:
            print(f"{pid}: screenSize NG {e}"); continue
        lines = open(MPII / pid / f"{pid}.txt").read().strip().splitlines()
        if limit: lines = lines[:limit]
        ok = 0
        for ln in lines:
            f = ln.split()
            imgp, gx, gy = f[0], float(f[1]), float(f[2])
            img = cv2.imread(str(MPII / pid / imgp))
            if img is None: continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks: continue
            arr = lms_to_array(res.face_landmarks[0])
            feat = rich_16d_from_lms(arr, w, h)
            if feat is None or not np.isfinite(feat).all(): continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                ap = np.concatenate([eye_patch(gray, arr, w, h, oi, ii) for oi, ii in EYES])
            except Exception:
                continue
            if not np.isfinite(ap).all(): continue
            X16.append(feat); XAP.append(ap); Y.append([gx / wpx, gy / hpx]); PID.append(pid); ok += 1
        if not test:
            np.savez_compressed(str(ckpath), X16=np.array(X16, np.float32), Xapp=np.array(XAP, np.float32),
                                y=np.array(Y, np.float32), pid=np.array(PID))
        el = time.time() - t0 + 1e-6
        print(f"{pid}: {ok}/{len(lines)}  (計{len(X16)}, {ok/el:.0f}fps)", flush=True)
    out = ROOT / "cache" / ("mpii_app_test.npz" if test else "mpii_app.npz")
    np.savez_compressed(str(out), X16=np.array(X16, np.float32), Xapp=np.array(XAP, np.float32),
                        y=np.array(Y, np.float32), pid=np.array(PID))
    print(f"[Done] {len(X16)}枚, Xapp次元={2*PW*PH} → {out}", flush=True)

if __name__ == "__main__":
    main()
