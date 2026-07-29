"""MPII目パッチを plain と CLAHE(局所コントラスト正規化) の両方で抽出(同サンプルに揃える)。
照明・人ごとの見た目差にロバスト化→person-indep(他人汎化)改善を狙う(exp58)。過学習でなく汎化狙い。GPU不要。mainは触らない。
出力: cache/mpii_app2.npz  X16, Xplain, Xclahe, y, pid
Usage: .venv/Scripts/python.exe benchmarks/extract_mpii_appearance2.py --limit 400   (run_in_background)
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
PW, PH = 24, 16
EYES = [(33, 133), (362, 263)]
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def make_lm():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)

def screen_size(pid):
    ss = sio.loadmat(str(MPII / pid / "Calibration" / "screenSize.mat"))
    return float(ss["width_pixel"][0][0]), float(ss["height_pixel"][0][0])

def sim_transform(p0, p1, q0, q1):
    dp = p1 - p0; dq = q1 - q0
    s = (np.hypot(*dq)) / (np.hypot(*dp) + 1e-9)
    ang = np.arctan2(dq[1], dq[0]) - np.arctan2(dp[1], dp[0])
    c, sn = np.cos(ang) * s, np.sin(ang) * s
    R = np.array([[c, -sn], [sn, c]]); t = q0 - R @ p0
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], np.float32)

def patch(gray, arr, w, h, oi, ii):
    O = np.array([arr[oi, 0] * w, arr[oi, 1] * h]); I = np.array([arr[ii, 0] * w, arr[ii, 1] * h])
    M = sim_transform(O, I, np.array([PW * 0.15, PH * 0.5]), np.array([PW * 0.85, PH * 0.5]))
    p = cv2.warpAffine(gray, M, (PW, PH), flags=cv2.INTER_AREA).astype(np.float32)
    return ((p - p.mean()) / (p.std() + 1e-6)).ravel()

def main():
    limit = None
    if "--limit" in sys.argv: limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--test" in sys.argv: limit = 40
    lm = make_lm()
    ck = ROOT / "cache" / "mpii_app2_ck.npz"
    X16, XP, XC, Y, PID = [], [], [], [], []
    done = set()
    if ck.exists() and "--test" not in sys.argv:
        d = np.load(ck); X16 = list(d["X16"]); XP = list(d["Xplain"]); XC = list(d["Xclahe"])
        Y = list(d["y"]); PID = list(d["pid"].tolist()); done = set(PID)
        print(f"[Resume] {len(X16)}枚, 済:{sorted(done)}", flush=True)
    parts = ["p00"] if "--test" in sys.argv else [f"p{i:02d}" for i in range(15)]
    t0 = time.time()
    for pid in parts:
        if pid in done: print(f"{pid}: skip", flush=True); continue
        try: wpx, hpx = screen_size(pid)
        except Exception as e: print(f"{pid}: NG {e}"); continue
        lines = open(MPII / pid / f"{pid}.txt").read().strip().splitlines()
        if limit: lines = lines[:limit]
        ok = 0
        for ln in lines:
            f = ln.split(); imgp, gx, gy = f[0], float(f[1]), float(f[2])
            img = cv2.imread(str(MPII / pid / imgp))
            if img is None: continue
            h, w = img.shape[:2]
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if not res.face_landmarks: continue
            arr = lms_to_array(res.face_landmarks[0])
            feat = rich_16d_from_lms(arr, w, h)
            if feat is None or not np.isfinite(feat).all(): continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = CLAHE.apply(gray)
            try:
                pp = np.concatenate([patch(gray, arr, w, h, oi, ii) for oi, ii in EYES])
                pc = np.concatenate([patch(clahe, arr, w, h, oi, ii) for oi, ii in EYES])
            except Exception: continue
            if not (np.isfinite(pp).all() and np.isfinite(pc).all()): continue
            X16.append(feat); XP.append(pp); XC.append(pc); Y.append([gx/wpx, gy/hpx]); PID.append(pid); ok += 1
        if "--test" not in sys.argv:
            np.savez_compressed(str(ck), X16=np.array(X16, np.float32), Xplain=np.array(XP, np.float32),
                                Xclahe=np.array(XC, np.float32), y=np.array(Y, np.float32), pid=np.array(PID))
        print(f"{pid}: {ok}/{len(lines)} (計{len(X16)}, {ok/(time.time()-t0+1e-6):.0f}fps)", flush=True)
    out = ROOT / "cache" / ("mpii_app2_test.npz" if "--test" in sys.argv else "mpii_app2.npz")
    np.savez_compressed(str(out), X16=np.array(X16, np.float32), Xplain=np.array(XP, np.float32),
                        Xclahe=np.array(XC, np.float32), y=np.array(Y, np.float32), pid=np.array(PID))
    print(f"[Done] {len(X16)}枚 → {out}", flush=True)

if __name__ == "__main__":
    main()
