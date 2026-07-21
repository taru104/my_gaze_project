"""MPIIFaceGaze の画像を mediapipe 処理し、アプリと同一の 7D 特徴 + 画面正規化座標を抽出。
汎用性(person-independent)検証用の学習データを作る。

各 pXX/pXX.txt: [img_path, gaze_x_px, gaze_y_px, ...6点/姿勢...]
screenSize.mat: width_pixel/height_pixel で画面座標を [0,1] に正規化。
7D 特徴は rich16d.rich_16d_from_lms(...)[:7] = アプリと厳密同一。

Usage:
    .venv/Scripts/python.exe benchmarks/extract_mpii.py --test   # p00の少数で動作確認
    .venv/Scripts/python.exe benchmarks/extract_mpii.py          # 全15人(要チェックポイント, run_in_background)
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

ROOT  = Path(__file__).parent.parent
MPII  = ROOT / "MPIIFaceGaze"
MODEL = ROOT / "face_landmarker.task"
CK_EVERY = 10000


def make_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def screen_size(pid):
    ss = sio.loadmat(str(MPII / pid / "Calibration" / "screenSize.mat"))
    return float(ss["width_pixel"][0][0]), float(ss["height_pixel"][0][0])


def main():
    test = "--test" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    tag = ""
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    lm = make_landmarker()
    ckpath = ROOT / "cache" / f"mpii_7d{tag}_ck.npz"
    X, Y, PID = [], [], []
    done = set()
    # Resume: 既に処理済みの人はスキップ(人ごとckから復元)
    if not test and ckpath.exists():
        d = np.load(ckpath)
        X = list(d["X"]); Y = list(d["y"]); PID = list(d["pid"].tolist())
        done = set(PID)
        print(f"[Resume] {len(X)}フレーム復元, 済み: {sorted(done)}", flush=True)
    parts = ["p00"] if test else [f"p{i:02d}" for i in range(15)]
    t0 = time.time()
    for pid in parts:
        if pid in done:
            print(f"{pid}: skip(処理済み)", flush=True); continue
        try:
            wpx, hpx = screen_size(pid)
        except Exception as e:
            print(f"{pid}: screenSize読めず {e}"); continue
        lines = open(MPII / pid / f"{pid}.txt").read().strip().splitlines()
        if test:
            lines = lines[:40]
        elif limit:
            lines = lines[:limit]   # 高速版: 各人 limit 枚だけ(汎用性評価に十分)
        ok = 0
        for ln in lines:
            f = ln.split()
            imgp, gx, gy = f[0], float(f[1]), float(f[2])
            img = cv2.imread(str(MPII / pid / imgp))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks:
                continue
            arr = lms_to_array(res.face_landmarks[0])
            feat = rich_16d_from_lms(arr, w, h)
            if feat is None or not np.isfinite(feat).all():
                continue
            X.append(feat[:7]); Y.append([gx / wpx, gy / hpx]); PID.append(pid); ok += 1
        # 人ごとにチェックポイント保存(途中停止に強い・最大1人分の損失で済む)
        if not test:
            np.savez_compressed(str(ckpath),
                                X=np.array(X, np.float32), y=np.array(Y, np.float32), pid=np.array(PID))
        el = time.time() - t0 + 1e-6
        print(f"{pid}: {ok}/{len(lines)} 処理  (計{len(X)}フレーム, {ok/el:.0f}fps)", flush=True)
    X = np.array(X, np.float32); Y = np.array(Y, np.float32); PID = np.array(PID)
    out = ROOT / "cache" / ("mpii_7d_test.npz" if test else f"mpii_7d{tag}.npz")
    np.savez_compressed(str(out), X=X, y=Y, pid=PID)
    print(f"[Done] {len(X)} frames → {out}")
    if len(X):
        print(f"  y_norm x[{Y[:,0].min():.2f},{Y[:,0].max():.2f}] y[{Y[:,1].min():.2f},{Y[:,1].max():.2f}]")
        print(f"  被験者: {sorted(set(PID.tolist()))}")


if __name__ == "__main__":
    main()
