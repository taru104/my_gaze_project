"""
保存した生ランドマークログ(logs/session_*_landmarks.bin)から、
オフラインで任意次元の特徴を再計算する。現状は16D(extract_rich相当)を再構成。

将来もっと高次元の特徴が欲しくなったら、rich_16d_from_lms を拡張するだけで
過去の全録画から再計算できる(生ランドマークを残しているから可能)。

数値は extract_rich_features.extract_rich(画像版)と一致する(同じ定数・同じ数式を使用)。
検証は scripts の validate で実画像照合済み。

Usage:
    .venv/Scripts/python.exe benchmarks/reprocess_raw_landmarks.py logs/session_<id>_landmarks
    → logs/session_<id>_rich16d.npz (X:(N,16), y_norm, has_target, frame_idx, time_s)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
# 16D特徴の単一定義（ライブ features.py と共有・divergence防止）
from rich16d import rich_16d_from_lms
from raw_landmark_logger import load_raw_landmarks


def main():
    if len(sys.argv) < 2:
        print("usage: reprocess_raw_landmarks.py logs/session_<id>_landmarks"); return
    base = sys.argv[1]
    d = load_raw_landmarks(base)
    n = d["n"]
    X, yn, ht, fi, ts = [], [], [], [], []
    ok = 0
    for k in range(n):
        w = int(d["img_w"][k]); h = int(d["img_h"][k])
        feat = rich_16d_from_lms(d["landmarks"][k], w, h)
        if feat is None or not np.isfinite(feat).all():
            continue
        X.append(feat); yn.append(d["target"][k]); ht.append(bool(d["has_target"][k]))
        fi.append(int(d["frame_idx"][k])); ts.append(float(d["time_s"][k])); ok += 1
    out = Path(str(base).replace("_landmarks", "_rich16d")).with_suffix(".npz")
    np.savez_compressed(str(out), X=np.array(X, np.float32), y_norm=np.array(yn, np.float32),
                        has_target=np.array(ht, bool), frame_idx=np.array(fi, np.int64),
                        time_s=np.array(ts, np.float32))
    print(f"[Done] {ok}/{n} frames → {out}")
    print(f"  has_target(正解あり) frames: {int(np.sum(ht))}")


if __name__ == "__main__":
    main()
