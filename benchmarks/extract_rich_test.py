"""
test被験者(split==2)のみ rich特徴(14D)を subj_id 付きで抽出。
被験者別ハイブリッド評価用。大規模抽出(extract_rich_features.py)完了後に実行する。

出力: cache/rich_test_cache.npz  (X:(N,14), y_cm, y_norm, subj_id)

Usage:
    .venv/Scripts/python.exe benchmarks/extract_rich_test.py
"""
import sys, io, os, time, re
# stdout を UTF-8 に。TextIOWrapper で包み直すと元stdoutのGC時に下層バッファが
# 閉じられ "I/O operation on closed file" になる罠があるため reconfigure を使う。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from gazeCapture_dataset import GazeCaptureRawIndex
from extract_rich_features import extract_rich, make_landmarker

ROOT = Path(__file__).parent.parent
ARCHIVE = ROOT / "archive"
OUT = ROOT / "cache" / "rich_test_cache.npz"

# img_path から被験者ID(数値)を取り出す: archive/00010/00010/frames/xxx.jpg -> 10
_SUBJ_RE = re.compile(r"archive[/\\](\d+)[/\\]")


def subj_of(img_path):
    m = _SUBJ_RE.search(img_path.replace("\\", "/"))
    return int(m.group(1)) if m else -1


def main():
    t0 = time.time()
    index = GazeCaptureRawIndex(str(ARCHIVE))
    test_recs = [r for r in index.records if r[5] == 2]   # split_code==2
    print(f"[Test] {len(test_recs)} frames from split=test")
    lm = make_landmarker()

    X, yc, yn, sid = [], [], [], []
    n_ok = n_fail = 0
    for i, rec in enumerate(test_recs):
        if i > 0 and i % 4000 == 0:
            fps = i / (time.time() - t0)
            print(f"  [{100*i/len(test_recs):.0f}%] {i}/{len(test_recs)} ok={n_ok} "
                  f"{fps:.0f}fps ETA {(len(test_recs)-i)/fps/60:.1f}min")
        img = cv2.imread(rec[0])
        if img is None:
            n_fail += 1; continue
        fr = extract_rich(img, lm, rec[6] if len(rec) > 6 else 1)
        if fr is None:
            n_fail += 1; continue
        X.append(fr); yn.append([rec[1], rec[2]]); yc.append([rec[3], rec[4]])
        sid.append(subj_of(rec[0])); n_ok += 1
    lm.close()

    np.savez_compressed(str(OUT), X=np.array(X, np.float32),
                        y_norm=np.array(yn, np.float32), y_cm=np.array(yc, np.float32),
                        subj_id=np.array(sid, np.int32))
    print(f"\n[Done] {time.time()-t0:.0f}s ok={n_ok} fail={n_fail} "
          f"subjects={len(set(sid))} -> {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
