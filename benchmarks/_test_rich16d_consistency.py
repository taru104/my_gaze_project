"""rich16d.py の定数が extract_rich_features(本番抽出) と一致するか検証。
数式は reprocess から移植した同一物なので、定数一致を確認すれば divergence しない。"""
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import rich16d as R
import extract_rich_features as E

checks = {
    "_FACE_3D_MODEL": (R._FACE_3D_MODEL, E._FACE_3D_MODEL),
    "_FACE_2D_IDX":   (R._FACE_2D_IDX,   E._FACE_2D_IDX),
    "_LEFT_IRIS":     (R._LEFT_IRIS,     E._LEFT_IRIS),
    "_RIGHT_IRIS":    (R._RIGHT_IRIS,    E._RIGHT_IRIS),
    "_L_IN":  (R._L_IN, E._L_IN), "_L_OUT": (R._L_OUT, E._L_OUT),
    "_R_IN":  (R._R_IN, E._R_IN), "_R_OUT": (R._R_OUT, E._R_OUT),
    "_L_UP":  (R._L_UP, E._L_UP), "_L_LO":  (R._L_LO, E._L_LO),
    "_R_UP":  (R._R_UP, E._R_UP), "_R_LO":  (R._R_LO, E._R_LO),
    "_DIST":  (R._DIST, E._DIST),
}
bad = []
for name, (a, b) in checks.items():
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        bad.append(name)
        print(f"  [MISMATCH] {name}: {a} != {b}")

# _geo_normalize の数値一致(ランダム入力ではなく固定値で)
p = np.array([10.0, 5.0]); i = np.array([0.0, 0.0]); o = np.array([20.0, 2.0])
if not np.allclose(R._geo_normalize(p, i, o), E._geo_normalize(p, i, o)):
    bad.append("_geo_normalize")
    print("  [MISMATCH] _geo_normalize")

if bad:
    print(f"[FAIL] {len(bad)} 個の定数/関数が不一致: {bad}")
    sys.exit(1)
print("[OK] rich16d.py は extract_rich_features と定数・_geo_normalize 完全一致")
