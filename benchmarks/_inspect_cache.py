"""キャッシュnpzの中身を確認する。高速反復実験の基盤把握用。"""
import numpy as np
from pathlib import Path

CACHE = Path(__file__).parent.parent / 'cache'
for name in ['sota_7d_cache.npz', 'sota_486d_cache.npz',
             'sota_7d_checkpoint.npz', 'gazeCapture_features_cache.npz']:
    p = CACHE / name
    if not p.exists():
        print(f"[MISSING] {name}")
        continue
    d = np.load(p, allow_pickle=True)
    print(f"\n=== {name} ({p.stat().st_size/1e6:.1f} MB) ===")
    for k in d.files:
        arr = d[k]
        shape = getattr(arr, 'shape', None)
        dtype = getattr(arr, 'dtype', None)
        extra = ''
        if shape is not None and len(shape) >= 1 and shape[0] > 0 and np.issubdtype(arr.dtype, np.number):
            try:
                extra = f" range=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]"
            except Exception:
                pass
        print(f"  {k:20s} shape={shape} dtype={dtype}{extra}")
        if k in ('subj_id', 'subject', 'subj') and shape is not None:
            print(f"     unique subjects: {len(set(arr.tolist()))}")
