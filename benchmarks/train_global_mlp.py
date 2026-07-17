"""
グローバルMLPを収束まで学習して cache/ に保存する。
以降のローカル補正実験はこの保存モデルを読むだけで高速反復できる。

Usage:
    .venv/Scripts/python.exe benchmarks/train_global_mlp.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_DIR = Path(__file__).parent.parent
CACHE_BIG   = PROJECT_DIR / "cache" / "gazeCapture_features_cache.npz"
CACHE_TEST  = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
MODEL_OUT   = PROJECT_DIR / "cache" / "global_mlp.joblib"


def euclidean_cm(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))


def main():
    t0 = time.time()
    db = np.load(str(CACHE_BIG))
    Xb, ycb, scb = db["X"], db["y_cm"], db["split_code"]
    tr = scb == 0; va = scb == 1
    Xtr, ytr = Xb[tr], ycb[tr]
    Xva, yva = Xb[va], ycb[va]
    print(f"[Train] {len(Xtr)} frames, [Val] {len(Xva)} frames")

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                     activation="relu", alpha=1e-4,
                     batch_size=512, learning_rate_init=1e-3,
                     max_iter=300, early_stopping=True,
                     n_iter_no_change=15, validation_fraction=0.08,
                     random_state=0, verbose=False),
    )
    model.fit(Xtr, ytr)
    mlp = model.named_steps["mlpregressor"]
    print(f"[Fit] iters={mlp.n_iter_}  best_val_loss={mlp.best_validation_score_:.4f}  "
          f"({time.time()-t0:.0f}s)")

    # val精度
    ev = euclidean_cm(model.predict(Xva), yva)
    print(f"[Val] Euc mean={ev.mean():.3f} median={np.median(ev):.3f} cm")

    # test精度 (無キャリブ)
    dt = np.load(str(CACHE_TEST))
    Xt, yct, subj = dt["X"], dt["y_cm"], dt["subj_id"]
    et = euclidean_cm(model.predict(Xt), yct)
    print(f"[Test raw] Euc mean={et.mean():.3f} median={np.median(et):.3f} cm (無キャリブ)")

    joblib.dump(model, MODEL_OUT)
    print(f"[Saved] {MODEL_OUT}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
