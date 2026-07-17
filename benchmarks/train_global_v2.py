"""
グローバルモデル v2: 左右対称ミラー拡張 + アンサンブル。

ミラー拡張(水平反転)の物理:
  画像を左右反転すると:
    左目↔右目 が入れ替わり、x成分の符号が反転:
      Lx' = -Rx, Ly' = Ry, Rx' = -Lx, Ry' = Ly
    Yaw' = -Yaw (左右反転), Pitch'=Pitch, dist'=dist
    目標 y_cm: x' = -x_cm, y' = y_cm
  → 横向きサンプルを対称に倍化でき、左右非対称な偏りを消す。

アンサンブル: 異なるseedのMLPを平均して分散低減。

Usage:
    .venv/Scripts/python.exe benchmarks/train_global_v2.py --mirror --ensemble 3
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import time, argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import euclidean_cm, pose_mag, bin_report, print_bins, print_header

PROJECT_DIR = Path(__file__).parent.parent
CACHE_BIG   = PROJECT_DIR / "cache" / "gazeCapture_features_cache.npz"
CACHE_TEST  = PROJECT_DIR / "cache" / "sota_7d_cache.npz"
MODEL_OUT   = PROJECT_DIR / "cache" / "global_mlp_v2.joblib"


def mirror_augment(X, y_cm):
    """水平反転でデータを倍化。"""
    Xm = np.empty_like(X)
    Xm[:, 0] = -X[:, 2]   # Lx' = -Rx
    Xm[:, 1] =  X[:, 3]   # Ly' = Ry
    Xm[:, 2] = -X[:, 0]   # Rx' = -Lx
    Xm[:, 3] =  X[:, 1]   # Ry' = Ly
    Xm[:, 4] =  X[:, 4]   # Pitch
    Xm[:, 5] = -X[:, 5]   # Yaw'
    Xm[:, 6] =  X[:, 6]   # dist
    ym = y_cm.copy()
    ym[:, 0] = -y_cm[:, 0]
    return np.vstack([X, Xm]), np.vstack([y_cm, ym])


class Ensemble:
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def make_mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                     alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                     max_iter=300, early_stopping=True, n_iter_no_change=15,
                     validation_fraction=0.08, random_state=seed),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--ensemble", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    db = np.load(str(CACHE_BIG))
    Xb, ycb, scb = db["X"], db["y_cm"], db["split_code"]
    tr = scb == 0
    Xtr, ytr = Xb[tr], ycb[tr]
    if args.mirror:
        Xtr, ytr = mirror_augment(Xtr, ytr)
        print(f"[Mirror] 拡張後 {len(Xtr)} frames")
    else:
        print(f"[Train] {len(Xtr)} frames")

    models = []
    for s in range(args.ensemble):
        m = make_mlp(s)
        m.fit(Xtr, ytr)
        models.append(m)
        print(f"  MLP seed={s} fit ({time.time()-t0:.0f}s)")
    model = Ensemble(models) if len(models) > 1 else models[0]

    # 評価
    dt = np.load(str(CACHE_TEST))
    Xt, yct, subj = dt["X"], dt["y_cm"], dt["subj_id"]
    euc = euclidean_cm(model.predict(Xt), yct)
    mag = pose_mag(Xt)
    rep = bin_report(euc, mag)

    tag = f"v2 mirror={args.mirror} ens={args.ensemble}"
    print(f"\n{'='*84}")
    print(f"  グローバル {tag}  (無キャリブ, bin別 median cm)")
    print(f"{'='*84}")
    print_header()
    print_bins(tag, rep)
    print(f"\n[{time.time()-t0:.0f}s]")

    joblib.dump(model, MODEL_OUT)
    print(f"[Saved] {MODEL_OUT}")


if __name__ == "__main__":
    main()
