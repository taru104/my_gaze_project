"""
勝った16DグローバルMLPを本番用に永続化する。

rich_hybrid_eval.py で 16D rich hybrid が全姿勢binで7Dを上回った(overall 3.339cm,
7Dハイブリッド比26%改善)。その "16D global" を同一構成で train split 全体に学習して保存。

構成 = rich_hybrid_eval.make_mlp と完全一致:
  StandardScaler -> MLP(128,64,32, relu, alpha=1e-4, batch=512, lr=1e-3,
                        max_iter=300, early_stopping, n_iter_no_change=15)

出力:
  cache/global_mlp_16d.joblib      … sklearn pipeline (.predict(X:(N,16))->cm(N,2))
  cache/global_mlp_16d.meta.json   … 特徴名・次元・学習情報(アプリが整合性チェックに使う)

Usage:
    .venv/Scripts/python.exe benchmarks/train_global_16d.py
"""
import sys, json, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent.parent
BIG_RICH  = ROOT / "cache" / "rich_features_cache.npz"
MODEL_OUT = ROOT / "cache" / "global_mlp_16d.joblib"
META_OUT  = ROOT / "cache" / "global_mlp_16d.meta.json"

FEATURE_NAMES = ['Lx', 'Ly', 'Rx', 'Ry', 'Pitch', 'Yaw', 'dist',
                 'roll', 'L_EAR', 'R_EAR', 'L_ivert', 'R_ivert',
                 'L_idiam', 'R_idiam', 'L_aspect', 'R_aspect']


def make_mlp(seed=0):
    return make_pipeline(StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", alpha=1e-4,
                     batch_size=512, learning_rate_init=1e-3, max_iter=300,
                     early_stopping=True, n_iter_no_change=15, random_state=seed))


def main():
    if not BIG_RICH.exists():
        print(f"[Error] {BIG_RICH} が無い。先に extract_rich_features.py を完了させる。")
        return
    t0 = time.time()
    d = np.load(str(BIG_RICH))
    X, y_cm, sc = d["X"], d["y_cm"], d["split_code"]
    assert X.shape[1] == 16, f"次元不一致: {X.shape[1]} != 16"
    tr = sc == 0
    Xtr, ytr = X[tr], y_cm[tr]
    print(f"[Train] {tr.sum()} frames, dim={X.shape[1]}")

    model = make_mlp().fit(Xtr, ytr)
    print(f"  fit done ({time.time()-t0:.0f}s)")

    # 学習内誤差(参考): train上の中央値cm
    pred = model.predict(Xtr)
    euc = np.sqrt(((pred - ytr) ** 2).sum(1))
    print(f"  train median={np.median(euc):.3f}cm mean={euc.mean():.3f}cm")

    joblib.dump(model, MODEL_OUT)
    meta = {
        "dim": 16,
        "feature_names": FEATURE_NAMES,
        "target": "y_cm (GazeCapture cm space, デバイス非依存)",
        "arch": "StandardScaler -> MLP(128,64,32)",
        "train_frames": int(tr.sum()),
        "train_median_cm": float(np.median(euc)),
        "note": "rich_hybrid_eval で 16D rich hybrid overall 3.339cm(7Dハイブリッド比26%改善)。"
                "HybridCalibration(global_model=この model) に16D特徴で使う。",
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] {time.time()-t0:.0f}s -> {MODEL_OUT.name} + {META_OUT.name}")


if __name__ == "__main__":
    main()
