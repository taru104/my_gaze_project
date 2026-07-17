"""
実アプリ統合の核心検証: グローバルモデルを正規化スクリーン座標 y_norm
(アプリと同じ空間)で学習しても頭部ロバスト性が保たれるか?

cm版と同じく pose bin で誤差が平坦なら、アプリへ直接転用できる。
指標は正規化座標のユークリッド誤差(単位[0,1])。cmと直接比較はできないが
「turn/front の比」でロバスト性(平坦さ)を評価する。

Usage:
    .venv/Scripts/python.exe benchmarks/ynorm_robustness.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import POSE_BINS

ROOT = Path(__file__).parent.parent
BIG  = ROOT / "cache" / "gazeCapture_features_cache.npz"
TEST = ROOT / "cache" / "sota_7d_cache.npz"


def eucl(p, g):
    return np.sqrt(np.sum((p - g) ** 2, axis=-1))


def main():
    t0 = time.time()
    db = np.load(str(BIG))
    Xb, ynb, scb = db["X"], db["y_norm"], db["split_code"]
    tr = scb == 0
    # 軽量MLP(診断用, 抽出と競合を抑える)
    mlp = make_pipeline(StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64,32), activation="relu", alpha=1e-3,
                     batch_size=512, max_iter=50, early_stopping=True,
                     n_iter_no_change=8, random_state=0))
    mlp.fit(Xb[tr], ynb[tr])
    print(f"[Fit y_norm global] {time.time()-t0:.0f}s")

    dt = np.load(str(TEST))
    Xt, ynt = dt["X"], dt["y_norm"]
    mag = np.sqrt(np.degrees(Xt[:,4])**2 + np.degrees(Xt[:,5])**2)
    euc = eucl(mlp.predict(Xt), ynt)

    print(f"\n  pose bin別 median正規化誤差 (単位=画面比):")
    front = turn = None
    for lo, hi in POSE_BINS:
        bm = (mag>=lo)&(mag<hi)
        if bm.sum() >= 20:
            v = np.median(euc[bm])
            print(f"    [{lo:>2},{hi:>2}): {v:.4f}  (frames {bm.sum()})")
    # front(<15) vs turn(>=20)
    fm = mag < 15; tm = mag >= 20
    ef = np.median(euc[fm]); et = np.median(euc[tm])
    print(f"\n  front(<15°)={ef:.4f}  turn(>=20°)={et:.4f}  劣化比={et/ef:.2f}x")
    print(f"  overall median={np.median(euc):.4f} (正規化座標)")
    print(f"\n  cm版グローバルの劣化比は約1.0(平坦)だった。y_norm版も同程度なら")
    print(f"  アプリ(正規化座標)へ直接転用可。1.3x以上なら座標変換設計が必要。")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
