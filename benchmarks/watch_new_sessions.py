"""追加録画を待ち受けて自動で精度探索する監視ループ（5時間駆動用）。

ユーザが main.py で新しくキャリブ/利用すると logs/session_*_landmarks.bin が増える。
それを検知したら自動で:
  1. reprocess_raw_landmarks で 16D化 (→ *_rich16d.npz)
  2. explore_accuracy(段階1) で 7D Huber 実効を評価
  3. 全セッション合算での精度も測る(点数が実質増える＝1cmに近づくか)
結果は results/exploration_log.md に追記。

録画中の未完成ファイルを処理しないよう、サイズが N秒安定してから着手する。

Usage:
    .venv/Scripts/python.exe benchmarks/watch_new_sessions.py [max_minutes]
"""
import sys, time, glob, subprocess
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

ROOT = Path(__file__).parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "results" / "exploration_log.md"
SCREEN = np.array([30.9, 17.4])


def logline(s):
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def bins():
    return sorted(glob.glob(str(ROOT / "logs" / "session_*_landmarks.bin")))


def stable(path, wait=8):
    """ファイルサイズが wait 秒変化しなければ True(録画完了とみなす)。"""
    try:
        s0 = Path(path).stat().st_size
    except OSError:
        return False
    time.sleep(wait)
    try:
        return Path(path).stat().st_size == s0 and s0 > 1000
    except OSError:
        return False


def process(binpath):
    base = binpath[:-len(".bin")]  # ..._landmarks
    sid = Path(base).name.replace("_landmarks", "").replace("session_", "")
    logline(f"\n### [watch] 新セッション検出 {sid} — 自動探索")
    # 1. 16D化
    r = subprocess.run([PY, str(ROOT / "benchmarks" / "reprocess_raw_landmarks.py"), base],
                       capture_output=True, text=True)
    logline(f"  reprocess: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode}")
    npz = ROOT / "logs" / f"session_{sid}_rich16d.npz"
    if not npz.exists():
        logline("  [watch] rich16d.npz が出来ず。スキップ"); return
    # 2. 単体探索(段階1のモデル部分だけ; 軽くするためstage=models)
    r = subprocess.run([PY, str(ROOT / "benchmarks" / "explore_accuracy.py"), sid, "models"],
                       capture_output=True, text=True)
    # 3. 全セッション合算精度(点数が増える効果)
    try:
        combined_eval()
    except Exception as ex:
        logline(f"  合算評価 err: {ex}")


def combined_eval():
    """全 *_rich16d.npz を合算し、7D Huber の点ごとLOO(session混在)を測る。
    セッションが増えるほど各点のデータが厚くなり、汎化が上がるか観察。"""
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    npzs = sorted(glob.glob(str(ROOT / "logs" / "session_*_rich16d.npz")))
    if len(npzs) < 2:
        return
    Xs, ys, sess = [], [], []
    for i, p in enumerate(npzs):
        d = np.load(p)
        m = d["has_target"].astype(bool)
        Xs.append(d["X"][m][:, :7]); ys.append(d["y_norm"][m]); sess.append(np.full(m.sum(), i))
    X = np.vstack(Xs); y = np.vstack(ys); sess = np.concatenate(sess)
    uniq, ids = np.unique(np.round(y, 4), axis=0, return_inverse=True)
    PM, GM = [], []
    for p in np.unique(ids):
        te, tr = ids == p, ids != p
        sc = StandardScaler().fit(X[tr])
        mx = HuberRegressor(max_iter=800).fit(sc.transform(X[tr]), y[tr, 0])
        my = HuberRegressor(max_iter=800).fit(sc.transform(X[tr]), y[tr, 1])
        pr = np.column_stack([mx.predict(sc.transform(X[te])), my.predict(sc.transform(X[te]))])
        PM.append(np.median(pr, axis=0)); GM.append(uniq[p])
    e = np.sqrt(((np.array(PM) - np.array(GM))[:, 0] * SCREEN[0]) ** 2 +
                ((np.array(PM) - np.array(GM))[:, 1] * SCREEN[1]) ** 2)
    logline(f"  [合算] {len(npzs)}セッション {len(X)}フレーム {len(uniq)}点 → "
            f"7D Huber 実効 median={np.median(e):.3f}cm")


def main():
    max_min = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    t0 = time.time()
    known = set(bins())
    logline(f"\n[watch] 監視開始 既知{len(known)}セッション max={max_min}分。新録画を待機...")
    while (time.time() - t0) < max_min * 60:
        cur = set(bins())
        new = sorted(cur - known)
        for f in new:
            if stable(f):
                try:
                    process(f)
                except Exception as ex:
                    logline(f"  [watch] 処理エラー {f}: {ex}")
                known.add(f)
            # 未安定なら次ループで再チェック(known に入れない)
        time.sleep(30)
    logline(f"[watch] 監視終了 ({max_min}分経過)")


if __name__ == "__main__":
    main()
