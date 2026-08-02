"""exp65: 「較正して悪化する」を潰す — 収縮(shrinkage)付き適応。

exp64 で判明した実害: LOPO で較正サンプルが少ないと**較正しない方がマシ**になる。
  16D+パッチ: k=0 6.24° → k=1 7.28° → k=3 11.54° → k=5 7.91° → k=9 6.33° → k=16 5.73°
k=3 のアフィン(6自由度)を3点で解くのは劣決定に近く、外挿が暴れる。k=1のバイアスも1点ではノイズ。
実アプリでも「数回タップしただけで精度が落ちる」ことを意味するので、放置できない。

方針: 補正を**残差リッジ**として解き、正則化を較正点数に応じて効かせる。
  r = y_cal - base_cal  を  r ≈ M [base_cal, 1] で説明し、pred = base + M[base,1]。
  alpha 大 → M→0 → 「補正しない」(=k=0と同じ)に安全に縮退する。
  alpha は k サンプル上の leave-one-out で選ぶ(データが少ないほど自動的に強い正則化が選ばれる)。
狙い: **k に対して単調改善し、どの k でも k=0 を下回らない**こと。

Usage: .venv/Scripts/python.exe experiments/exp65_safe_adapt.py
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import cv2, scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA

REPORT = ROOT / "experiments" / "REPORT8_metrics.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

d = np.load(ROOT / "cache" / "mpii_full.npz")
X16, patch, y, fc, gt, scr, pid = (d["X16"], d["patch"], d["y_norm"],
                                   d["fc"], d["gt"], d["scr"], d["pid"])
pids = sorted(set(pid.tolist()))

def monitor_pose(p):
    m = sio.loadmat(str(ROOT / "MPIIFaceGaze" / p / "Calibration" / "monitorPose.mat"))
    R = cv2.Rodrigues(np.ravel(m["rvects"] if "rvects" in m else m["rvecs"]).astype(np.float64))[0]
    return R, np.ravel(m["tvecs"]).astype(np.float64)
MP = {p: monitor_pose(p) for p in pids}

def err_deg(pred, idx):
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        R, T = MP[pid[i]]
        p3 = R @ np.array([pred[k][0] * scr[i][0], pred[k][1] * scr[i][1], 0.0]) + T
        a, b = p3 - fc[i], gt[i] - fc[i]
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        out[k] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out

def zpatch(rows):
    P = patch[rows].astype(np.float32)
    f = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return np.hstack([f(P[:, :1536]), f(P[:, 1536:])])

def global_model(tr, npca=16):
    sub = tr if len(tr) <= 12000 else np.random.RandomState(0).choice(tr, 12000, replace=False)
    pca = PCA(n_components=npca, svd_solver="randomized", random_state=0).fit(zpatch(sub))
    A0 = np.hstack([X16[tr], pca.transform(zpatch(tr))])
    sc = StandardScaler().fit(A0)
    hs = [HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500).fit(sc.transform(A0), y[tr][:, i])
          for i in range(2)]
    tf = lambda te: sc.transform(np.hstack([X16[te], pca.transform(zpatch(te))]))
    return lambda te: np.column_stack([h.predict(tf(te)) for h in hs])

# ── 適応の3方式 ────────────────────────────────────────────
def adapt_old(bc, yc, bt, k):
    """exp64 と同じ(比較用): k<3 はバイアス、k>=3 は正則化ほぼ無しのアフィン。"""
    if k == 0: return bt
    if k < 3:  return bt + (yc - bc).mean(0)
    D = np.hstack([bc, np.ones((len(bc), 1))])
    W = np.linalg.lstsq(D.T @ D + 1e-3 * np.eye(3), D.T @ yc, rcond=None)[0]
    return np.hstack([bt, np.ones((len(bt), 1))]) @ W

ALPHAS = np.array([1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4])

def _ridge_resid(D, r, alpha):
    """残差 r を D で説明するリッジ解。alpha→∞ で 0 に縮退(=補正しない)。"""
    G = D.T @ D + alpha * np.eye(D.shape[1])
    return np.linalg.solve(G, D.T @ r)

def adapt_shrink(bc, yc, bt, k):
    """残差リッジ + alpha を較正点上の LOO で自動選択。データが少なければ強く縮退する。"""
    if k == 0: return bt
    D = np.hstack([bc, np.ones((len(bc), 1))])
    r = yc - bc
    if k == 1:                                    # 1点では傾き成分を作らない(バイアスのみ+縮小)
        return bt + r.mean(0) * 0.5
    best, best_a = np.inf, ALPHAS[-1]
    for a in ALPHAS:                              # leave-one-out で alpha を選ぶ
        se = 0.0
        for j in range(len(D)):
            m = np.ones(len(D), bool); m[j] = False
            W = _ridge_resid(D[m], r[m], a)
            se += np.sum((r[j] - D[j] @ W) ** 2)
        if se < best: best, best_a = se, a
    W = _ridge_resid(D, r, best_a)
    return bt + np.hstack([bt, np.ones((len(bt), 1))]) @ W


def adapt_guarded(bc, yc, bt, k):
    """収縮 + **『補正しない』を明示的な候補に入れる**。

    LOO で選んだ補正が、無補正(W=0)より良いと確認できたときだけ適用する。
    k が小さいと LOO 自体がノイジーで有害な alpha を選びうるため、この保険で
    「較正して悪化する」を構造的に断つ(最悪でも k=0 と同じ)。
    """
    if k == 0: return bt
    D = np.hstack([bc, np.ones((len(bc), 1))])
    r = yc - bc
    loo_none = float(np.sum(r ** 2))              # 無補正の LOO 誤差(そのまま残差)
    best, best_W = loo_none, None
    cands = [("bias", None)] + [("ridge", a) for a in ALPHAS]
    for kind, a in cands:
        se = 0.0
        for j in range(len(D)):
            m = np.ones(len(D), bool); m[j] = False
            if kind == "bias":
                pred_j = r[m].mean(0)
            else:
                pred_j = D[j] @ _ridge_resid(D[m], r[m], a)
            se += np.sum((r[j] - pred_j) ** 2)
        if se < best:
            best = se
            best_W = ("bias", None) if kind == "bias" else ("ridge", a)
    if best_W is None:                            # どれも無補正に勝てない → 触らない
        return bt
    if best_W[0] == "bias":
        return bt + r.mean(0)
    W = _ridge_resid(D, r, best_W[1])
    return bt + np.hstack([bt, np.ones((len(bt), 1))]) @ W

KS = [0, 1, 3, 5, 9, 16, 45, 100]
SEEDS = 5
METHODS = [("exp64の適応(現行)", adapt_old), ("収縮付き適応", adapt_shrink),
           ("**収縮+無補正ガード(提案)**", adapt_guarded)]

log(f"\n## 3. exp65: 「較正すると悪化する」問題と収縮付き適応（16D+パッチ, LOPO, subject-macro角度RMSE）\n")
res = {name: {k: [] for k in KS} for name, _ in METHODS}
for p in pids:
    te_all = np.where(pid == p)[0]; tr = np.where(pid != p)[0]
    base_all = global_model(tr)(te_all)
    for name, fn in METHODS:
        for k in KS:
            rms = []
            for s in range(SEEDS if k > 0 else 1):
                rs = np.random.RandomState(1000 * s + 7)
                cal = rs.choice(len(te_all), k, replace=False) if k else np.array([], int)
                mask = np.ones(len(te_all), bool); mask[cal] = False
                pr = fn(base_all[cal], y[te_all[cal]], base_all[mask], k)
                rms.append(float(np.sqrt(np.mean(err_deg(pr, te_all[mask]) ** 2))))
            res[name][k].append(np.mean(rms))
    print(f"  {p} 完了", flush=True)

log("| 適応方式 | " + " | ".join(f"k={k}" for k in KS) + " |")
log("|---|" + "---:|" * len(KS))
for name, _ in METHODS:
    log(f"| {name} | " + " | ".join(f"{np.mean(res[name][k]):.2f}°" for k in KS) + " |")

base0 = np.mean(res["exp64の適応(現行)"][0])
log("")
log(f"- 較正なし(k=0) = {base0:.2f}°。**どの k でもこれを上回って(悪化して)はいけない**というのが要件。")
for name, _ in METHODS:
    w = max(np.mean(res[name][k]) for k in KS)
    bad = [k for k in KS if np.mean(res[name][k]) > base0 + 1e-9]
    log(f"- {name}: 最悪 {w:.2f}°  / k=0 より悪化する k = {bad if bad else 'なし（要件クリア）'}")
