"""exp48: 姿勢条件づき交互作用特徴。iris位置×頭部姿勢角の交互作用項を足し、
「虹彩ズレ→視線ゲイン」を姿勢依存にする(単一連続モデル=hybrid切替の破綻回避)。
自分honest(基準4.71cm)＋MPII15名=客観(個人内4.18/person-indep6.18)の両輪で検証。mainは触らない。
"""
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from raw_landmark_logger import load_raw_landmarks
from rich16d import rich_16d_from_lms
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
CM_W = 30.0
REPORT = ROOT / "experiments" / "REPORT6_beyond16d.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

# ---- 特徴拡張 ----
# 16D: [Lx,Ly,Rx,Ry, pitch,yaw,dist, roll, L_EAR,R_EAR, L_ivert,R_ivert, L_idiam,R_idiam, L_asp,R_asp]
def augment(X, mode):
    Xr = np.asarray(X, float)
    iris = Xr[:, 0:4]                      # Lx,Ly,Rx,Ry
    pose = Xr[:, [4, 5, 7]]                # pitch,yaw,roll
    feats = [Xr]
    if mode in ("inter", "inter2"):
        inter = np.stack([iris[:, i] * pose[:, j] for i in range(4) for j in range(3)], axis=1)  # 12
        feats.append(inter)
    if mode == "inter2":
        feats.append(iris ** 2)           # 虹彩応答の非線形 4
    return np.concatenate(feats, axis=1)

def fit_predict(Xtr, Ytr, Xte, alpha=1e-3):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=800).fit(A, Ytr[:, i]).predict(B)
    return pr

def euc_cm(pred, tgt, sw, sh):
    dd = pred - tgt; return np.hypot(dd[:, 0] * sw, dd[:, 1] * sh)

MODES = ["base", "inter", "inter2"]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0]

# ================= 自分の実データ honest =================
sessions = []
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 60: continue
    Xs, Ys = [], []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        w, h = int(d["img_w"][k]), int(d["img_h"][k])
        try: f16 = rich_16d_from_lms(d["landmarks"][k], w, h)
        except Exception: f16 = None
        if f16 is None: continue
        Xs.append(np.asarray(f16, float)); Ys.append(np.asarray(t, float))
    if len(Xs) >= 60: sessions.append((np.array(Xs), np.array(Ys)))

rng = np.random.RandomState(0)
splits = []
for X, Y in sessions:
    groups = defaultdict(list)
    for i in range(len(X)): groups[(round(Y[i, 0], 1), round(Y[i, 1], 1))].append(i)
    gk = list(groups.keys())
    if len(gk) < 5: continue
    order = rng.permutation(len(gk)); cut = max(3, int(len(gk) * 0.7))
    trg = set(gk[j] for j in order[:cut]); teg = set(gk[j] for j in order[cut:])
    tr = [i for i in range(len(X)) if (round(Y[i, 0], 1), round(Y[i, 1], 1)) in trg]
    te = [i for i in range(len(X)) if (round(Y[i, 0], 1), round(Y[i, 1], 1)) in teg]
    if len(tr) < 30 or len(te) < 10: continue
    splits.append((X, Y, tr, te))

log("\n---\n## exp48: 姿勢条件づき交互作用特徴（iris×pose）")
log(f"\n### (1) 自分の実データ honest（{len(splits)}セッション, target-group-split, 基準16D=4.71cm）")
log(f"  {'mode':>7} | " + " | ".join(f"α={a:g}".rjust(9) for a in ALPHAS))
self_best = {}
for mode in MODES:
    row = []
    for a in ALPHAS:
        err = []
        for (X, Y, tr, te) in splits:
            Xa = augment(X, mode)
            err += list(euc_cm(fit_predict(Xa[tr], Y[tr], Xa[te], a), Y[te], SW, SH))
        m = float(np.median(err)); row.append(m)
    self_best[mode] = min(row)
    dim = augment(splits[0][0][:1], mode).shape[1]
    log(f"  {mode:>7}({dim:>2}D) | " + " | ".join(f"{v:8.3f}c" for v in row))
log(f"  → base最良={self_best['base']:.3f} / inter最良={self_best['inter']:.3f} / inter2最良={self_best['inter2']:.3f}")

# ================= MPII 15名 客観 =================
d = np.load(str(ROOT / "cache" / "mpii_16d_ck.npz"))
Xm, ym, pid = d["X"], d["y"], d["pid"]
pids = sorted(set(pid.tolist()))

def mpii_inner(mode, alpha):
    rr = []
    for p in pids:
        m = pid == p; Xp, yp = Xm[m], ym[m]
        if len(Xp) < 20: continue
        idx = rng.permutation(len(Xp)); cut = int(len(Xp) * 0.5)
        Xa = augment(Xp, mode)
        e = np.sqrt(np.sum((fit_predict(Xa[idx[:cut]], yp[idx[:cut]], Xa[idx[cut:]], alpha) - yp[idx[cut:]]) ** 2, axis=1))
        rr.append(np.median(e))
    return np.median(rr) * CM_W

def mpii_lopo(mode, alpha):
    rr = []
    Xa = augment(Xm, mode)
    for p in pids:
        te = pid == p; trm = ~te
        e = np.sqrt(np.sum((fit_predict(Xa[trm], ym[trm], Xa[te], alpha) - ym[te]) ** 2, axis=1))
        rr.append(np.median(e))
    return np.median(rr) * CM_W

log(f"\n### (2) MPII 15名 客観（個人内 基準4.18cm / person-indep 基準6.18cm, cmは幅30cm目安）")
log(f"  {'mode':>7} | {'個人内(α最良)':>14} | {'person-indep(α最良)':>18}")
for mode in MODES:
    inn = min(mpii_inner(mode, a) for a in ALPHAS)
    lop = min(mpii_lopo(mode, a) for a in ALPHAS)
    log(f"  {mode:>7} | {inn:12.3f}cm | {lop:16.3f}cm")

log("\n**判定**: interが自分honest かつ MPII客観の両方でbaseを下回れば採用候補。"
    "自分だけ改善/MPII悪化なら過適合。次はexp49(虹彩楕円)。")
