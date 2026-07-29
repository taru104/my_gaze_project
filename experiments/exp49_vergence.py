"""exp49: 両眼視差・対称性特徴。左右虹彩の差(vergence)/虹彩径の左右非対称/EAR左右差など、
奥行き・縦視線の手掛かりを16Dに足す。全て16Dからの派生変換なのでMPII客観評価も可能。
自分honest(基準4.71)＋MPII15名(個人内4.18/person-indep6.18)。mainは触らない。
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

# 16D: [0Lx,1Ly,2Rx,3Ry, 4pitch,5yaw,6dist, 7roll, 8L_EAR,9R_EAR, 10L_ivert,11R_ivert, 12L_idiam,13R_idiam, 14L_asp,15R_asp]
def augment(X, mode):
    Xr = np.asarray(X, float)
    if mode == "base":
        return Xr
    verg = np.stack([
        Xr[:, 0] - Xr[:, 2],           # vergence_x (左右虹彩の水平差)
        Xr[:, 1] - Xr[:, 3],           # vergence_y
        Xr[:, 10] - Xr[:, 11],         # 虹彩縦位置の左右差
        Xr[:, 12] - Xr[:, 13],         # 虹彩径の左右差(yaw非対称)
        Xr[:, 8] - Xr[:, 9],           # EAR左右差
        Xr[:, 14] - Xr[:, 15],         # aspect左右差
    ], axis=1)
    meanf = np.stack([
        (Xr[:, 10] + Xr[:, 11]) / 2,   # 虹彩縦位置の平均(縦視線)
        (Xr[:, 8] + Xr[:, 9]) / 2,     # EAR平均
    ], axis=1)
    if mode == "verg":
        return np.concatenate([Xr, verg], axis=1)
    if mode == "verg+mean":
        return np.concatenate([Xr, verg, meanf], axis=1)
    raise ValueError(mode)

def fit_predict(Xtr, Ytr, Xte, alpha=1e-3):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=800).fit(A, Ytr[:, i]).predict(B)
    return pr

def euc_cm(pred, tgt, sw, sh):
    dd = pred - tgt; return np.hypot(dd[:, 0] * sw, dd[:, 1] * sh)

MODES = ["base", "verg", "verg+mean"]
ALPHAS = [1e-3, 1e-2, 1e-1]

# ---- 自分honest ----
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

log("\n---\n## exp49: 両眼視差・対称性特徴（vergence）")
log(f"\n### (1) 自分honest（{len(splits)}セッション, 基準16D=4.71cm）")
for mode in MODES:
    best = min(
        np.median([e for (X, Y, tr, te) in splits
                   for e in euc_cm(fit_predict(augment(X, mode)[tr], Y[tr], augment(X, mode)[te], a), Y[te], SW, SH)])
        for a in ALPHAS)
    dim = augment(splits[0][0][:1], mode).shape[1]
    log(f"  {mode:>10}({dim:>2}D) | {best:.3f}cm")

# ---- MPII客観 ----
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
    rr = []; Xa = augment(Xm, mode)
    for p in pids:
        te = pid == p; trm = ~te
        e = np.sqrt(np.sum((fit_predict(Xa[trm], ym[trm], Xa[te], alpha) - ym[te]) ** 2, axis=1))
        rr.append(np.median(e))
    return np.median(rr) * CM_W

log(f"\n### (2) MPII 15名 客観（個人内4.18 / person-indep6.18 基準）")
log(f"  {'mode':>10} | {'個人内':>10} | {'person-indep':>13}")
for mode in MODES:
    inn = min(mpii_inner(mode, a) for a in ALPHAS)
    lop = min(mpii_lopo(mode, a) for a in ALPHAS)
    log(f"  {mode:>10} | {inn:8.3f}cm | {lop:11.3f}cm")
log("\n**判定**: vergが両testでbaseを下回れば採用候補。片方だけなら過適合。")
