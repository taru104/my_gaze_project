"""exp50: クロスセッション検証(特徴いじりでない本質)。同一人物の別セッションで学習したモデルが
新セッションでどれだけ効くか。「データ量が効くのか」「軽いキャリブが必須か」を客観的に測る。
 (A) per-session honest: 各セッション内でtarget-group-split(基準4.71)
 (B) LOSO no-calib: 他17セッション学習→held1セッションをキャリブ無し予測(=別日にそのまま使えるか)
 (C) LOSO + few-shot: 他17 + held内K点キャリブ(=軽い再キャリブでどこまで回復)
16Dのみ・自分実データ。mainは触らない。
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
REPORT = ROOT / "experiments" / "REPORT6_beyond16d.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def fit_predict(Xtr, Ytr, Xte, alpha=1e-3):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=800).fit(A, Ytr[:, i]).predict(B)
    return pr
def euc(pred, tgt): dd = pred - tgt; return np.hypot(dd[:, 0] * SW, dd[:, 1] * SH)

# セッション別に16D+target読み込み
S = []
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
    if len(Xs) >= 60: S.append((np.array(Xs), np.array(Ys)))
log(f"\n---\n## exp50: クロスセッション検証（{len(S)}セッション, 16D）")

rng = np.random.RandomState(0)
# (A) per-session honest
A_err = []
for X, Y in S:
    groups = defaultdict(list)
    for i in range(len(X)): groups[(round(Y[i,0],1), round(Y[i,1],1))].append(i)
    gk = list(groups.keys())
    if len(gk) < 5: continue
    order = rng.permutation(len(gk)); cut = max(3, int(len(gk)*0.7))
    trg = set(gk[j] for j in order[:cut])
    tr = [i for i in range(len(X)) if (round(Y[i,0],1), round(Y[i,1],1)) in trg]
    te = [i for i in range(len(X)) if (round(Y[i,0],1), round(Y[i,1],1)) not in trg]
    if len(tr) < 30 or len(te) < 10: continue
    A_err += list(euc(fit_predict(X[tr], Y[tr], X[te]), Y[te]))
log(f"\n  (A) per-session honest（各セッション内キャリブ）  = {np.median(A_err):.3f}cm  (基準4.71)")

# (B) LOSO no-calib & (C) few-shot
B_err, C_err = {}, {}
Klist = [0, 3, 9, 20]  # 0=no-calib(B), >0=few-shot(C)
for K in Klist: C_err[K] = []
for h in range(len(S)):
    Xh, Yh = S[h]
    others = [S[j] for j in range(len(S)) if j != h]
    Xo = np.vstack([x for x, _ in others]); Yo = np.vstack([y for _, y in others])
    # held内をtarget-groupでcalib候補/評価に分割
    groups = defaultdict(list)
    for i in range(len(Xh)): groups[(round(Yh[i,0],1), round(Yh[i,1],1))].append(i)
    gk = list(groups.keys())
    if len(gk) < 5: continue
    order = rng.permutation(len(gk)); cut = max(3, int(len(gk)*0.7))
    calg = [gk[j] for j in order[:cut]]                     # キャリブに使える群
    teg = set(gk[j] for j in order[cut:])
    te = [i for i in range(len(Xh)) if (round(Yh[i,0],1), round(Yh[i,1],1)) in teg]
    if len(te) < 10: continue
    for K in Klist:
        if K == 0:
            pr = fit_predict(Xo, Yo, Xh[te])               # 純他セッション
        else:
            # held内からK群を1点ずつ選びキャリブに追加(=K点再キャリブ)
            useg = calg[:K]
            cidx = [groups[g][len(groups[g])//2] for g in useg]
            Xtr = np.vstack([Xo, Xh[cidx]]); Ytr = np.vstack([Yo, Yh[cidx]])
            pr = fit_predict(Xtr, Ytr, Xh[te])
        C_err[K] += list(euc(pr, Yh[te]))
log(f"\n  (B/C) LOSO: 他{len(S)-1}セッション学習 + held内K点キャリブ")
log(f"  {'K点':>5} | {'median cm':>10} | 意味")
meanings = {0:"キャリブ無し(別日そのまま)", 3:"3点だけ再キャリブ", 9:"9点(通常キャリブ相当)", 20:"20点(密)"}
for K in Klist:
    if C_err[K]:
        log(f"  {K:>5} | {np.median(C_err[K]):9.3f}cm | {meanings[K]}")
log("\n**狙いの読み**: (B)K=0が悪く→少Kで急改善なら『個人キャリブ必須・でも数点で足りる』を客観実証。"
    "K増でper-session(4.71)に近づけば『別日データの蓄積が効く=データ収集に価値』。頭打ちなら特徴の限界側。")
