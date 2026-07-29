"""exp51: データ量スケーリング。学習セッション数nを1→16に増やすと、新セッション(held, キャリブ無し)の
誤差が下がるか。「別日データの蓄積が効く=データ収集(友人テスト等)に価値がある」を定量化する。
評価はexp50同様の内挿的LOSO(同じ9点グリッドを他日で見た位置)。フェアさの注意はexp50参照。16Dのみ。mainは触らない。
"""
import sys, glob
from pathlib import Path
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
N = len(S)
log(f"\n---\n## exp51: データ量スケーリング（{N}セッション, 16D, キャリブ無しLOSO）")

rng = np.random.RandomState(0)
NS = [n for n in [1, 2, 4, 8, 12, N - 1] if n <= N - 1]
NS = sorted(set(NS))
REPEAT = 4  # (held, n)ごとに他セッションの選び方をREPEAT回サンプルして平均
log(f"\n  学習セッション数n → 新セッション(held)キャリブ無し median cm")
log(f"  {'n':>4} | {'median cm':>10} | {'改善(vs n=1)':>12}")
curve = {}
for n in NS:
    errs = []
    for h in range(N):
        pool = [j for j in range(N) if j != h]
        Xh, Yh = S[h]
        for r in range(REPEAT):
            sel = rng.choice(pool, size=n, replace=False)
            Xo = np.vstack([S[j][0] for j in sel]); Yo = np.vstack([S[j][1] for j in sel])
            errs += list(euc(fit_predict(Xo, Yo, Xh), Yh))
    curve[n] = float(np.median(errs))
base1 = curve[NS[0]]
for n in NS:
    imp = (base1 - curve[n]) / base1 * 100
    log(f"  {n:>4} | {curve[n]:9.3f}cm | {imp:+10.1f}%")
log(f"\n**読み**: nが増えて誤差が単調に下がるなら『個人データの蓄積が効く』を定量実証"
    f"=友人テスト/継続録画でデータを貯める戦略に客観的裏付け。頭打ちなら蓄積の効果も限界。")
