"""exp67: 「タップは"厳しい所"で起きる」を前提に、タップ適応を測り直す。

exp66 の設計ミス（ユーザ指摘）:
  - 「タップ」をセッション後半の**先頭から連続フレーム**で代用していた。
    実際のユーザは**うまく合っていない時にクリックする**ので、タップは難しい条件
    （キャリブ姿勢から離れた頭部姿勢・画面の隅）に偏る。
  - しかも**全フレーム平均**で評価したので、効くはずの厳しい領域が薄まって消えていた。

本実験で直す点:
  1. タップを**難しいフレームから優先的に**引く（実挙動の再現）。対照として一様抽出も測る。
  2. 評価を**難易度で層別**する（easy / hard）。厳しい所で効くなら、そこに出るはず。
  3. 2つの機構を分けて比較する:
       B) 残差シフト = DynamicCalibration.correct（exp66 が測ったもの）
       C) **タップ点をキャリブに加えて再学習**（exp15 で横向き-28%が出た方）

難易度の定義: キャリブ時の平均頭部姿勢からの |Δyaw|+|Δpitch|（大きいほど厳しい）。
リーク対策: 使用区間を**ターゲット位置ごとのグループ**に分け、タップに使った
グループはテストから丸ごと除外する（隣接フレームは強相関のため）。

main は一切変更しない。

Usage: .venv/Scripts/python.exe experiments/exp67_hard_taps.py
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
from calibration import DynamicCalibration
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT8_metrics.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

def euc(pred, tgt):
    d = pred - tgt
    return np.hypot(d[:, 0] * SW, d[:, 1] * SH)

def fit_predict(Xtr, Ytr, Xte):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3,
                                  max_iter=500).fit(A, Ytr[:, i]).predict(B)
    return pr

# ── 実ログ読み込み ──
sessions = []
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 150: continue
    pts = []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        try: f16 = rich_16d_from_lms(d["landmarks"][k], int(d["img_w"][k]), int(d["img_h"][k]))
        except Exception: f16 = None
        if f16 is None: continue
        f16 = np.asarray(f16, float)
        pts.append(dict(f16=f16, tgt=np.asarray(t, float),
                        head=np.array([f16[4], f16[5]], float)))
    if len(pts) >= 150: sessions.append(pts)

NTAPS = [0, 3, 5, 10, 20]
ARMS = ["A_無適応", "B_残差シフト", "C_再学習"]
# 結果格納: [抽出方法][腕][タップ数] -> 層別誤差リスト
res = {samp: {a: {n: {"all": [], "easy": [], "hard": []} for n in NTAPS} for a in ARMS}
       for samp in ["hard", "uniform"]}

for pts in sessions:
    half = len(pts) // 2
    cal, use = pts[:half], pts[half:]
    if len(cal) < 50 or len(use) < 80: continue

    Xc = np.array([p["f16"] for p in cal]); Yc = np.array([p["tgt"] for p in cal])
    ref_head = np.mean([p["head"] for p in cal], axis=0)

    # 使用区間を「ターゲット位置」でグループ化（同一点の連続フレームを1グループに）
    groups = {}
    for i, p in enumerate(use):
        key = (round(p["tgt"][0], 3), round(p["tgt"][1], 3))
        groups.setdefault(key, []).append(i)
    gkeys = list(groups.keys())
    if len(gkeys) < 8: continue

    # グループごとの難易度 = キャリブ姿勢からの頭部姿勢のズレ(中央値)
    gdiff = {k: float(np.median([np.abs(use[i]["head"] - ref_head).sum() for i in groups[k]]))
             for k in gkeys}
    order_hard = sorted(gkeys, key=lambda k: -gdiff[k])     # 厳しい順
    med_diff = np.median(list(gdiff.values()))

    # ★テストセットは N によらず固定する（exp67 初版の欠陥: N毎にテスト集合が変わり
    #   無適応(A)の値まで動いてしまい、列をまたいだ比較が成立していなかった）。
    #   グループを難易度順に交互配分し、片方をテスト、もう片方をタップ供給源に固定する。
    test_keys = [k for i, k in enumerate(order_hard) if i % 2 == 0]
    tap_pool_all = [k for i, k in enumerate(order_hard) if i % 2 == 1]
    if len(test_keys) < 4 or len(tap_pool_all) < 4: continue
    test_idx = [i for k in test_keys for i in groups[k]]
    hard_mask = np.array([gdiff[k] > med_diff for k in test_keys for _ in groups[k]])
    Xt = np.array([use[i]["f16"] for i in test_idx])
    Yt = np.array([use[i]["tgt"] for i in test_idx])
    pred_A = fit_predict(Xc, Yc, Xt)                    # 無適応はNに依存しないので1回だけ

    for samp in ["hard", "uniform"]:
        rs = np.random.RandomState(0)
        pool = (tap_pool_all if samp == "hard"
                else [tap_pool_all[i] for i in rs.permutation(len(tap_pool_all))])
        for n in NTAPS:
            if n > len(pool): continue
            tap_keys = pool[:n]
            # タップは各グループの代表1フレーム
            tap_pts = [use[groups[k][len(groups[k]) // 2]] for k in tap_keys]

            for arm in ARMS:
                if arm == "A_無適応" or n == 0:
                    pred = pred_A
                elif arm == "B_残差シフト":
                    base = pred_A
                    tp = fit_predict(Xc, Yc, np.array([p["f16"] for p in tap_pts]))
                    dyn = DynamicCalibration()
                    for j, p in enumerate(tap_pts):
                        dyn.add(p["tgt"].astype(np.float32), tp[j].astype(np.float32),
                                p["head"].astype(np.float32))
                    pred = np.array([dyn.correct(base[i].astype(np.float32),
                                                 use[test_idx[i]]["head"].astype(np.float32))
                                     for i in range(len(test_idx))])
                else:  # C_再学習
                    Xa = np.vstack([Xc, np.array([p["f16"] for p in tap_pts])])
                    Ya = np.vstack([Yc, np.array([p["tgt"] for p in tap_pts])])
                    pred = fit_predict(Xa, Ya, Xt)
                e = euc(pred, Yt)
                r = res[samp][arm][n]
                r["all"].append(float(np.median(e)))
                if hard_mask.any():  r["hard"].append(float(np.median(e[hard_mask])))
                if (~hard_mask).any(): r["easy"].append(float(np.median(e[~hard_mask])))

log(f"\n## 5. exp67: タップは「厳しい所」で起きる — 層別＋再学習で測り直す（{len(sessions)}セッション）\n")
log("exp66 の設計ミス(ユーザ指摘)を修正: ①タップを難しいフレームから優先抽出 "
    "②難易度で層別評価 ③残差シフトと「キャリブに加えて再学習」を分離。\n")

for samp, label in [("hard", "タップを**厳しいフレームから**引く（実挙動の再現）"),
                    ("uniform", "タップを一様に引く（対照）")]:
    log(f"### {label}")
    log("| 腕 | 層 | " + " | ".join(f"N={n}" for n in NTAPS) + " |")
    log("|---|---|" + "---:|" * len(NTAPS))
    for arm in ARMS:
        for layer in ["all", "hard", "easy"]:
            vals = []
            for n in NTAPS:
                v = res[samp][arm][n][layer]
                vals.append(f"{np.median(v):.2f}" if v else "—")
            log(f"| {arm} | {layer} | " + " | ".join(vals) + " |")
    log("")
    base_hard = np.median(res[samp]["A_無適応"][0]["hard"])
    for arm in ["B_残差シフト", "C_再学習"]:
        v = res[samp][arm][20]["hard"]
        if v:
            d = (np.median(v) - base_hard) / base_hard * 100
            log(f"- {arm} N=20 の **hard層**: {np.median(v):.2f}cm "
                f"(無適応 {base_hard:.2f}cm 比 {d:+.1f}%)")
    log("")
