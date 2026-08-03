"""exp66: 実機の `DynamicCalibration`（タップ適応）そのものを実データで測る。

背景: exp65 で「較正サンプルが少ないと補正が有害」を MPII で確認したが、あれは
**グローバルモデル+アフィン適応**の話で、実機に載っている `DynamicCalibration` の
アルゴリズムそのものを測ったわけではなかった。構造は似ている(少数サンプル・正則化なし)が、
**推論のまま main を触るのは危険**なので、実ログで直接測る。

測るもの: タップ数 N を 0,1,2,3,5,9,16 と変えたとき、`DynamicCalibration.correct()` が
予測を**良くするのか悪くするのか**。N=0(補正なし)を下回ったら「タップして悪化」が実在する。

手順(セッション毎):
  時系列で 前半=9点キャリブ相当(Huber学習) → 後半=使用。
  後半の先頭から N 個を「タップ」として DynamicCalibration に流し、残りで誤差を測る。
  ※ 実機と同じく `add(screen_gt, predicted, head_vec)` → `correct(predicted, head)` を使う。

main は一切変更しない。

Usage: .venv/Scripts/python.exe experiments/exp66_dynamic_taps.py
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

SW, SH = 30.9, 17.4                      # ユーザ実機の画面(既存実験と揃える)
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

# ── 実ログ読み込み（16D + 正解target + 頭部姿勢[pitch,yaw]） ──
sessions = []
for binp in sorted(glob.glob(str(ROOT / "logs" / "*_landmarks.bin"))):
    try: d = load_raw_landmarks(binp)
    except Exception: continue
    idx = np.where(d["has_target"])[0]
    if len(idx) < 120: continue
    pts = []
    for k in idx:
        t = d["target"][k]
        if np.isnan(t).any(): continue
        w, h = int(d["img_w"][k]), int(d["img_h"][k])
        try: f16 = rich_16d_from_lms(d["landmarks"][k], w, h)
        except Exception: f16 = None
        if f16 is None: continue
        f16 = np.asarray(f16, float)
        pts.append(dict(f16=f16, tgt=np.asarray(t, float),
                        head=np.array([f16[4], f16[5]], float)))   # [pitch, yaw]
    if len(pts) >= 120: sessions.append(pts)

NS = [0, 1, 2, 3, 5, 9, 16]
log(f"\n## 4. exp66: 実機 DynamicCalibration（タップ適応）をタップ数別に実測"
    f" — {len(sessions)}セッション（実機ログ）\n")
log("exp65(MPII)の知見が**実機のこのアルゴリズムにも当てはまるか**の直接検証。"
    "N=0(補正なし)を上回ったら『タップして悪化』が実在する。\n")

res = {n: [] for n in NS}
for pts in sessions:
    half = len(pts) // 2
    cal, use = pts[:half], pts[half:]
    ncut = len(use) // 2
    tap, test = use[:ncut], use[ncut:]
    if len(cal) < 40 or len(tap) < 16 or len(test) < 30: continue

    Xc = np.array([p["f16"] for p in cal]); Yc = np.array([p["tgt"] for p in cal])
    Xt = np.array([p["f16"] for p in test]); Yt = np.array([p["tgt"] for p in test])
    base = fit_predict(Xc, Yc, Xt)                       # 補正前の予測
    tap_pred = fit_predict(Xc, Yc, np.array([p["f16"] for p in tap]))

    for n in NS:
        dyn = DynamicCalibration()
        for j in range(n):                               # 実機と同じ流れでタップを登録
            dyn.add(tap[j]["tgt"].astype(np.float32),
                    tap_pred[j].astype(np.float32),
                    tap[j]["head"].astype(np.float32))
        if n == 0:
            corr = base
        else:
            corr = np.array([dyn.correct(base[i].astype(np.float32),
                                         test[i]["head"].astype(np.float32))
                             for i in range(len(test))])
        res[n].append(float(np.median(euc(corr, Yt))))

log("| タップ数 N | " + " | ".join(str(n) for n in NS) + " |")
log("|---|" + "---:|" * len(NS))
log("| 誤差 cm (全セッション中央値) | "
    + " | ".join(f"{np.median(res[n]):.2f}" for n in NS) + " |")
log("| 悪化したセッション数 | "
    + " | ".join(("—" if n == 0 else
                  f"{sum(1 for a, b in zip(res[n], res[0]) if a > b)}/{len(res[0])}")
                 for n in NS) + " |")

base0 = np.median(res[0])
worse = [n for n in NS if n > 0 and np.median(res[n]) > base0]
log("")
log(f"- 補正なし(N=0) = {base0:.2f}cm")
log(f"- N=0 より悪化するタップ数: **{worse if worse else 'なし'}**")
if worse:
    log(f"- → exp65 の懸念は**実機のアルゴリズムでも実在**。収縮/しきい値の導入に根拠あり。")
else:
    log(f"- → 実機の DynamicCalibration は少数タップでも悪化しない。"
        f"exp65 の懸念は**このアルゴリズムには当てはまらなかった**（推論の棄却）。")
