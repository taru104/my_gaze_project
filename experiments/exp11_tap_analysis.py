"""7時間研究フェーズ2 exp11: タップ点(DynamicCalibration)の分析。
「タップは信憑性高い正解、全姿勢改善に使えるか」を検証。全セッションCSVからタップを集め、
(1)総数と姿勢分布 (2)タップ時のraw予測の誤差(=タップを足せばどれだけ補正余地があるか)。
公知手法のみ。mainは触らない。
"""
import sys, glob, csv, math
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

ROOT = Path(__file__).parent.parent
SW, SH = 30.9, 17.4
REPORT = ROOT / "experiments" / "REPORT2.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def ff(r, k):
    try: return float(r.get(k, ""))
    except Exception: return None

taps = []  # (yaw, tx, ty, gx, gy, rx, ry)
for p in sorted(glob.glob(str(ROOT/"logs"/"session_*.csv"))):
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("tap_target_x"):
            taps.append((ff(r,"yaw_deg"), ff(r,"tap_target_x"), ff(r,"tap_target_y"),
                         ff(r,"gaze_x"), ff(r,"gaze_y"), ff(r,"raw_x"), ff(r,"raw_y")))

log(f"\n---\n## exp11: タップ点(信憑性の高い正解)の分析")
log(f"総タップ数: {len(taps)}  (全セッション合計)")
yaws = [abs(t[0]) for t in taps if t[0] is not None]
if yaws:
    log(f"タップ時 |yaw|: median={np.median(yaws):.1f}°  正面(<10)={sum(1 for y in yaws if y<10)}  "
        f"中(10-25)={sum(1 for y in yaws if 10<=y<25)}  横(>=25)={sum(1 for y in yaws if y>=25)}")
# 信憑性: タップ時の raw予測(=モデルが今出してた値) と タップ正解 の誤差
err = []
for yaw, tx, ty, gx, gy, rx, ry in taps:
    if None in (tx, ty, rx, ry): continue
    err.append((yaw, math.hypot((rx-tx)*SW, (ry-ty)*SH)))
if err:
    e = [x[1] for x in err]
    log(f"タップ時のraw予測誤差(モデルがどれだけズレてたか): median={np.median(e):.2f}cm max={max(e):.2f}cm")
    log(f"  → この誤差ぶんは、タップを正解として学習に足せば補正できる余地。")
    for lo, hi in [(0,10),(10,25),(25,90)]:
        vs = [d for y, d in err if y is not None and lo<=abs(y)<hi]
        if vs: log(f"  |yaw|{lo}-{hi}: n={len(vs)} median誤差={np.median(vs):.2f}cm")
log(f"\n### 考察(全姿勢へのタップ活用)")
log(f"- タップは現状少数だが『使うほど貯まる』。特に横向きでタップが貯まれば全姿勢キャリブになる。")
log(f"- 設計案: 使用中のタップを DynamicCalibration だけでなく『16D/7Dモデルの再fitにも供給』し、")
log(f"  横向きタップが増えたら横向き精度がオンラインで上がる仕組み(公知のonline/incremental回帰)。")
