"""ユーザ実ログ(logs/*.csv)の実測精度と頭部姿勢品質を集計。
既存ログには生ランドマークが無く16D再現不可。現行パイプラインの実カメラ精度を確認する。"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import csv, glob
import numpy as np

ROOT = Path(__file__).parent.parent
files = sorted(glob.glob(str(ROOT / "logs" / "session_*.csv")))

def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        try:
            f = float(v)
            if np.isfinite(f): out.append(f)
        except: pass
    return np.array(out)

print(f"{'session':<28} {'rows':>6} {'loo_euc_cm(med/p90)':>20} {'loo_deg(med)':>12} "
      f"{'|pitch|max':>10} {'|yaw|max':>9} {'Xfeat外れ':>9}")
for f in files:
    rows = list(csv.DictReader(open(f, encoding="utf-8", errors="ignore")))
    name = Path(f).stem
    euc = col(rows, "loo_euc_cm"); deg = col(rows, "loo_mgae_deg")
    pit = col(rows, "pitch_deg");  yaw = col(rows, "yaw_deg")
    xf  = col(rows, "X_feat")
    euc = euc[euc > 0]; deg = deg[deg > 0]
    em = f"{np.median(euc):.2f}/{np.percentile(euc,90):.2f}" if len(euc) else "-"
    dm = f"{np.median(deg):.2f}" if len(deg) else "-"
    pm = f"{np.abs(pit).max():.0f}" if len(pit) else "-"
    ym = f"{np.abs(yaw).max():.0f}" if len(yaw) else "-"
    xo = f"{int((np.abs(xf)>3).sum())}" if len(xf) else "-"
    print(f"{name:<28} {len(rows):>6} {em:>20} {dm:>12} {pm:>10} {ym:>9} {xo:>9}")

# 全ログ合算のloo精度
alle, alld, allp = [], [], []
for f in files:
    rows = list(csv.DictReader(open(f, encoding="utf-8", errors="ignore")))
    e = col(rows, "loo_euc_cm"); alle.append(e[e>0])
    d = col(rows, "loo_mgae_deg"); alld.append(d[d>0])
    allp.append(np.abs(col(rows, "pitch_deg")))
alle = np.concatenate(alle) if alle else np.array([])
alld = np.concatenate(alld) if alld else np.array([])
allp = np.concatenate(allp) if allp else np.array([])
print("\n=== 全ログ合算(現行パイプラインの実カメラ実測) ===")
if len(alle):
    print(f"  loo_euc_cm : median={np.median(alle):.2f}cm  p90={np.percentile(alle,90):.2f}cm  n={len(alle)}")
if len(alld):
    print(f"  loo_mgae_deg: median={np.median(alld):.2f}°  p90={np.percentile(alld,90):.2f}°")
if len(allp):
    print(f"  |pitch|>40°の割合: {100*(allp>40).mean():.0f}%  (solvePnPフリップ疑い。非現実値)")
