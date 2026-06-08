"""
MGAE breakdownと Y軸問題の詳細分析 (ASCII only to avoid encoding issues)
"""
import os, glob, numpy as np, csv
from collections import defaultdict

LOG_DIR = r"C:\Users\hazib\my_gaze_project\logs"

def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def flt(v):
    try: return float(v)
    except: return None

files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))

# architecture split: 002704/003134 had low MGAE like old -> classify separately
OLD      = ["legacy","20260505","20260506_220539","20260506_220621",
            "20260508_002704","20260508_003134"]
NEW_EARLY = ["20260508_023235"]   # first session with high MGAE
NEW_LATE  = ["20260508_143943","20260508_145535","20260508_145907","20260508_150040"]

def classify(name):
    for tag in OLD:
        if tag in name: return "OLD"
    for tag in NEW_EARLY:
        if tag in name: return "NEW_EARLY"
    for tag in NEW_LATE:
        if tag in name: return "NEW_LATE"
    return "UNKNOWN"

groups = defaultdict(lambda: defaultdict(list))

for f in files:
    name = os.path.basename(f)
    grp = classify(name)
    rows = read_csv(f)
    for r in rows:
        cal = flt(r.get("calibrated",""))
        if cal != 1.0:
            continue
        gx  = flt(r.get("gaze_x",""))
        gy  = flt(r.get("gaze_y",""))
        rx  = flt(r.get("raw_x",""))
        ry  = flt(r.get("raw_y",""))
        pit = flt(r.get("pitch_deg",""))
        yaw = flt(r.get("yaw_deg",""))
        mgae= flt(r.get("calib_mgae_deg",""))
        if gx is not None: groups[grp]["gx"].append(gx)
        if gy is not None: groups[grp]["gy"].append(gy)
        if rx is not None and abs(rx) < 100: groups[grp]["rx"].append(rx)
        if ry is not None and abs(ry) < 100: groups[grp]["ry"].append(ry)
        if pit is not None and abs(pit) < 90: groups[grp]["pit"].append(pit)
        if yaw is not None and abs(yaw) < 90: groups[grp]["yaw_"].append(yaw)
        if mgae is not None: groups[grp]["mgae"].append(mgae)

def stat(arr, name):
    a = np.array(arr)
    if len(a) == 0:
        print(f"  {name}: NO DATA")
        return
    print(f"  {name}: n={len(a):5d}  mean={a.mean():+.4f}  std={a.std():.4f}"
          f"  P25={np.percentile(a,25):.3f}  P50={np.percentile(a,50):.3f}  P75={np.percentile(a,75):.3f}")

sep = "=" * 78

print(sep)
print("  [1] gaze X vs Y distribution per architecture group")
print(sep)
for grp in ["OLD", "NEW_EARLY", "NEW_LATE"]:
    g = groups[grp]
    print(f"\n--- {grp} ---")
    stat(g["gx"], "gaze_x")
    stat(g["gy"], "gaze_y")
    stat(g["rx"], "raw_x ")
    stat(g["ry"], "raw_y ")
    stat(g["mgae"], "MGAE  ")
    stat(g["pit"], "pitch ")
    stat(g["yaw_"], "yaw   ")

print()
print(sep)
print("  [2] Y-axis bias: gaze_y mean vs 0.5")
print("      Negative = model pushes gaze UP; Positive = pushes gaze DOWN")
print(sep)
for grp in ["OLD", "NEW_EARLY", "NEW_LATE"]:
    gy = np.array(groups[grp]["gy"])
    if len(gy) == 0: continue
    bias = gy.mean() - 0.5
    print(f"  {grp:12s}  gaze_y mean={gy.mean():.4f}  bias={bias:+.4f}  std={gy.std():.4f}")

print()
print(sep)
print("  [3] Pitch vs raw_y correlation and regression")
print("      A high correlation means head pitch is contaminating vertical gaze")
print(sep)
for grp in ["OLD", "NEW_EARLY", "NEW_LATE"]:
    pit = np.array(groups[grp]["pit"])
    ry  = np.array(groups[grp]["ry"])
    gy  = np.array(groups[grp]["gy"])
    n = min(len(pit), len(ry), len(gy))
    if n < 30: continue
    pit, ry, gy = pit[:n], ry[:n], gy[:n]
    corr_ry = float(np.corrcoef(pit, ry)[0,1]) if n > 1 else 0
    corr_gy = float(np.corrcoef(pit, gy)[0,1]) if n > 1 else 0
    # linear regression pitch -> raw_y
    slope, intercept = np.polyfit(pit, ry, 1)
    print(f"  {grp:12s}  pitch vs raw_y: r={corr_ry:+.3f}  slope={slope:+.4f} raw_y/deg")
    print(f"  {grp:12s}  pitch vs gaze_y: r={corr_gy:+.3f}")
    print(f"  {grp:12s}  pitch range: {pit.min():.1f} ~ {pit.max():.1f} deg")

print()
print(sep)
print("  [4] Y histogram: where does gaze_y cluster?")
print("      [0.0=top, 1.0=bottom of screen]")
print(sep)
for grp in ["OLD", "NEW_EARLY", "NEW_LATE"]:
    gy = np.array(groups[grp]["gy"])
    if len(gy) == 0: continue
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(gy, bins=bins)
    total = hist.sum()
    print(f"  [{grp}]  n={total}")
    labels = ["TOP  ","0.1  ","0.2  ","0.3  ","0.4  ","0.5  ","0.6  ","0.7  ","0.8  ","BOT  "]
    for i, (h, lab) in enumerate(zip(hist, labels)):
        pct = h / total * 100
        bar = "#" * int(pct * 1.5)
        print(f"    {lab} {bar:<50} {pct:4.1f}%")
    print()

print()
print(sep)
print("  [5] X histogram: where does gaze_x cluster?")
print(sep)
for grp in ["OLD", "NEW_EARLY", "NEW_LATE"]:
    gx = np.array(groups[grp]["gx"])
    if len(gx) == 0: continue
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(gx, bins=bins)
    total = hist.sum()
    print(f"  [{grp}]  n={total}")
    labels = ["LEFT ","0.1  ","0.2  ","0.3  ","0.4  ","0.5  ","0.6  ","0.7  ","0.8  ","RIGHT"]
    for i, (h, lab) in enumerate(zip(hist, labels)):
        pct = h / total * 100
        bar = "#" * int(pct * 1.5)
        print(f"    {lab} {bar:<50} {pct:4.1f}%")
    print()

print()
print(sep)
print("  [6] raw_y vs raw_x std comparison (instability)")
print(sep)
for grp in ["OLD","NEW_EARLY","NEW_LATE"]:
    rx = np.array(groups[grp]["rx"])
    ry = np.array(groups[grp]["ry"])
    if len(rx) == 0 or len(ry) == 0: continue
    print(f"  {grp:12s}  raw_x std={rx.std():.4f}  raw_y std={ry.std():.4f}  ratio_Y/X={ry.std()/max(rx.std(),1e-6):.3f}")

print()
print(sep)
print("  [7] MGAE jump analysis: old vs new calibration fit")
print(sep)
for grp in ["OLD","NEW_EARLY","NEW_LATE"]:
    m = np.array(groups[grp]["mgae"])
    if len(m) == 0: continue
    # get last mgae per session block (training error after finalize)
    print(f"  {grp:12s}  train_MGAE: mean={m.mean():.2f}  last={m[-1]:.2f} deg")
