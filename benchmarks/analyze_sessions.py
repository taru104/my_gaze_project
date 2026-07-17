"""
セッションログ分析スクリプト。
全 CSV を読み込み、旧アーキテクチャ vs 新アーキテクチャの比較を行う。
"""

import os
import glob
import numpy as np
import pandas as pd

LOG_DIR = r"C:\Users\hazib\my_gaze_project\logs"

# ─── 全CSVを読み込む ──────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))
sessions = []
for f in files:
    try:
        df = pd.read_csv(f)
        name = os.path.basename(f)
        df["_session"] = name
        sessions.append((name, df))
    except Exception as e:
        print(f"[SKIP] {f}: {e}")

print(f"読み込んだセッション数: {len(sessions)}")
print()

# ─── 各セッションのサマリ ──────────────────────────────────────────────────────
print("=" * 80)
print("  セッション別サマリ")
print("=" * 80)
print(f"{'セッション':<38} {'全行':>6} {'顔検出':>6} {'校正済':>6} {'MGAE':>8} {'RMSE':>8}")
print("-" * 80)

for name, df in sessions:
    n_total    = len(df)
    n_face     = df["face_detected"].sum() if "face_detected" in df.columns else 0
    calib_rows = df[df["calibrated"] == 1] if "calibrated" in df.columns else pd.DataFrame()
    n_calib    = len(calib_rows)
    mgae_vals  = calib_rows["calib_mgae_deg"].dropna() if "calib_mgae_deg" in calib_rows.columns else pd.Series()
    rmse_vals  = calib_rows["calib_rmse"].dropna()     if "calib_rmse"     in calib_rows.columns else pd.Series()
    mgae_str   = f"{mgae_vals.iloc[-1]:.2f}" if len(mgae_vals) > 0 else "N/A"
    rmse_str   = f"{rmse_vals.iloc[-1]:.4f}" if len(rmse_vals) > 0 else "N/A"
    print(f"{name:<38} {n_total:>6} {int(n_face):>6} {n_calib:>6} {mgae_str:>8} {rmse_str:>8}")

print()

# ─── 旧 vs 新 の判定 ──────────────────────────────────────────────────────────
# アーキテクチャ切り替えは session_20260506_220539 あたりから新規実装が入り始めた。
# 今回の「距離不変化」修正は 20260508_15xxxx 以降。
# まずはアーキテクチャ別にセッションを2分割して比較する。
# 旧 (7D+Ridge): 20260505_* / 20260506_*
# 新 (IrisDepth+Affine): 20260508_*

OLD_SESSIONS = [n for n, _ in sessions if "20260505" in n or "20260506" in n or "legacy" in n]
NEW_SESSIONS = [n for n, _ in sessions if "20260508" in n]

def collect_calibrated(session_names, all_sessions):
    frames = []
    for name, df in all_sessions:
        if name in session_names and "calibrated" in df.columns:
            cal = df[df["calibrated"] == 1].copy()
            cal["_session"] = name
            frames.append(cal)
    return pd.concat(frames) if frames else pd.DataFrame()

old_df = collect_calibrated(OLD_SESSIONS, sessions)
new_df = collect_calibrated(NEW_SESSIONS, sessions)

print("=" * 80)
print("  旧 (20260505/06, 7D+Ridge) vs 新 (20260508, IrisDepth+Affine)")
print("=" * 80)
print(f"  旧フレーム数: {len(old_df):,}   新フレーム数: {len(new_df):,}")
print()

def stats(series, label):
    if len(series) == 0:
        return f"{label}: N/A"
    return (f"{label}: mean={series.mean():.4f}  std={series.std():.4f}  "
            f"med={series.median():.4f}  [min={series.min():.4f}, max={series.max():.4f}]")

for tag, df in [("旧", old_df), ("新", new_df)]:
    print(f"--- {tag} ---")
    for col in ["gaze_x", "gaze_y", "raw_x", "raw_y"]:
        if col in df.columns:
            print("  " + stats(df[col].dropna(), col))
    print()

# ─── X/Y の分散比較（上下vs左右の精度差） ──────────────────────────────────────
print("=" * 80)
print("  X (左右) vs Y (上下) の分散・レンジ比較")
print("=" * 80)

for tag, df in [("旧", old_df), ("新", new_df)]:
    if df.empty:
        continue
    gx = df["gaze_x"].dropna()
    gy = df["gaze_y"].dropna()
    rx = df["raw_x"].dropna()
    ry = df["raw_y"].dropna()
    print(f"  [{tag}] gaze  X range={gx.max()-gx.min():.4f}  Y range={gy.max()-gy.min():.4f}")
    print(f"  [{tag}] gaze  X std  ={gx.std():.4f}           Y std  ={gy.std():.4f}")
    print(f"  [{tag}] raw   X range={rx.max()-rx.min():.4f}  Y range={ry.max()-ry.min():.4f}")
    print(f"  [{tag}] raw   X std  ={rx.std():.4f}           Y std  ={ry.std():.4f}")
    print()

# ─── キャリブレーション精度（MGAE）の推移 ─────────────────────────────────────
print("=" * 80)
print("  キャリブレーション精度 (MGAE) の推移")
print("=" * 80)
print(f"  {'セッション':<38} MGAE(deg)  RMSE")
print(f"  {'-'*60}")
for name, df in sessions:
    if "calibrated" not in df.columns:
        continue
    cal = df[df["calibrated"] == 1]
    mgae = cal["calib_mgae_deg"].dropna()
    rmse = cal["calib_rmse"].dropna()
    if len(mgae) == 0:
        continue
    arch = "旧" if ("20260505" in name or "20260506" in name or "legacy" in name) else "新"
    print(f"  [{arch}] {name:<35} {mgae.iloc[-1]:>8.2f}  {rmse.iloc[-1]:>8.4f}")
print()

# ─── Y 座標のバイアス（上下のズレ） ──────────────────────────────────────────
print("=" * 80)
print("  Y (上下) バイアス分析")
print("  ※ 0.5 が理想中央。大きくズレているほどキャリブ後も上下にオフセットがある")
print("=" * 80)

for tag, df in [("旧", old_df), ("新", new_df)]:
    if df.empty:
        continue
    gy = df["gaze_y"].dropna()
    print(f"  [{tag}] gaze_y 平均: {gy.mean():.4f}  (0.5からの乖離: {gy.mean()-0.5:+.4f})")
    # ヒストグラム（ASCII）
    hist, edges = np.histogram(gy, bins=10, range=(0.0, 1.0))
    print(f"  [{tag}] gaze_y 分布 [0.0 → 1.0]:")
    max_h = max(hist) if max(hist) > 0 else 1
    for i, (h, e) in enumerate(zip(hist, edges[:-1])):
        bar = "#" * int(h / max_h * 30)
        print(f"         {e:.1f}-{edges[i+1]:.1f}  {bar:<30} {h}")
    print()

# ─── pitch_deg と gaze_y の相関（上下精度の原因探索） ─────────────────────────
print("=" * 80)
print("  pitch_deg vs gaze_y の相関（上下精度の原因探索）")
print("  ※ 相関が高い = 頭部の上下傾きが視線Y推定に影響している")
print("=" * 80)
for tag, df in [("旧", old_df), ("新", new_df)]:
    if df.empty or "pitch_deg" not in df.columns:
        continue
    sub = df[["pitch_deg", "gaze_y", "raw_y"]].dropna()
    if len(sub) < 10:
        continue
    corr_gy  = sub["pitch_deg"].corr(sub["gaze_y"])
    corr_ry  = sub["pitch_deg"].corr(sub["raw_y"])
    print(f"  [{tag}] pitch vs gaze_y: r={corr_gy:+.3f}   pitch vs raw_y: r={corr_ry:+.3f}")
    # pitch 範囲
    print(f"  [{tag}] pitch_deg 範囲: {sub['pitch_deg'].min():.1f} ~ {sub['pitch_deg'].max():.1f} deg")
print()

# ─── raw_y の四分位 (上下の生推定値がどこに集まっているか) ────────────────────
print("=" * 80)
print("  raw_y / gaze_y 四分位 (上下の推定値分布)")
print("=" * 80)
for tag, df in [("旧", old_df), ("新", new_df)]:
    if df.empty:
        continue
    for col in ["raw_y", "gaze_y"]:
        if col not in df.columns:
            continue
        v = df[col].dropna()
        q = np.percentile(v, [10, 25, 50, 75, 90])
        print(f"  [{tag}] {col}: P10={q[0]:.3f}  P25={q[1]:.3f}  P50={q[2]:.3f}  P75={q[3]:.3f}  P90={q[4]:.3f}")
    print()

print("=" * 80)
print("  完了")
print("=" * 80)
