"""
session_175045.csv の実キャリブデータを使ってオフラインで手法比較。
CSV に X_feat/Y_feat が入っているので実データで評価できる。
"""
import csv, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import AffineCalibration, PolyRidgeCalibration

CSV = r'C:\Users\hazib\my_gaze_project\logs\session_20260508_175045.csv'
SCREEN_CM_W = 30.9
SCREEN_CM_H = 17.4

# ── キャリブレーションフレームを読む ─────────────────────────────────────────
rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
calib_frames = [
    r for r in rows
    if r.get('calibrated', '') == '0'
    and r.get('calib_point_idx', '') != ''
    and r.get('X_feat', '') != ''
    and r.get('Y_feat', '') != ''
    and r.get('calib_target_x', '') != ''
]

def f(v): return float(v)

# ── 点ごとにグループ化 ──────────────────────────────────────────────────────
groups = defaultdict(list)
for r in calib_frames:
    tx = round(f(r['calib_target_x']), 4)
    ty = round(f(r['calib_target_y']), 4)
    X  = f(r['X_feat'])
    Y  = f(r['Y_feat'])
    p  = f(r['pitch_deg']) * np.pi / 180.0
    groups[(tx, ty)].append((X, Y, p))

print('Calibration points found: %d' % len(groups))
for key, samples in sorted(groups.items()):
    Xs = [s[0] for s in samples]
    Ys = [s[1] for s in samples]
    print('  (%.1f, %.1f): n=%d  X_feat med=%.4f std=%.4f  Y_feat med=%.4f std=%.4f' % (
        key[0], key[1], len(samples),
        np.median(Xs), np.std(Xs), np.median(Ys), np.std(Ys)))

# ── ヘルパ: MGAE ─────────────────────────────────────────────────────────────
def mgae(preds, targets):
    def v3(p): return np.array([p[0]-0.5, p[1]-0.5, 1.0])
    angs = []
    for pr, tg in zip(preds, targets):
        v1, v2 = v3(pr), v3(tg)
        cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angs.append(float(np.degrees(np.arccos(np.clip(cos,-1+1e-7,1-1e-7)))))
    return float(np.mean(angs))

def euc_cm(preds, targets):
    errs = np.array(preds) - np.array(targets)
    ex = float(np.mean(np.abs(errs[:,0]))) * SCREEN_CM_W
    ey = float(np.mean(np.abs(errs[:,1]))) * SCREEN_CM_H
    return ex, ey, float(np.sqrt(ex**2 + ey**2))

def loo(calib_factory, pts):
    """pts: {(tx,ty): (X_med, Y_med, p_med)}"""
    keys = list(pts.keys())
    preds, targets = [], []
    for held in keys:
        af = calib_factory()
        for k, (X,Y,p) in pts.items():
            if k == held: continue
            af.add(X, Y, k[0], k[1], weight=1.0, pitch_rad=p)
        raw_buf = af._raw if hasattr(af,'_raw') else af._design
        if len(raw_buf) < 4: continue
        af.fit()
        pred = af.predict(pts[held][0], pts[held][1], pts[held][2])
        preds.append(pred)
        targets.append(list(held))
    return preds, targets

sep = '='*65

# ══ 実験1: 全フレームでフィット (現状と同じ) ════════════════════════════════
print()
print(sep)
print('  [A] 全フレームでフィット (現在の実装と同等, 約%d samples)' % len(calib_frames))
print(sep)

for name, factory in [('Affine', AffineCalibration), ('PolyRidge', PolyRidgeCalibration)]:
    af = factory()
    all_pts = []
    for (tx,ty), samples in groups.items():
        for X,Y,p in samples:
            af.add(X, Y, tx, ty, weight=1.0, pitch_rad=p)
            all_pts.append(((X,Y,p), (tx,ty)))
    af.fit()
    preds   = [af.predict(s[0][0], s[0][1], s[0][2]) for s,_ in all_pts]
    targets = [list(s[1]) for _,s in all_pts]
    m = mgae(preds, targets)
    ex,ey,ec = euc_cm(preds, targets)
    print('  %s: train_MGAE=%.2f deg  Euc=%.2fcm (X=%.2f Y=%.2f)' % (name,m,ec,ex,ey))

# ══ 実験2: 点ごとメジアンでフィット ════════════════════════════════════════
print()
print(sep)
print('  [B] 点ごとメジアンでフィット (9点)')
print(sep)

medians = {}
for (tx,ty), samples in groups.items():
    medians[(tx,ty)] = (
        float(np.median([s[0] for s in samples])),
        float(np.median([s[1] for s in samples])),
        float(np.median([s[2] for s in samples])),
    )

for name, factory in [('Affine', AffineCalibration), ('PolyRidge', PolyRidgeCalibration)]:
    af = factory()
    for (tx,ty),(X,Y,p) in medians.items():
        af.add(X, Y, tx, ty, weight=1.0, pitch_rad=p)
    af.fit()
    preds   = [af.predict(v[0],v[1],v[2]) for v in medians.values()]
    targets = [list(k) for k in medians.keys()]
    m  = mgae(preds, targets)
    ex,ey,ec = euc_cm(preds, targets)
    print('  %s: train_MGAE=%.2f deg  Euc=%.2fcm (X=%.2f Y=%.2f)' % (name,m,ec,ex,ey))
    # LOO
    lp, lt = loo(factory, medians)
    lm  = mgae(lp, lt)
    lex,ley,lec = euc_cm(lp, lt)
    print('    LOO:       MGAE=%.2f deg  Euc=%.2fcm (X=%.2f Y=%.2f)' % (lm,lec,lex,ley))

# ══ 実験3: 外れ値除去 + メジアンでフィット ══════════════════════════════════
print()
print(sep)
print('  [C] 外れ値除去 (1.5σ clip) + メジアンでフィット')
print(sep)

medians_clean = {}
for (tx,ty), samples in groups.items():
    Xs = np.array([s[0] for s in samples])
    Ys = np.array([s[1] for s in samples])
    Ps = np.array([s[2] for s in samples])
    # 1.5σ以内のサンプルだけ使う
    mask = ((np.abs(Xs - np.median(Xs)) < 1.5*Xs.std()+1e-8) &
            (np.abs(Ys - np.median(Ys)) < 1.5*Ys.std()+1e-8))
    if mask.sum() < 5:
        mask = np.ones(len(Xs), dtype=bool)
    medians_clean[(tx,ty)] = (
        float(np.median(Xs[mask])),
        float(np.median(Ys[mask])),
        float(np.median(Ps[mask])),
    )
    print('  (%.1f,%.1f): %d/%d samples kept' % (tx,ty,mask.sum(),len(samples)))

print()
for name, factory in [('Affine', AffineCalibration), ('PolyRidge', PolyRidgeCalibration)]:
    af = factory()
    for (tx,ty),(X,Y,p) in medians_clean.items():
        af.add(X, Y, tx, ty, weight=1.0, pitch_rad=p)
    af.fit()
    preds   = [af.predict(v[0],v[1],v[2]) for v in medians_clean.values()]
    targets = [list(k) for k in medians_clean.keys()]
    m  = mgae(preds, targets)
    ex,ey,ec = euc_cm(preds, targets)
    print('  %s: train_MGAE=%.2f deg  Euc=%.2fcm (X=%.2f Y=%.2f)' % (name,m,ec,ex,ey))
    lp, lt = loo(factory, medians_clean)
    lm  = mgae(lp, lt)
    lex,ley,lec = euc_cm(lp, lt)
    print('    LOO:       MGAE=%.2f deg  Euc=%.2fcm (X=%.2f Y=%.2f)' % (lm,lec,lex,ley))

# ══ 実験4: X_feat/Y_feat の分布確認 ════════════════════════════════════════
print()
print(sep)
print('  [D] 特徴量の値域チェック (per-point median vs キャリブ中std)')
print(sep)
all_X_med = [v[0] for v in medians.values()]
all_Y_med = [v[1] for v in medians.values()]
print('  X_feat medians: min=%.4f max=%.4f range=%.4f' % (
    min(all_X_med), max(all_X_med), max(all_X_med)-min(all_X_med)))
print('  Y_feat medians: min=%.4f max=%.4f range=%.4f' % (
    min(all_Y_med), max(all_Y_med), max(all_Y_med)-min(all_Y_med)))
print()
print('  Ratio X_range/Y_range: %.2f (should be ~30.9/17.4=1.78 if features are well-scaled)' %
    ((max(all_X_med)-min(all_X_med)) / max(max(all_Y_med)-min(all_Y_med), 1e-6)))
