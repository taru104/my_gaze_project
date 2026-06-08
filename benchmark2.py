"""
フィッティング戦略の比較ベンチマーク。

問題の根本: 1フレームごとのX_feat/Y_featノイズ(σ≈0.01)で
1080サンプルを直接フィットすると、モデルは「ノイズの平均」に
フィットしようとするが、点ごとのサンプルばらつきがRidgeの
正則化に干渉して fit が悪化する。

解決策: キャリブ点ごとにメジアンを1点計算してから9点でフィット。
"""
import numpy as np
from collections import defaultdict
from calibration import AffineCalibration, PolyRidgeCalibration, CALIB_POINTS_9

SCREEN_CM_W = 30.9
SCREEN_CM_H = 17.4
N_SAMPLES   = 120
N_TRIALS    = 30
SIGMA_FEAT  = 0.010
BASE_PITCH  = -0.08

AX=1.25; BX=-0.20; CX=-0.15
AY=1.10; CY=0.40;  DY=0.60; EY=0.35

def screen_to_feat(tx, ty, pitch):
    dx, dy = tx-0.5, ty-0.5
    X_feat = AX*dx + BX*dy**2 + CX*dx*dy
    Y_feat = AY*dy + CY*pitch + DY*dy**2 + EY*dy*pitch
    return X_feat, Y_feat

def compute_mgae(preds, targets):
    def v3(p): return np.array([p[0]-0.5, p[1]-0.5, 1.0])
    angs = []
    for pr, tg in zip(preds, targets):
        v1, v2 = v3(pr), v3(tg)
        cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angs.append(float(np.degrees(np.arccos(np.clip(cos,-1+1e-7,1-1e-7)))))
    return float(np.mean(angs))

def run_strategy(strategy_name, calib_factory, all_samples_by_key, medians_by_key):
    """strategy: 'all_frames' or 'medians'"""
    if strategy_name == 'all_frames':
        af = calib_factory()
        preds, targets = [], []
        for (tx,ty), samples in all_samples_by_key.items():
            for X,Y,p,w in samples:
                af.add(X, Y, tx, ty, weight=w, pitch_rad=p)
        af.fit()
        for (tx,ty), samples in all_samples_by_key.items():
            for X,Y,p,w in samples:
                preds.append(af.predict(X,Y,p))
                targets.append([tx,ty])
        train_mgae = compute_mgae(preds, targets)

        # LOO on per-group median (fair LOO)
        loo_preds, loo_targets = [], []
        keys = list(medians_by_key.keys())
        for held in keys:
            laf = calib_factory()
            for (tx,ty), samples in all_samples_by_key.items():
                if (tx,ty) == held: continue
                for X,Y,p,w in samples:
                    laf.add(X, Y, tx, ty, weight=w, pitch_rad=p)
            raw_buf = laf._raw if hasattr(laf,'_raw') else laf._design
            if len(raw_buf) < 5: continue
            laf.fit()
            Xm,Ym,pm = medians_by_key[held]
            loo_preds.append(laf.predict(Xm,Ym,pm))
            loo_targets.append(list(held))
    else:  # 'medians'
        af = calib_factory()
        for (tx,ty),(X,Y,p) in medians_by_key.items():
            af.add(X, Y, tx, ty, weight=1.0, pitch_rad=p)
        af.fit()
        preds   = [af.predict(v[0],v[1],v[2]) for v in medians_by_key.values()]
        targets = [list(k) for k in medians_by_key.keys()]
        train_mgae = compute_mgae(preds, targets)

        # LOO on medians
        loo_preds, loo_targets = [], []
        keys = list(medians_by_key.keys())
        for held in keys:
            laf = calib_factory()
            for k,(X,Y,p) in medians_by_key.items():
                if k == held: continue
                laf.add(X, Y, k[0], k[1], weight=1.0, pitch_rad=p)
            raw_buf = laf._raw if hasattr(laf,'_raw') else laf._design
            if len(raw_buf) < 4: continue
            laf.fit()
            Xm,Ym,pm = medians_by_key[held]
            loo_preds.append(laf.predict(Xm,Ym,pm))
            loo_targets.append(list(held))

    loo_mgae = compute_mgae(loo_preds, loo_targets) if loo_preds else 999.0
    return train_mgae, loo_mgae

# ── 実験 ───────────────────────────────────────────────────────────────────
strategies = [
    ('Affine  + all_frames ', AffineCalibration,    'all_frames'),
    ('Affine  + medians    ', AffineCalibration,    'medians'),
    ('Poly    + all_frames ', PolyRidgeCalibration, 'all_frames'),
    ('Poly    + medians    ', PolyRidgeCalibration, 'medians'),
]

results = {s[0]: {'train':[], 'loo':[]} for s in strategies}

print('Running...')
for trial in range(N_TRIALS):
    rng = np.random.RandomState(trial)
    all_samples = defaultdict(list)
    medians     = {}

    for point in CALIB_POINTS_9:
        tx, ty = float(point[0]), float(point[1])
        key = (round(tx,4), round(ty,4))
        pitch_base = BASE_PITCH + rng.normal(0,0.02)
        Xs, Ys, Ps = [], [], []
        for i in range(N_SAMPLES):
            p_i = pitch_base + rng.normal(0,0.01)
            X0,Y0 = screen_to_feat(tx, ty, p_i)
            X_n = X0 + rng.normal(0, SIGMA_FEAT)
            Y_n = Y0 + rng.normal(0, SIGMA_FEAT)
            w   = min(float(i)/(N_SAMPLES*0.5), 1.0)
            all_samples[key].append((X_n, Y_n, p_i, w))
            Xs.append(X_n); Ys.append(Y_n); Ps.append(p_i)
        medians[key] = (float(np.median(Xs)), float(np.median(Ys)), float(np.median(Ps)))

    for label, factory, strategy in strategies:
        tm, lm = run_strategy(strategy, factory, all_samples, medians)
        results[label]['train'].append(tm)
        results[label]['loo'].append(lm)

sep = '='*65
print()
print(sep)
print('  Strategy Comparison (N=%d trials, sigma=%.3f)' % (N_TRIALS, SIGMA_FEAT))
print('  Target: LOO MGAE < 5 deg')
print(sep)
print('  %-26s  train MGAE   LOO MGAE' % 'Strategy')
print('  ' + '-'*55)
for label, _, _ in strategies:
    tm = np.array(results[label]['train'])
    lm = np.array(results[label]['loo'])
    flag = '<< TARGET' if lm.mean() < 5.0 else ''
    print('  %-26s  %5.2f±%.2f   %5.2f±%.2f  %s' % (
        label, tm.mean(), tm.std(), lm.mean(), lm.std(), flag))
print(sep)
