"""
ノイズレベルを変えて各戦略のロバスト性を比較。
実データのtrain_MGAE=14-15°から逆算すると実効ノイズσ≈0.05-0.15。
"""
import numpy as np
from collections import defaultdict
from calibration import AffineCalibration, PolyRidgeCalibration, CALIB_POINTS_9

SCREEN_CM_W = 30.9; SCREEN_CM_H = 17.4
N_SAMPLES = 120; N_TRIALS = 20
BASE_PITCH = -0.08

AX=1.25; BX=-0.20; CX=-0.15
AY=1.10; CY=0.40;  DY=0.60; EY=0.35

def screen_to_feat(tx, ty, pitch):
    dx, dy = tx-0.5, ty-0.5
    return (AX*dx + BX*dy**2 + CX*dx*dy,
            AY*dy + CY*pitch + DY*dy**2 + EY*dy*pitch)

def mgae(preds, targets):
    def v3(p): return np.array([p[0]-0.5, p[1]-0.5, 1.0])
    angs = []
    for pr, tg in zip(preds, targets):
        v1, v2 = v3(pr), v3(tg)
        cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
        angs.append(float(np.degrees(np.arccos(np.clip(cos,-1+1e-7,1-1e-7)))))
    return float(np.mean(angs))

def run_one(factory, all_s, meds):
    # Train on all frames
    af = factory()
    for (tx,ty), samps in all_s.items():
        for X,Y,p,w in samps: af.add(X,Y,tx,ty,weight=w,pitch_rad=p)
    af.fit()
    # LOO on medians
    keys = list(meds.keys())
    loo_p, loo_t = [], []
    for held in keys:
        laf = factory()
        for k,(X,Y,p) in meds.items():
            if k==held: continue
            # train LOO model on ALL frames except held point
            for X2,Y2,p2,w2 in all_s[k]: laf.add(X2,Y2,k[0],k[1],weight=w2,pitch_rad=p2)
        raw_buf = laf._raw if hasattr(laf,'_raw') else laf._design
        if len(raw_buf)<5: continue
        laf.fit()
        Xm,Ym,pm = meds[held]
        loo_p.append(laf.predict(Xm,Ym,pm))
        loo_t.append(list(held))
    train_m = mgae(
        [af.predict(X,Y,p) for (tx,ty),samps in all_s.items() for X,Y,p,w in samps],
        [[tx,ty]           for (tx,ty),samps in all_s.items() for X,Y,p,w in samps]
    )
    loo_m = mgae(loo_p, loo_t) if loo_p else 999
    return train_m, loo_m

sigmas = [0.01, 0.03, 0.05, 0.08, 0.12, 0.18]
factories = [('Affine   ', AffineCalibration), ('PolyRidge', PolyRidgeCalibration)]

sep = '='*75
print()
print(sep)
print('  LOO MGAE vs noise level  (target: < 5 deg, marked with *)')
print(sep)
print('  %-10s' % 'sigma', end='')
for s in sigmas: print('  σ=%.2f' % s, end='')
print()
print('  '+'-'*65)

for fname, factory in factories:
    train_row = []
    loo_row   = []
    for sigma in sigmas:
        trains, loos = [], []
        for trial in range(N_TRIALS):
            rng = np.random.RandomState(trial*100 + int(sigma*1000))
            all_s = defaultdict(list)
            meds  = {}
            for pt in CALIB_POINTS_9:
                tx,ty = float(pt[0]), float(pt[1])
                key   = (round(tx,4), round(ty,4))
                pb    = BASE_PITCH + rng.normal(0,0.02)
                Xs,Ys,Ps = [],[],[]
                for i in range(N_SAMPLES):
                    p_i   = pb + rng.normal(0,0.01)
                    X0,Y0 = screen_to_feat(tx,ty,p_i)
                    Xn    = X0 + rng.normal(0,sigma)
                    Yn    = Y0 + rng.normal(0,sigma)
                    w     = min(float(i)/(N_SAMPLES*0.5),1.0)
                    all_s[key].append((Xn,Yn,p_i,w))
                    Xs.append(Xn); Ys.append(Yn); Ps.append(p_i)
                meds[key] = (float(np.median(Xs)),float(np.median(Ys)),float(np.median(Ps)))
            tm, lm = run_one(factory, all_s, meds)
            trains.append(tm); loos.append(lm)
        loo_row.append(float(np.mean(loos)))
        train_row.append(float(np.mean(trains)))

    print('  %-10s train:' % fname, end='')
    for v in train_row: print(' %5.1f°' % v, end='')
    print()
    print('  %-10s LOO  :' % fname, end='')
    for v in loo_row:
        mark = '*' if v < 5.0 else ' '
        print(' %4.1f°%s' % (v, mark), end='')
    print()
    print()

print(sep)
print('  NOTE: Real system train_MGAE=14-15deg => σ_effective ≈ 0.08-0.12')
print(sep)
