"""
全手法のノイズ耐性比較。目標: 実ノイズ域(sigma=0.08-0.12)でLOO < 5deg。
"""
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))
from calibration import AffineCalibration, PolyRidgeCalibration, TargetedPolyCalibration, CALIB_POINTS_9

SCREEN_CM_W = 30.9; SCREEN_CM_H = 17.4
N_SAMPLES = 120; N_TRIALS = 30; BASE_PITCH = -0.08

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

def run(factory, all_s, meds):
    # train on all frames
    af = factory()
    for (tx,ty),samps in all_s.items():
        for X,Y,p,w in samps: af.add(X,Y,tx,ty,weight=w,pitch_rad=p)
    af.fit()
    # LOO: train on 8 sets of all-frames, eval at held-out median
    keys = list(meds.keys())
    lp, lt = [], []
    for held in keys:
        laf = factory()
        for (tx,ty),samps in all_s.items():
            if (tx,ty)==held: continue
            for X,Y,p,w in samps: laf.add(X,Y,tx,ty,weight=w,pitch_rad=p)
        raw_buf = laf._raw if hasattr(laf,'_raw') else laf._design
        if len(raw_buf)<5: continue
        laf.fit()
        Xm,Ym,pm = meds[held]
        lp.append(laf.predict(Xm,Ym,pm)); lt.append(list(held))
    train_m = mgae(
        [af.predict(X,Y,p) for (_,__),samps in all_s.items() for X,Y,p,w in samps],
        [[tx,ty]            for (tx,ty),samps in all_s.items() for X,Y,p,w in samps]
    )
    return train_m, mgae(lp,lt) if lp else 999.0

factories = [
    ('Affine       ', AffineCalibration),
    ('TargetedPoly ', TargetedPolyCalibration),
    ('PolyRidge    ', PolyRidgeCalibration),
]
sigmas = [0.05, 0.08, 0.10, 0.12, 0.15]

sep = '='*72
print()
print(sep)
print('  LOO MGAE by noise level  (* = under 5 deg target)')
print('  Estimated real system sigma: 0.08-0.12')
print(sep)
print('  %-14s  %7s  %7s  %7s  %7s  %7s' % ('Method', 's=.05', 's=.08', 's=.10', 's=.12', 's=.15'))
print('  '+'-'*65)

for fname, factory in factories:
    row_train, row_loo = [], []
    for sigma in sigmas:
        trains, loos = [], []
        for trial in range(N_TRIALS):
            rng = np.random.RandomState(trial*100+int(sigma*1000))
            all_s = defaultdict(list); meds = {}
            for pt in CALIB_POINTS_9:
                tx,ty = float(pt[0]),float(pt[1])
                key   = (round(tx,4),round(ty,4))
                pb    = BASE_PITCH + rng.normal(0,0.02)
                Xs,Ys,Ps = [],[],[]
                for i in range(N_SAMPLES):
                    p_i = pb+rng.normal(0,0.01)
                    X0,Y0 = screen_to_feat(tx,ty,p_i)
                    Xn = X0+rng.normal(0,sigma); Yn = Y0+rng.normal(0,sigma)
                    w  = min(float(i)/(N_SAMPLES*0.5),1.0)
                    all_s[key].append((Xn,Yn,p_i,w))
                    Xs.append(Xn); Ys.append(Yn); Ps.append(p_i)
                meds[key] = (float(np.median(Xs)),float(np.median(Ys)),float(np.median(Ps)))
            tm,lm = run(factory, all_s, meds)
            trains.append(tm); loos.append(lm)
        row_loo.append(float(np.mean(loos)))
        row_train.append(float(np.mean(trains)))
    print('  %-14s LOO  ' % fname, end='')
    for v in row_loo:
        m = '*' if v<5.0 else ' '
        print(' %5.1f%s ' % (v,m), end='')
    print()
    print('  %-14s train' % '', end='')
    for v in row_train: print(' %5.1f  ' % v, end='')
    print()
    print()
print(sep)
