"""exp47: GazeCapture大規模(16D特徴26万フレーム)で16Dのperson-independent精度を測る。
朝報告の材料: 16Dはキャリブなしglobalでは何cmか → あなたの個人キャリブ(2.2cm)がどれだけ効いてるかを定量化。
標準split(0=train,2=test, 別人)。cm誤差。mainは触らない。
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

REPORT = ROOT / "experiments" / "REPORT5_sota_transfer.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")

d = np.load(str(ROOT / "cache" / "rich_features_cache.npz"))
X, ycm, sp = d["X"], d["y_cm"], d["split_code"]
rng = np.random.RandomState(0)
tr = np.where(sp == 0)[0]; te = np.where(sp == 2)[0]
tr = rng.choice(tr, min(40000, len(tr)), replace=False)
te = rng.choice(te, min(15000, len(te)), replace=False)
sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
pred = np.zeros((len(te), 2))
for i in range(2):
    pred[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500).fit(A, ycm[tr][:, i]).predict(B)
e = np.sqrt(((pred - ycm[te]) ** 2).sum(1))
log("\n---\n## exp47: GazeCapture 16D person-independent（キャリブなしglobal, 大規模26万）")
log(f"  train={len(tr)}(別人), test={len(te)}フレーム")
log(f"  ★16D person-indep(キャリブなし) = median {np.median(e):.2f}cm / mean {e.mean():.2f}cm")
log(f"  (参考: あなたのlogsで個人キャリブありの16D=実機2.2cm)")
log(f"  → person-indep(誰でも,キャリブ無)は{np.median(e):.1f}cm。個人キャリブで2.2cmまで下がる=個人キャリブの威力を定量化。")
log(f"  ※GazeCaptureはモバイル(cm範囲±20-25)でPC画面と条件が違うので絶対値でなく『キャリブ有無の差』が要点。")
