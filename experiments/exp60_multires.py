"""exp60: 複数解像度アンサンブル。24x16と48x32のパッチを各PCA16で圧縮して両方使う(過学習抑えつつ情報増)。
過学習警戒: 各解像度をPCA16に絞ってから結合。person-indep主。mainは触らない。
"""
import sys
from pathlib import Path
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA

REPORT = ROOT / "experiments" / "REPORT6_beyond16d.md"
def log(s):
    print(s, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(s + "\n")
def err(pred, gt): return np.sqrt(np.sum((pred - gt) ** 2, axis=1)) * 30.0

d2 = np.load(str(ROOT / "cache" / "mpii_app2.npz"))   # 24x16 CLAHE = Xclahe
d3 = np.load(str(ROOT / "cache" / "mpii_app3.npz"))   # 48x32 CLAHE = Xhi
log("\n---\n## exp60: 複数解像度アンサンブル（24x16 + 48x32, 各PCA16）")
# 整合チェック
if len(d2["pid"]) != len(d3["pid"]) or not np.array_equal(d2["pid"], d3["pid"]) or not np.allclose(d2["y"], d3["y"], atol=1e-5):
    log(f"  ⚠️ キャッシュ非整合(app2={len(d2['pid'])}枚 / app3={len(d3['pid'])}枚)=サンプル対応が取れずアンサンブル不可。")
    log("  → 結合には両解像度を1passで抽出し直しが必要。今回は単一解像度48x32(exp59)が確定チャンピオンのまま。")
    sys.exit(0)

X16, Xlo, Xhi, y, pid = d3["X16"], d2["Xclahe"], d3["Xhi"], d3["y"], d3["pid"]
pids = sorted(set(pid.tolist()))
def predict(tr, te, mode):
    parts_tr, parts_te = [X16[tr]], [X16[te]]
    for src in ({"lo": [Xlo], "hi": [Xhi], "both": [Xlo, Xhi]}[mode]):
        pca = PCA(n_components=min(16, len(tr)-1)).fit(src[tr])
        parts_tr.append(pca.transform(src[tr])); parts_te.append(pca.transform(src[te]))
    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    pr = np.zeros((len(Xte), 2))
    for i in range(2):
        pr[:, i] = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=800).fit(A, y[tr][:, i]).predict(B)
    return pr
MODES = ["lo", "hi", "both"]
log(f"\n### person-indep（主, 基準16D=6.17 / hi単独=5.54）")
for m in MODES:
    e = [np.median(err(predict(np.where(pid!=p)[0], np.where(pid==p)[0], m), y[pid==p])) for p in pids]
    log(f"  {('24x16' if m=='lo' else '48x32' if m=='hi' else '両解像度'):>10} | {np.median(e):.3f}cm")
log(f"\n### 個人内 時間分割（副, 基準16D=5.20 / hi単独=4.46）")
for m in MODES:
    e = []
    for p in pids:
        idx = np.where(pid==p)[0]; cut=len(idx)//2
        e.append(np.median(err(predict(idx[:cut], idx[cut:], m), y[idx[cut:]])))
    log(f"  {('24x16' if m=='lo' else '48x32' if m=='hi' else '両解像度'):>10} | {np.median(e):.3f}cm")
log("\n**判定**: 両解像度がhi単独を下回れば=複数スケール情報が相補的。同等なら48x32単独で十分。")
