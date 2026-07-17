"""
姿勢ゲート・ハイブリッドキャリブレーション (研究成果の本番実装)

研究 (results/research_log.md) で確立した勝ち筋:
  - グローバルモデル: 大規模多姿勢データで「7D特徴→視線」を事前学習。頭部姿勢に頑健。
  - ローカルモデル  : ユーザ個別キャリブ(正面中心)。キャリブ姿勢付近で高精度だが横向きで崩壊。
  - 姿勢ゲート融合  : キャリブ姿勢に近ければローカル、遠ければグローバルへ滑らかに切替。

  w(m) = exp(-max(0, m - m_cal) / tau)      m=現在の頭部姿勢magnitude(deg)
  pred = w * local + (1 - w) * global

これにより「正面はローカルの高精度」「横向きはグローバルの頑健性」を両立する。
GazeCaptureベンチで横向き20-25°の誤差を現行9.6cm→3.8cm(60%改善)。

このクラスは特徴次元に非依存(add/fit/predictは渡された次元をそのまま扱う)。
Pitch/Yawは常に idx4/idx5 なので 7D でも 16D でも姿勢ゲートは同じに機能する。
global_model の入力次元と add/predict に渡す特徴の次元を一致させること。

特徴レイアウト:
  7D : [Lx, Ly, Rx, Ry, Pitch(rad), Yaw(rad), dist]
  16D: 上記7D + [roll, L_EAR, R_EAR, L_ivert, R_ivert, L_idiam, R_idiam,
                 L_aspect, R_aspect]   ← 現行ベスト。豊富特徴はグローバル学習で効く。

現行ベスト(2026-07-16): 16D rich hybrid = overall 3.339cm (7Dハイブリッド4.508比26%改善,
現行ローカル6.679比50%改善)。横向きほど改善大。詳細は results/research_log.md。
グローバルモデルは cache/global_mlp_16d.joblib (16D入力→cm)。抽出は extract_rich_features.extract_rich。

使い方:
    import joblib
    gm  = joblib.load("cache/global_mlp_16d.joblib")   # predict(X:(N,16)) -> cm(N,2)
    hyb = HybridCalibration(gm)
    for (feat16, target) in calib_frames:    # 個人キャリブ(正面中心)、feat16 = extract_rich(...)
        hyb.add(feat16, target)
    hyb.fit()
    pred = hyb.predict(feat16_now)           # 姿勢ゲートで融合
"""
import numpy as np
from typing import Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def pose_magnitude_deg(feat7: np.ndarray) -> np.ndarray:
    """7D特徴から頭部姿勢magnitude(deg)。Pitch=idx4, Yaw=idx5 (rad)。"""
    f = np.atleast_2d(feat7)
    return np.sqrt(np.degrees(f[:, 4])**2 + np.degrees(f[:, 5])**2)


class HybridCalibration:
    """
    グローバルモデル + ユーザ個別ローカルRidge を姿勢ゲートで融合。

    global_model: fit済みで .predict(X:(N,7)) -> (N,2) を持つ任意オブジェクト
                  (グローバル無しで純ローカルにしたい場合は None)
    tau         : ゲートの滑らかさ(deg)。小さいほど早くグローバルへ切替。研究では6。
    alpha       : ローカルRidgeの正則化。研究では5〜10。
    m_cal_pct   : キャリブ姿勢の代表値をとるパーセンタイル(90=キャリブ姿勢の外縁)。
    """

    def __init__(self, global_model=None, tau: float = 6.0,
                 alpha: float = 10.0, m_cal_pct: float = 90.0):
        self.global_model = global_model
        self.tau       = float(tau)
        self.alpha     = float(alpha)
        self.m_cal_pct = float(m_cal_pct)
        self._feats:   list = []
        self._targets: list = []
        self._scaler:  Optional[StandardScaler] = None
        self._ridge:   Optional[Ridge] = None
        self._m_cal:   float = 0.0
        self._fitted   = False

    def add(self, feat7: np.ndarray, target: np.ndarray) -> None:
        self._feats.append(np.asarray(feat7, dtype=np.float64).ravel())
        self._targets.append(np.asarray(target, dtype=np.float64).ravel())

    def fit(self) -> None:
        if len(self._feats) < 6:
            raise ValueError(f"キャリブサンプル不足: {len(self._feats)} < 6")
        X = np.array(self._feats)
        y = np.array(self._targets)
        self._scaler = StandardScaler().fit(X)
        self._ridge  = Ridge(alpha=self.alpha).fit(self._scaler.transform(X), y)
        self._m_cal  = float(np.percentile(pose_magnitude_deg(X), self.m_cal_pct))
        self._fitted = True

    def _local_predict(self, X7: np.ndarray) -> np.ndarray:
        return self._ridge.predict(self._scaler.transform(X7))

    def predict(self, feat7: np.ndarray) -> np.ndarray:
        """姿勢ゲート融合予測。(N,2) を返す。"""
        X7 = np.atleast_2d(np.asarray(feat7, dtype=np.float64))
        if not self._fitted:
            if self.global_model is not None:
                return self.global_model.predict(X7)
            return np.tile([0.5, 0.5], (len(X7), 1))

        local = self._local_predict(X7)
        if self.global_model is None:
            return local

        glob = self.global_model.predict(X7)
        m = pose_magnitude_deg(X7)
        if self.tau <= 0:
            w = (m <= self._m_cal).astype(float)
        else:
            w = np.exp(-np.maximum(0.0, m - self._m_cal) / self.tau)
        w = w[:, None]
        pred = w * local + (1.0 - w) * glob
        return pred if len(pred) > 1 else pred[0]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def reset(self) -> None:
        self._feats.clear(); self._targets.clear()
        self._scaler = self._ridge = None
        self._m_cal = 0.0; self._fitted = False
