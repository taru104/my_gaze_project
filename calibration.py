"""
キャリブレーションモジュール。
- AffineCalibration  : 2×3アフィン変換 (lstsq) -- [X_mm, Y_mm, 1] → [x_norm, y_norm]
- DynamicCalibration : タップ履歴ベースの動的自動補正
- CalibrationPipeline: 9点キャリブレーション + 動的補正の統合
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── 9点キャリブレーション座標（正規化 [0,1]）────────────────────────────────
CALIB_POINTS_9 = np.array([
    [0.5, 0.5],   # 中央
    [0.1, 0.1],   # 左上
    [0.9, 0.1],   # 右上
    [0.1, 0.9],   # 左下
    [0.9, 0.9],   # 右下
    [0.5, 0.1],   # 上中央
    [0.1, 0.5],   # 左中央
    [0.9, 0.5],   # 右中央
    [0.5, 0.9],   # 下中央
], dtype=np.float32)

CALIB_POINTS_5 = np.array([
    [0.5, 0.5],
    [0.1, 0.1],
    [0.9, 0.1],
    [0.1, 0.9],
    [0.9, 0.9],
], dtype=np.float32)


class AffineCalibration:
    """
    重み付き最小二乗法による 3×2 アフィン変換。
    入力: [X_feat, Y_feat, pitch_rad]  出力: [x_norm, y_norm]
    設計行列 (N,4) = [X_feat, Y_feat, pitch_rad, 1] を lstsq で解く。
    pitch_rad を加えることで頭部仰俯角による Y 推定汚染を補正する。
    """

    def __init__(self):
        self._A:       Optional[np.ndarray] = None  # (4, 2)
        self._design:  List[List[float]]    = []
        self._targets: List[List[float]]    = []
        self._weights: List[float]          = []

    def add(self, X_mm: float, Y_mm: float,
            x_norm: float, y_norm: float,
            weight: float = 1.0,
            pitch_rad: float = 0.0) -> None:
        self._design.append([X_mm, Y_mm, pitch_rad, 1.0])
        self._targets.append([x_norm, y_norm])
        self._weights.append(max(0.0, weight))

    def fit(self) -> None:
        D = np.array(self._design,  dtype=np.float64)   # (N, 4): [X, Y, pitch, 1]
        T = np.array(self._targets, dtype=np.float64)   # (N, 2)
        W = np.sqrt(np.array(self._weights, dtype=np.float64))[:, None]

        Dw = D * W
        Tw = T * W

        # X出力: pitch を含まない [X_feat, Y_feat, 1] で解く
        Ax, _, _, _ = np.linalg.lstsq(Dw[:, [0, 1, 3]], Tw[:, 0:1], rcond=None)  # (3,1)

        # Y出力: pitch を含む [X_feat, Y_feat, pitch_rad, 1] で解く
        Ay, _, _, _ = np.linalg.lstsq(Dw,               Tw[:, 1:2], rcond=None)  # (4,1)

        # (4, 2) 行列に組み立て。X列の pitch 係数(index 2)は 0 のまま
        self._A = np.zeros((4, 2), dtype=np.float64)
        self._A[[0, 1, 3], 0] = Ax[:, 0]   # X: X_feat, Y_feat, bias
        self._A[:, 1]         = Ay[:, 0]   # Y: X_feat, Y_feat, pitch, bias

    def predict(self, X_mm: float, Y_mm: float, pitch_rad: float = 0.0) -> np.ndarray:
        if self._A is None:
            return np.array([0.5, 0.5], dtype=np.float32)
        return (np.array([X_mm, Y_mm, pitch_rad, 1.0], dtype=np.float64) @ self._A).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._A is not None

    def reset(self) -> None:
        self._A = None
        self._design.clear()
        self._targets.clear()
        self._weights.clear()


class TargetedPolyCalibration:
    """
    X方向: Ridge([X_feat, Y_feat])                         → 2特徴+intercept = 3パラメータ
    Y方向: Ridge([X_feat, Y_feat, pitch, Y_feat^2, Y*pitch])→ 5特徴+intercept = 6パラメータ

    根拠:
      X は水平注視に対してほぼ線形。Y^2 や X*Y は不要。
      Y にのみ非線形項が必要: まぶた遮蔽→Y^2、r=0.37のpitch汚染→Y*pitch。
      LOO(8点訓練)でもX:3、Y:6と十分過決定。PolyRidge(10パラメータ)より安定。
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._ridge_x  = Ridge(alpha=alpha, fit_intercept=True)
        self._ridge_y  = Ridge(alpha=alpha, fit_intercept=True)
        self._is_fitted = False
        self._raw:     List[List[float]] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def _fx(self, X: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.column_stack([X, Y])

    def _fy(self, X: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.column_stack([X, Y, p, Y**2, Y * p])

    def add(self, X_mm: float, Y_mm: float,
            x_norm: float, y_norm: float,
            weight: float = 1.0,
            pitch_rad: float = 0.0) -> None:
        self._raw.append([X_mm, Y_mm, pitch_rad])
        self._targets.append([x_norm, y_norm])
        self._weights.append(max(0.0, weight))

    def fit(self) -> None:
        raw = np.array(self._raw,     dtype=np.float64)
        T   = np.array(self._targets, dtype=np.float64)
        W   = np.array(self._weights, dtype=np.float64)
        X, Y, p = raw[:, 0], raw[:, 1], raw[:, 2]
        Fx = self._scaler_x.fit_transform(self._fx(X, Y, p))
        Fy = self._scaler_y.fit_transform(self._fy(X, Y, p))
        self._ridge_x.fit(Fx, T[:, 0], sample_weight=W)
        self._ridge_y.fit(Fy, T[:, 1], sample_weight=W)
        self._is_fitted = True

    def predict(self, X_mm: float, Y_mm: float, pitch_rad: float = 0.0) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        X = np.array([[X_mm]]); Y = np.array([[Y_mm]]); p = np.array([[pitch_rad]])
        Fx = self._scaler_x.transform(self._fx(X, Y, p))
        Fy = self._scaler_y.transform(self._fy(X, Y, p))
        return np.array([float(self._ridge_x.predict(Fx)[0]),
                         float(self._ridge_y.predict(Fy)[0])], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._is_fitted = False
        self._raw.clear()
        self._targets.clear()
        self._weights.clear()


class PolyRidgeCalibration:
    """
    2次多項式展開 + StandardScaler + Ridge 回帰によるキャリブレーション。
    入力: [X_feat, Y_feat, pitch_rad]  出力: [x_norm, y_norm]

    Pipeline: PolynomialFeatures(degree=2) → StandardScaler → Ridge(alpha=1.0)
    StandardScaler が必須: Ridgeはスケール依存のため正規化なしでは
    多項式項(X², XY, Y²)のスケール差でRidgeが正常に機能しない。
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import Ridge
        self._pipe_x = Pipeline([
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge',  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        self._pipe_y = Pipeline([
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge',  Ridge(alpha=alpha, fit_intercept=True)),
        ])
        self._is_fitted = False
        self._raw:     List[List[float]] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def add(self, X_mm: float, Y_mm: float,
            x_norm: float, y_norm: float,
            weight: float = 1.0,
            pitch_rad: float = 0.0) -> None:
        self._raw.append([X_mm, Y_mm, pitch_rad])
        self._targets.append([x_norm, y_norm])
        self._weights.append(max(0.0, weight))

    def fit(self) -> None:
        X_raw = np.array(self._raw,     dtype=np.float64)
        T     = np.array(self._targets, dtype=np.float64)
        W     = np.array(self._weights, dtype=np.float64)
        self._pipe_x.fit(X_raw, T[:, 0], ridge__sample_weight=W)
        self._pipe_y.fit(X_raw, T[:, 1], ridge__sample_weight=W)
        self._is_fitted = True

    def predict(self, X_mm: float, Y_mm: float, pitch_rad: float = 0.0) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        X_raw = np.array([[X_mm, Y_mm, pitch_rad]], dtype=np.float64)
        x = float(self._pipe_x.predict(X_raw)[0])
        y = float(self._pipe_y.predict(X_raw)[0])
        return np.array([x, y], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._is_fitted = False
        self._raw.clear()
        self._targets.clear()
        self._weights.clear()


class RidgeCalibration:
    """16D リッチ特徴ベクトル → 2D 画面座標。StandardScaler + Ridge(alpha=1.0) を x,y 各々。

    旧 TargetedPolyCalibration は入力が [X_feat, Y_feat, pitch] の実質2特徴で、しかも
    X_feat が『画像中心基準』のため顔の平行移動を視線と誤認していた(実測 loo 9.7cm)。
    本クラスは目頭・目尻基準の両眼虹彩＋頭部姿勢を含む16Dを受ける(実測 loo 3.4cm)。
    9点キャリブ(データが少ない)では豊富特徴でも過学習しないよう Ridge 正則化＋標準化で足りる
    (poly項は不要。むしろ有害。research_log/RESUME §3.5 参照)。
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self._scaler  = StandardScaler()
        self._ridge_x = Ridge(alpha=alpha, fit_intercept=True)
        self._ridge_y = Ridge(alpha=alpha, fit_intercept=True)
        self._is_fitted = False
        self._feats:   List[np.ndarray] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def add(self, feat: np.ndarray, x_norm: float, y_norm: float,
            weight: float = 1.0) -> None:
        self._feats.append(np.asarray(feat, dtype=np.float64).ravel())
        self._targets.append([float(x_norm), float(y_norm)])
        self._weights.append(max(0.0, float(weight)))

    def fit(self) -> None:
        F = np.array(self._feats,   dtype=np.float64)
        T = np.array(self._targets, dtype=np.float64)
        W = np.array(self._weights, dtype=np.float64)
        Fs = self._scaler.fit_transform(F)
        self._ridge_x.fit(Fs, T[:, 0], sample_weight=W)
        self._ridge_y.fit(Fs, T[:, 1], sample_weight=W)
        self._is_fitted = True

    def predict(self, feat: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        Fs = self._scaler.transform(np.asarray(feat, dtype=np.float64).reshape(1, -1))
        return np.array([float(self._ridge_x.predict(Fs)[0]),
                         float(self._ridge_y.predict(Fs)[0])], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._is_fitted = False
        self._feats.clear()
        self._targets.clear()
        self._weights.clear()


class HuberCalibration:
    """16D/7D 特徴 → 2D 画面座標。StandardScaler + HuberRegressor を x,y 各々。

    RidgeCalibration より外れフレーム(まばたき/視線が的に届く前の遷移フレーム)に強い。
    実データ探索(explore_accuracy.py 段階1/2)で全特徴セット・全前処理を通じて最良:
    7D で実効 median 1.874cm (Ridge 3.120cm から ~40%改善)。小細工(除去/アンサンブル)は
    かえって悪化＝素のHuberが一番。ハイパラもデフォルト(epsilon=1.35)が最良だった。
    I/F は RidgeCalibration と同一(次元非依存)。
    """

    def __init__(self, epsilon: float = 1.35, alpha: float = 1e-4):
        from sklearn.linear_model import HuberRegressor
        from sklearn.preprocessing import StandardScaler
        self._scaler  = StandardScaler()
        self._eps, self._alpha = epsilon, alpha
        self._ridge_x = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=800)
        self._ridge_y = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=800)
        self._is_fitted = False
        self._feats:   List[np.ndarray] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def add(self, feat: np.ndarray, x_norm: float, y_norm: float,
            weight: float = 1.0) -> None:
        self._feats.append(np.asarray(feat, dtype=np.float64).ravel())
        self._targets.append([float(x_norm), float(y_norm)])
        self._weights.append(max(0.0, float(weight)))

    def fit(self) -> None:
        F = np.array(self._feats,   dtype=np.float64)
        T = np.array(self._targets, dtype=np.float64)
        W = np.array(self._weights, dtype=np.float64)
        Fs = self._scaler.fit_transform(F)
        # HuberRegressor は sample_weight 対応。0重みは避けたいので下限クリップ。
        Wc = np.clip(W, 1e-3, None)
        try:
            self._ridge_x.fit(Fs, T[:, 0], sample_weight=Wc)
            self._ridge_y.fit(Fs, T[:, 1], sample_weight=Wc)
        except Exception:
            # 収束失敗時は Ridge にフォールバック(まれ)
            from sklearn.linear_model import Ridge
            self._ridge_x = Ridge(alpha=1.0).fit(Fs, T[:, 0], sample_weight=Wc)
            self._ridge_y = Ridge(alpha=1.0).fit(Fs, T[:, 1], sample_weight=Wc)
        self._is_fitted = True

    def predict(self, feat: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        Fs = self._scaler.transform(np.asarray(feat, dtype=np.float64).reshape(1, -1))
        return np.array([float(self._ridge_x.predict(Fs)[0]),
                         float(self._ridge_y.predict(Fs)[0])], dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        from sklearn.linear_model import HuberRegressor
        self._is_fitted = False
        self._ridge_x = HuberRegressor(epsilon=self._eps, alpha=self._alpha, max_iter=800)
        self._ridge_y = HuberRegressor(epsilon=self._eps, alpha=self._alpha, max_iter=800)
        self._feats.clear()
        self._targets.clear()
        self._weights.clear()


class H1Calibration:
    """7D → 2D。視線キャリブの古典(虹彩→画面 2次多項式)に頭部姿勢補正を足した構成。

    2段: (1)各ターゲット点で虹彩4D(目頭基準)を集約→2次多項式でベース注視点を予測、
         (2)pitch/yaw/dist の2次多項式で「頭部姿勢による残差」を補正。
    次元は7Dのまま(多項式は既存7Dの非線形展開でモデル内部。新しい特徴は足していない)。
    多姿勢データで横向き30+を 線形Huber単体 14.55cm → 8.72cm に改善(explore_hybrid_pose H1)。
    I/F は Ridge/HuberCalibration と同一(collect→add, finalize→fit, predict)。
    """

    def __init__(self, alpha_base: float = 0.1, alpha_pose: float = 1.0):
        self._ab, self._ap = alpha_base, alpha_pose
        self._is_fitted = False
        self._feats:   List[np.ndarray] = []
        self._targets: List[List[float]] = []
        self._weights: List[float]       = []

    def add(self, feat: np.ndarray, x_norm: float, y_norm: float,
            weight: float = 1.0) -> None:
        self._feats.append(np.asarray(feat, dtype=np.float64).ravel())
        self._targets.append([float(x_norm), float(y_norm)])
        self._weights.append(max(0.0, float(weight)))

    def _mk(self, alpha):
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        return make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=alpha))

    def fit(self) -> None:
        F = np.array(self._feats,   dtype=np.float64)
        T = np.array(self._targets, dtype=np.float64)
        # (1) 各ターゲット点で虹彩4Dを集約(中央値) → 2次多項式でベース予測
        key = np.round(T, 4)
        uniq, ids = np.unique(key, axis=0, return_inverse=True)
        Xa = np.array([np.median(F[ids == i][:, :4], axis=0) for i in range(len(uniq))])
        self._bx = self._mk(self._ab).fit(Xa, uniq[:, 0])
        self._by = self._mk(self._ab).fit(Xa, uniq[:, 1])
        # (2) 全フレームでベースの残差を pitch/yaw/dist の2次で補正
        base = np.column_stack([self._bx.predict(F[:, :4]), self._by.predict(F[:, :4])])
        resid = T - base
        P = F[:, 4:7]
        self._rx = self._mk(self._ap).fit(P, resid[:, 0])
        self._ry = self._mk(self._ap).fit(P, resid[:, 1])
        # kappa_offset_norm 互換(ベース多項式の intercept を露出)
        self._ridge_x = self._bx.named_steps['ridge']
        self._ridge_y = self._by.named_steps['ridge']
        self._is_fitted = True

    def predict(self, feat: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        f = np.asarray(feat, dtype=np.float64).reshape(1, -1)
        base = np.array([self._bx.predict(f[:, :4])[0], self._by.predict(f[:, :4])[0]])
        corr = np.array([self._rx.predict(f[:, 4:7])[0], self._ry.predict(f[:, 4:7])[0]])
        return (base + corr).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._is_fitted = False
        self._feats.clear()
        self._targets.clear()
        self._weights.clear()


class HybridCalibration:
    """正面=7D H1, 横向き=16D Huber を |yaw| 閾値でハード切替。キャリブで両方を並列学習。
    ユーザ要望(2026-07-22): 正面の使用感(H1 実機1.4cm)を保ちつつ、横向きは16Dで崩れにくく。
    入力 feat は16D。7D側は feat[:7] を使う。切替は feat[5]=yaw(rad)。"""

    def __init__(self, yaw_thresh_deg: float = 10.0):
        self._h1  = H1Calibration()      # 正面用(7D)
        self._h16 = HuberCalibration()   # 横向き用(16D)
        self._thr = np.radians(yaw_thresh_deg)
        self._yaws: List[float] = []
        self._ref = 0.0
        self._is_fitted = False

    def add(self, feat: np.ndarray, x_norm: float, y_norm: float, weight: float = 1.0) -> None:
        feat = np.asarray(feat, dtype=np.float64).ravel()
        self._yaws.append(abs(float(feat[5])))           # キャリブ姿勢の基準yawを収集
        self._h1.add(feat[:7], x_norm, y_norm, weight)   # 7D側にも
        self._h16.add(feat,    x_norm, y_norm, weight)   # 16D側にも(両方=並列学習)

    def fit(self) -> None:
        self._h1.fit()
        self._h16.fit()
        # キャリブ姿勢の代表|yaw|。ユーザの正面がyaw=0でなくても、ここを基準に切替する。
        self._ref = float(np.median(self._yaws)) if self._yaws else 0.0
        self._is_fitted = True

    def predict(self, feat: np.ndarray) -> np.ndarray:
        feat = np.asarray(feat, dtype=np.float64).ravel()
        # キャリブ姿勢(ref)からの |Δyaw| で切替。ユーザの正面がyaw=0でなくても正しく効く。
        if abs(abs(float(feat[5])) - self._ref) < self._thr:   # キャリブ姿勢の近く = 正面 → 7D H1
            return self._h1.predict(feat[:7])
        return self._h16.predict(feat)          # キャリブ姿勢から離れた = 横向き → 16D Huber

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def _feats(self):     # _compute_loo のサンプル数判定用(16D側に委譲)
        return self._h16._feats

    @property
    def _ridge_x(self):   # kappa_offset_norm 互換(7D側のintercept)
        return self._h1._ridge_x

    @property
    def _ridge_y(self):
        return self._h1._ridge_y

    def reset(self) -> None:
        self._h1.reset(); self._h16.reset(); self._yaws.clear(); self._ref = 0.0; self._is_fitted = False


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HistEntry:
    screen_gt: np.ndarray
    predicted: np.ndarray
    head_vec:  np.ndarray   # [pitch, yaw] 正規化済み


class DynamicCalibration:
    """
    ユーザーの操作履歴（タップ等）を使った動的自動補正。

    補正式:
        G_C = G_E + (Σ λ_i * h_i * dG_i) / Σ λ_i * h_i
        λ_i = 1 / ||dG_i||          距離逆数重み
        h_i = dot(H_current, H_i)   頭部姿勢コサイン類似度（負は0でクランプ）
    """

    def __init__(self, max_history: int = 16, min_err: float = 1e-6):
        self.max_history = max_history
        self.min_err     = min_err
        self._hist: List[_HistEntry] = []

    def add(self, screen_gt: np.ndarray, predicted: np.ndarray, head_vec: np.ndarray) -> None:
        norm  = np.linalg.norm(head_vec) + 1e-8
        entry = _HistEntry(
            screen_gt=screen_gt.astype(np.float32),
            predicted=predicted.astype(np.float32),
            head_vec=(head_vec / norm).astype(np.float32),
        )
        self._hist.append(entry)
        if len(self._hist) > self.max_history:
            self._hist.pop(0)

    def correct(self, predicted: np.ndarray, current_head: np.ndarray) -> np.ndarray:
        if not self._hist:
            return predicted.copy()
        h_cur = current_head / (np.linalg.norm(current_head) + 1e-8)
        num = np.zeros(2, dtype=np.float64)
        den = 0.0
        for e in self._hist:
            dG  = e.screen_gt - e.predicted
            err = np.linalg.norm(dG)
            if err < self.min_err:
                continue
            h_sim        = max(0.0, float(np.dot(h_cur, e.head_vec)))
            spatial_dist = np.linalg.norm(predicted - e.screen_gt)
            w            = h_sim / (spatial_dist + 1e-4)
            num  += w * dG
            den  += w
        if den < 1e-10:
            return predicted.copy()
        return (predicted + num / den).astype(np.float32)

    def __len__(self) -> int:
        return len(self._hist)


# ─────────────────────────────────────────────────────────────────────────────

class CalibrationPipeline:
    """
    9点キャリブレーション + 動的補正 の統合パイプライン。

    使い方:
        1. collect_point(gaze_2d, target, weight, pitch_rad, yaw_rad) を各注視点中に呼ぶ
        2. finalize() でポリリッジ回帰をフィット
        3. predict(gaze_2d, head_vec) で補正済み画面座標を得る
        4. record_interaction(gt, gaze_2d, head_vec) でタップ履歴を蓄積
    """

    # 推定を止める閾値ではなく「キャリブ姿勢からどれだけ外れたか」の警告閾値。
    # 外れても estimator は推定を続ける(劣化するが止まらない)。
    HEAD_POSE_TOLERANCE_RAD: float = np.radians(20.0)

    def __init__(self, mode: str = '7d', yaw_thresh_deg: float = 10.0):
        # config.MODE から。'7d'=H1(正面特化), '16d'=Huber(横向き改善), 'hybrid'=姿勢で両方切替。
        self._mode = mode
        self._yaw_thr = yaw_thresh_deg
        if mode == 'hybrid':
            self.poly_ridge = HybridCalibration(yaw_thresh_deg)
        elif mode == '16d':
            self.poly_ridge = HuberCalibration()
        else:
            self.poly_ridge = H1Calibration()
        self.dynamic    = DynamicCalibration()
        self._samples:       List[Tuple] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples:   List[float] = []
        self.train_mgae: Optional[float] = None
        self.train_rmse: Optional[float] = None
        self.loo_mgae:   Optional[float] = None
        self.loo_euc:    Optional[float] = None
        self.loo_euc_x:  Optional[float] = None
        self.loo_euc_y:  Optional[float] = None
        self._ref_pitch: float = 0.0
        self._ref_yaw:   float = 0.0

    def collect_point(
        self,
        gaze_2d:   np.ndarray,
        target:    np.ndarray,
        weight:    float = 1.0,
        pitch_rad: Optional[float] = None,
        yaw_rad:   Optional[float] = None,
    ) -> None:
        pitch = float(pitch_rad) if pitch_rad is not None else 0.0
        self._samples.append((gaze_2d.copy(), target.astype(np.float32).copy(), float(weight), pitch))
        if pitch_rad is not None:
            self._pitch_samples.append(float(pitch_rad))
        if yaw_rad is not None:
            self._yaw_samples.append(float(yaw_rad))

    def finalize(self) -> None:
        if len(self._samples) < 5:
            raise ValueError(f"キャリブレーションサンプルが不足: {len(self._samples)} < 5")

        for feat, target, weight, pitch in self._samples:
            self.poly_ridge.add(feat, float(target[0]), float(target[1]), weight)
        self.poly_ridge.fit()

        if self._pitch_samples:
            self._ref_pitch = float(np.mean(self._pitch_samples))
            self._ref_yaw   = float(np.mean(self._yaw_samples))

        y_raw  = np.stack([s[1] for s in self._samples])
        y_pred = np.array([self.poly_ridge.predict(s[0]) for s in self._samples])
        self.train_rmse = float(np.sqrt(np.mean((y_pred - y_raw) ** 2)))
        self.train_mgae = float(self._compute_mgae(y_pred, y_raw))

        self.loo_mgae, self.loo_euc, self.loo_euc_x, self.loo_euc_y = self._compute_loo()

    def head_pose_ok(self, pitch: float, yaw: float) -> bool:
        if not self.is_calibrated:
            return True
        tol = self.HEAD_POSE_TOLERANCE_RAD
        return (abs(pitch - self._ref_pitch) <= tol and
                abs(yaw   - self._ref_yaw)   <= tol)

    def head_pose_delta_deg(self, pitch: float, yaw: float) -> Tuple[float, float]:
        return float(np.degrees(pitch - self._ref_pitch)), float(np.degrees(yaw - self._ref_yaw))

    def predict(
        self,
        gaze_2d:     np.ndarray,
        head_vec:    Optional[np.ndarray] = None,
        use_dynamic: bool = True,
    ) -> np.ndarray:
        if not self.poly_ridge.is_fitted:
            return np.array([0.5, 0.5], dtype=np.float32)
        pred = self.poly_ridge.predict(gaze_2d)
        if use_dynamic and head_vec is not None and len(self.dynamic) > 0:
            pred = self.dynamic.correct(pred, head_vec)
        return pred

    def record_interaction(
        self,
        screen_gt: np.ndarray,
        gaze_2d:   np.ndarray,
        head_vec:  np.ndarray,
    ) -> None:
        if self.poly_ridge.is_fitted:
            predicted = self.poly_ridge.predict(gaze_2d)
        else:
            predicted = np.array([0.5, 0.5], dtype=np.float32)
        self.dynamic.add(screen_gt, predicted, head_vec)

    def _compute_loo(self) -> Tuple[Optional[float], ...]:
        """
        Leave-One-Out Cross-Validation。
        9点のうち1点を除いて残り8点でフィット → 除いた点を予測 → 誤差計算。
        """
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for s in self._samples:
            key = (round(float(s[1][0]), 4), round(float(s[1][1]), 4))
            groups[key].append(s)

        if len(groups) < 3:
            return None, None, None, None

        mgae_list = []
        euc_list  = []
        ex_list   = []
        ey_list   = []

        for held_key, held_samples in groups.items():
            if self._mode == 'hybrid':
                af = HybridCalibration(self._yaw_thr)
            elif self._mode == '16d':
                af = HuberCalibration()
            else:
                af = H1Calibration()
            for key, samples in groups.items():
                if key == held_key:
                    continue
                # 各点フレームを間引いて H1 fit を高速化(キャリブ後の数秒固まりを解消)。
                # 多姿勢でも代表~80点あれば LOO 表示は十分正確。
                step = max(1, len(samples) // 80)
                for feat, target, weight, pitch in samples[::step]:
                    af.add(feat, float(target[0]), float(target[1]), weight)
            if len(af._feats) < 5:
                continue
            af.fit()

            preds = np.array([af.predict(s[0]) for s in held_samples])
            pred_med  = np.median(preds, axis=0)
            target_pt = np.array(held_key)
            err       = pred_med - target_pt

            euc_list.append(float(np.linalg.norm(err)))
            ex_list.append(float(abs(err[0])))
            ey_list.append(float(abs(err[1])))

            def to3d(p): return np.array([p[0]-0.5, p[1]-0.5, 1.0])
            v1, v2  = to3d(pred_med), to3d(target_pt)
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            mgae_list.append(float(np.degrees(np.arccos(np.clip(cos_sim, -1+1e-7, 1-1e-7)))))

        if not mgae_list:
            return None, None, None, None
        return (float(np.mean(mgae_list)), float(np.mean(euc_list)),
                float(np.mean(ex_list)),   float(np.mean(ey_list)))

    @staticmethod
    def _compute_mgae(pred: np.ndarray, target: np.ndarray) -> float:
        def to3d(p): return np.array([p[0] - 0.5, p[1] - 0.5, 1.0])
        angles = []
        for p, t in zip(pred, target):
            v1, v2 = to3d(p), to3d(t)
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos_sim, -1 + 1e-7, 1 - 1e-7))))
        return float(np.mean(angles))

    def reset(self) -> None:
        self._samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        if self._mode == 'hybrid':
            self.poly_ridge = HybridCalibration(self._yaw_thr)
        elif self._mode == '16d':
            self.poly_ridge = HuberCalibration()
        else:
            self.poly_ridge = H1Calibration()
        self.dynamic    = DynamicCalibration()
        self.train_mgae = None
        self.train_rmse = None
        self.loo_mgae   = None
        self.loo_euc    = None
        self.loo_euc_x  = None
        self.loo_euc_y  = None
        self._ref_pitch = 0.0
        self._ref_yaw   = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self.poly_ridge.is_fitted

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def kappa_offset_norm(self) -> np.ndarray:
        if not self.is_calibrated:
            return np.zeros(2, dtype=np.float32)
        rx = self.poly_ridge._ridge_x.intercept_
        ry = self.poly_ridge._ridge_y.intercept_
        return np.array([float(rx), float(ry)], dtype=np.float32)
