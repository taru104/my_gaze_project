# リアルタイム視線推定システム

Webカメラ1台で動作するリアルタイム視線推定システム。  
MediaPipe FaceLandmarker で顔ランドマーク・虹彩を検出し、幾何学的な距離不変特徴量と アフィン回帰でキャリブレーション後に画面上の視線座標を推定する。

---

## 目次

1. [システム概要](#1-システム概要)
2. [ファイル構成](#2-ファイル構成)
3. [処理パイプライン全体像](#3-処理パイプライン全体像)
4. [各モジュール詳細](#4-各モジュール詳細)
   - 4.1 features.py — 特徴量抽出
   - 4.2 calibration.py — キャリブレーション
   - 4.3 filters.py — フィルタリング
   - 4.4 estimator.py — 統合パイプライン
   - 4.5 main.py — アプリケーション
5. [キャリブレーション詳細フロー](#5-キャリブレーション詳細フロー)
6. [GazeCapture ベンチマーク結果](#6-gazecapture-ベンチマーク結果)
7. [バグ修正履歴](#7-バグ修正履歴)
8. [操作方法](#8-操作方法)
9. [依存ライブラリ](#9-依存ライブラリ)
10. [パラメータ一覧](#10-パラメータ一覧)
11. [既知の制限と今後の課題](#11-既知の制限と今後の課題)

---

## 1. システム概要

### 何をするシステムか

Webカメラの映像をリアルタイムで処理し、ユーザーが画面のどこを見ているか（視線座標）をピクセル単位で追跡する。

- **入力**: Webカメラ映像 (640×480, 30fps)
- **出力**: 正規化視線座標 (x, y) ∈ [0,1]²、画面上の光るカーソル
- **追加機能**: セッションごとの CSV ログ保存、動的自動補正（タップ記録）

### アーキテクチャ方針

外部の大型 DNN モデルは使用しない。代わりに：

1. **MediaPipe** で顔ランドマーク (478点) と虹彩ランドマーク (10点) を検出
2. **虹彩直径ベース深度推定** で距離不変の 2D 特徴量 `[X_feat, Y_feat]` を作る
3. **AffineCalibration**（2×3 最小二乗）でキャリブレーション時に 2D→2D の写像を学習する
4. **One Euro Filter** でフレーム間のノイズ・ジッターを平滑化する

---

## 2. ファイル構成

```
my_gaze_project/
├── main.py                   アプリエントリポイント・UI
├── estimator.py              リアルタイム推定パイプライン統合
├── features.py               MediaPipe 特徴量抽出 + HeadFilter
├── calibration.py            AffineCalibration + 動的補正
├── filters.py                IQR / EMA / Kalman / One Euro フィルタ
├── evaluate_new_pipeline.py  GazeCapture ベンチマーク（バッチ評価用）
├── face_landmarker.task      MediaPipe モデルファイル (3.6 MB)
├── logs/                     セッションごとの CSV ログ（自動生成）
│   └── session_YYYYMMDD_HHMMSS.csv
└── new_pipeline_results.txt  直近ベンチマーク結果
```

---

## 3. 処理パイプライン全体像

```
[Webカメラ フレーム]
        │
        ▼
┌─────────────────────────────────┐
│  GazeFeatureExtractor           │  features.py
│  ┌─────────────────────────┐    │
│  │ MediaPipe FaceLandmarker│    │  478 点ランドマーク検出
│  │  (VIDEO mode, Tasks API) │   │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │ solvePnP + HeadFilter   │    │  6点で頭部姿勢推定
│  │ (OneEuroFilterND 6D)    │    │  min_cutoff=0.3, beta=0.01
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │ 虹彩深度推定（距離不変）  │    │  X_feat=(iris_x-cx)/iris_diam
│  └─────────────────────────┘    │
│  出力: [X_feat, Y_feat] (2D)    │
│         pitch, yaw (rad)        │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  CalibrationPipeline            │  calibration.py
│  ┌─────────────────────────┐    │
│  │ AffineCalibration       │    │  [X_feat, Y_feat, 1] lstsq
│  │ 2×3 最小二乗            │    │  → [x_norm, y_norm]
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │ DynamicCalibration      │    │  タップ履歴で誤差補正
│  └─────────────────────────┘    │
│  出力: raw_pred (2D)             │
└─────────────────────────────────┘
        │
        │ np.clip(raw_pred, -0.05, 1.05)  ← エッジ張り付き防止
        ▼
┌─────────────────────────────────┐
│  OneEuroFilter2D                │  filters.py
│  適応型ローパスフィルタ          │
│  遅い動き → 強平滑              │
│  速い動き（サッケード）→ 追従    │
└─────────────────────────────────┘
        │
        │ np.clip(smoothed, 0.0, 1.0)
        ▼
[視線座標 gaze (x, y) ∈ [0,1]²]
        │
        ▼
[画面上のカーソル描画 + CSV ログ記録]
```

---

## 4. 各モジュール詳細

### 4.1 `features.py` — 特徴量抽出

#### クラス: `GazeFeatureExtractor`

**役割**: フレーム1枚を受け取り、距離不変の 2D 特徴量ベクトルと頭部姿勢を返す。

#### 距離不変特徴量（コアアイデア）

```
X_feat = ((iris_x_L - cx) / iris_diam_L + (iris_x_R - cx) / iris_diam_R) / 2
Y_feat = ((iris_y_L - cy) / iris_diam_L + (iris_y_R - cy) / iris_diam_R) / 2
```

**なぜ距離不変か**:  
虹彩の画像上の変位 `iris_x - cx` は `Z_eye * tan(θ) / f_px` に比例する（`Z_eye` = 眼球深度）。  
虹彩の画像上の直径 `iris_diam_px` も `f_px * 11.7mm / Z_eye` に比例する。  
両者の比を取ると `f_px` と `Z_eye` の両方がキャンセルされ、純粋に視線角 `tan(θ)` に比例した距離不変量になる。

以前の `X_mm = 11.7 * (iris_x - cx) / iris_diam_px` も同様の計算だが、スケール定数 11.7mm を乗じていたため AffineCalibration の係数が変化した。現在はスケール定数を省いた無次元形式を使用。

#### MediaPipe の使い方（Tasks API）

MediaPipe 0.10 以降は `mediapipe.tasks.python.vision.FaceLandmarker` を使用する。

```python
options = mp_vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=mp_vision.RunningMode.VIDEO,   # VIDEO モード必須
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
```

**VIDEO モードとタイムスタンプの注意点**:  
`detect_for_video(mp_image, ts_ms)` に渡すタイムスタンプは単調増加でなければならない。  
Unix タイムスタンプをそのまま渡すとゼロ検出バグが起きるため、プロセス起動からの相対時刻 (ms) を使用する。

#### 頭部姿勢推定（solvePnP + HeadFilter）

6点の解剖学的 3D モデル（mm スケール）と対応する 2D 画像座標から `cv2.solvePnP` で `rvec/tvec` を求める。  
`rvec (3D) + tvec (3D) = 6D` を **OneEuroFilterND（HeadFilter）** でスムージングしてから  
`cv2.RQDecomp3x3` で Pitch/Yaw に変換する。

```python
_FACE_3D_MODEL = [
    [ 0.0,    0.0,    0.0  ],   # 鼻先      (landmark 1)
    [ 0.0,  -63.6,  -12.5 ],   # 顎先      (landmark 152)
    [-43.3,  32.7,  -26.0 ],   # 左目外角  (landmark 33)
    [ 43.3,  32.7,  -26.0 ],   # 右目外角  (landmark 263)
    [-28.9, -28.9,  -24.1 ],   # 左口角    (landmark 61)
    [ 28.9, -28.9,  -24.1 ],   # 右口角    (landmark 291)
]
```

**HeadFilter パラメータ**: `min_cutoff=0.3, beta=0.01`  
頭部姿勢ゲートの安定性のために低周波成分のみ通す。

#### 虹彩スムージングについて

EyeFilter（以前は虹彩位置+直径を別途スムージング）は除去済み。  
虹彩の生値を直接使い、最終的な視線座標は `estimator.py` の `OneEuroFilter2D` だけでスムージングする。  
EyeFilter を残すと `OneEuroFilter2D` との二重フィルタになりラグが増加するため削除した。

#### 虹彩ランドマーク

- 左虹彩: indices 468〜472（5点の重心を虹彩中心、4点の最大対角距離を直径とする）
- 右虹彩: indices 473〜477

---

### 4.2 `calibration.py` — キャリブレーション

#### クラス: `AffineCalibration`

**役割**: `[X_feat, Y_feat]` → `[x_norm, y_norm]` の 2×3 アフィン変換を重み付き最小二乗で学習する。

**設計行列**: `(N, 3) = [X_feat, Y_feat, 1]`  
定数項 `1` によってオフセット（カッパ角補正）が自動的に吸収される。

```python
# sqrt(w) でスケールして重み付き lstsq と等価にする
D * W → lstsq → A (3×2)

# 推定
[X_feat, Y_feat, 1] @ A → [x_norm, y_norm]
```

**カッパ角について**:  
アフィン行列のバイアス項 `A[2, :]` が各ユーザーの光学軸-視覚軸ズレ（カッパ角）に対応する。  
`CalibrationPipeline.kappa_offset_norm` プロパティで参照できる（デバッグ用）。

#### クラス: `DynamicCalibration`

**役割**: ユーザーがタップ・クリックした座標を使ってリアルタイムで誤差を補正する。

**補正式**:

```
G_C = G_E + (Σ λ_i * h_i * dG_i) / (Σ λ_i * h_i)

λ_i = 1 / ||dG_i||           ← 誤差が大きいほど信頼度低い
h_i = max(0, cos(H_cur, H_i)) ← 頭部方向が近いタップほど重視
dG_i = screen_gt_i - predicted_i
```

最大 50 件の履歴を保持。現在の頭部方向に似た角度でのタップ誤差を優先的に使う。

#### クラス: `CalibrationPipeline`

`AffineCalibration` と `DynamicCalibration` を統合したインターフェース。

- `collect_point(gaze_2d, target, weight, pitch_rad, yaw_rad)`: キャリブ中の1フレーム分を蓄積
- `finalize()`: アフィン回帰をフィット + 訓練誤差 (MGAE, RMSE) を計算
- `predict(gaze_2d, head_vec)`: 補正済み画面座標を返す
- `record_interaction(screen_gt, gaze_2d, head_vec)`: タップ記録
- `head_pose_ok(pitch, yaw)`: 頭部姿勢が許容範囲内か確認（許容 ±20°）
- `kappa_offset_norm`: アフィンバイアス項（カッパ角補正量）を返す

---

### 4.3 `filters.py` — フィルタリング

#### `OneEuroFilter` / `OneEuroFilter2D`（メイン平滑化器）

**論文**: Casiez et al. 2012, "1€ Filter"

- 静止中: `fc ≈ min_cutoff` → 強い平滑化（ノイズ除去）
- サッケード: 速度に応じて `fc = min_cutoff + beta * |dx/dt|` → α が 1.0 に近づき追従

```python
OneEuroFilter2D(min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
```

#### `OneEuroFilterND`

N次元に独立して One Euro Filter を適用する。

| 用途 | dim | min_cutoff | beta | 説明 |
|------|-----|-----------|------|------|
| HeadFilter | 6 | 0.3 | 0.01 | rvec+tvec → 安定した頭部姿勢ゲート |

#### その他のフィルタ

- `IQRFilter`: 四分位範囲ベース外れ値検出。定義済みだがパイプラインからは除外済み
- `EMAFilter`: 指数移動平均。速度適応なし
- `KalmanFilter2D`: 状態 `[x, y, vx, vy]`。定常ゲインが高すぎ（≈0.91）平滑化が効かないため未使用

---

### 4.4 `estimator.py` — 統合パイプライン

#### クラス: `GazeEstimator`

**主な処理フロー（`process_frame`）**:

```python
1. GazeFeatureExtractor.extract(frame)
   → gaze_2d [X_feat, Y_feat] (2D), debug dict, または None（顔未検出）

2. CalibrationPipeline.predict(gaze_2d, head_vec)
   → raw_pred (2D)

3. np.clip(raw_pred, -0.05, 1.05)  # エッジ張り付き防止

4. OneEuroFilter2D.update(raw_clipped, dt)
   → smoothed (2D)

5. np.clip(smoothed, 0.0, 1.0)
   → gaze (2D)
```

**キャリブレーション収集**:

```python
def collect_calibration(self, target_normalized, weight=1.0):
    if self._current_features is not None:
        self.calibration.collect_point(
            self._current_features, target_normalized, weight,
            pitch_rad, yaw_rad
        )
        return True
    return False
```

---

### 4.5 `main.py` — アプリケーション

#### 状態管理

```
idle → calibrating → running
         ↑
       C キーで遷移
```

#### UI 要素

| 要素 | 説明 |
|------|------|
| 左上プレビュー | 213×160 px のカメラ映像 |
| 視線カーソル | 青い円（半径 24px） + 内円（4px） |
| 軌跡 | 最大 60 フレームのグラデーション |
| HUD 左下 | キャリブ状態 + 訓練誤差 (MGAE / RMSE / N) |
| HUD 右上 | FPS |
| HUD デバッグ | D キーで Pitch/Yaw/特徴量/生予測 を表示 |

#### セッション CSV ログ

`logs/session_YYYYMMDD_HHMMSS.csv` として起動ごとに別ファイルを作成。

```csv
time_s, gaze_x, gaze_y, raw_x, raw_y, pitch_deg, yaw_deg,
face_detected, calibrated, calib_mgae_deg, calib_rmse
```

---

## 5. キャリブレーション詳細フロー

### キャリブレーション点（5点 or 9点）

```
(0.1, 0.1) ─────────── (0.9, 0.1)
     │                      │
     │         (0.5, 0.5)   │
     │                      │
(0.1, 0.9) ─────────── (0.9, 0.9)
```

`CALIB_POINTS_9` (3×3グリッド) も `calibration.py` に定義済み。

### サンプル収集（時間的重み付き）

各点 3.0 秒間注視。最初の 1.0 秒は "Stabilizing" として収集しない。  
後半 2.0 秒間は時間の経過とともに線形に重みが増加する:

```
t=1.0s: weight = 0.0
t=2.0s: weight = 0.5
t=3.0s: weight = 1.0
```

### アフィン回帰のフィット

5点 × 約 60fps × 2秒 = 最大 600 サンプルで `AffineCalibration.fit()` を呼ぶ。  
フィット後に訓練誤差（MGAE, RMSE）を計算して HUD に表示。

---

## 6. GazeCapture ベンチマーク結果

**評価プロトコル** (`evaluate_new_pipeline.py`):
- データセット: GazeCapture テストスプリット（26 被験者、43,506 フレーム）
- キャリブレーション: 各被験者の最初の 5% フレーム（AffineCalibration, 2×3 lstsq）
- 評価: 残り 95% フレーム

```
  Method                                    MGAE_2D   MGAE_3D   Euc(cm)
                                               mean      mean    median
  ------------------------------------------------------------------------
  EyeTrax(alpha=1.0)                          54.51     26.83     24.66
  EyeTrax(alpha=adaptive)                     44.15     18.06     16.36
  Custom(7D+poly36,adaptive)                  45.54     22.29     21.30
  Webcam3DTracker(PCA sphere)                 44.55      9.94      7.59
  solvePnP+AffineCalib(v1)                    44.21     14.61      8.22
  IrisDepth+AffineCalib(v2,current)           43.92     14.71      7.81
  ------------------------------------------------------------------------
```

**ベンチマーク上の注意**: 最初の 5% キャリブデータは 1〜2 点しかスクリーン位置をカバーしない場合があり、  
アフィン変換の外挿誤差が大きい被験者が存在する（例: 被験者 509 の Euc=159cm は外挿起因）。  
実アプリでは 5 点キャリブレーションで均等分布を保証するため、この問題は発生しない。

---

## 7. バグ修正履歴

### バグ1: MediaPipe `mp.solutions` が AttributeError

**原因**: MediaPipe 0.10 以降で `mp.solutions.face_mesh` が削除された。  
**修正**: `mediapipe.tasks.python.vision.FaceLandmarker`（Tasks API）に全面書き直し。

### バグ2: VIDEO モードで顔が 0 件検出

**原因**: `detect_for_video()` に Unix タイムスタンプ（約 1.7 兆 ms）を渡していた。  
**修正**: プロセス起動からの相対時刻 (ms) を使用。`max(ts, last+1)` で重複を防ぐ。

### バグ3: キャリブレーション 0 サンプル問題

**原因**: サンプル収集条件が `gaze is not None` だったが、キャリブ前は gaze が None。  
**修正**: `self.estimator.face_detected` に条件変更 + `_current_features`（生特徴量）を収集に使用。

### バグ4: 視線が画面左上に貼り付く

**原因**: キャリブ未実施時の `predict()` が生の虹彩座標（≈ 0, 0）を返していた。  
**修正**: フォールバックを `np.array([0.5, 0.5])` に変更。

### バグ5: IQR がキャリブ後の 97% のフレームを弾く

**原因**: バッファがキャリブ最終注視点のデータのみで埋まり、サッケードが全て外れ値扱いされた。  
**修正**: IQR をパイプラインから除去。

### バグ6: 視線が画面端に張り付く（エッジスタッキング）

**原因**: アフィン変換は外挿するため `raw_pred` が [0,1] 外の値になることがある。  
Kalman/One Euro フィルタの内部状態が高い値に収束し、その後戻るのに時間がかかった。  
**修正**: `np.clip(raw_pred, -0.05, 1.05)` でフィルタに渡す前にクリップ。

### バグ7: 頭部を動かすと視線が飛ぶ（距離依存性）

**原因**: 旧特徴量 `X_mm = 11.7 * (iris_x - cx) / iris_diam_px` は `Z_eye * tan(θ)` に比例し、  
頭部を 10cm 前後させると値が ~17% 変動していた。  
**修正**: 無次元・距離不変の `X_feat = (iris_x - cx) / iris_diam_px` に変更（`Z_eye` がキャンセル）。

### バグ8: 視線追従にラグがある（二重フィルタ）

**原因**: EyeFilter（features.py の OneEuroFilterND, min_cutoff=1.5）と  
OneEuroFilter2D（estimator.py）が直列に適用されていた。  
**修正**: EyeFilter を除去。最終スムージングは estimator の OneEuroFilter2D のみ。

### バグ9: 頭部姿勢ゲートが不安定

**原因**: HeadFilter の `min_cutoff=0.1` が低すぎ、頭部姿勢の追従が遅れてゲートが誤動作した。  
**修正**: `min_cutoff=0.3` に緩和。

---

## 8. 操作方法

### 起動

```powershell
python C:\Users\hazib\my_gaze_project\main.py
# カメラ ID を指定する場合（例: カメラ 1）
python C:\Users\hazib\my_gaze_project\main.py 1
```

### キーバインド

| キー | 動作 |
|------|------|
| `C` | 9点（または5点）キャリブレーション開始 |
| `R` | キャリブレーションリセット（idle 状態に戻る） |
| `Space` | 現在の視線位置をタップとして動的キャリブレーションに記録 |
| `D` | デバッグ HUD の表示 / 非表示 |
| `Q` または `Esc` | 終了 |

### キャリブレーション手順

1. `C` キーを押す
2. 画面に点が順番に表示される
3. 各点を 3 秒間しっかり見つめる（最初の 1 秒は "Stabilizing..." と表示）
4. 全点が完了すると自動的に推定モードに移行
5. 下部ステータスに `CALIBRATED [train MGAE=X.XX deg RMSE=X.XXXX n=NNN]` が表示される

### GazeCapture ベンチマーク（開発用）

```powershell
python C:\Users\hazib\my_gaze_project\evaluate_new_pipeline.py
```

---

## 9. 依存ライブラリ

```
mediapipe >= 0.10      Tasks API が必要（FaceLandmarker）
opencv-python          cv2 (solvePnP 含む)
numpy
```

---

## 10. パラメータ一覧

### `GazeFeatureExtractor` (`features.py`)

| パラメータ | 値 | 説明 |
|-----------|---|------|
| `min_face_detection_confidence` | 0.5 | MediaPipe 顔検出閾値 |
| HeadFilter `min_cutoff` | 0.3 Hz | 頭部姿勢フィルタのカットオフ |
| HeadFilter `beta` | 0.01 | 頭部姿勢フィルタの速度感度 |

### `OneEuroFilter2D` (`estimator.py`)

| パラメータ | デフォルト | 効果 |
|-----------|----------|------|
| `min_cutoff` | 1.5 Hz | 静止時の平滑化強度 |
| `beta` | 0.05 | 速度感度（大きいほどサッケード追従性UP・ノイズ感度UP） |
| `d_cutoff` | 1.0 Hz | 速度推定用カットオフ（通常変更不要） |

### `AffineCalibration` (`calibration.py`)

設計行列 `[X_feat, Y_feat, 1]` (N×3) → 解 `A` (3×2)。  
正則化なし（最小二乗解）。重み付きにより時間的安定サンプルを優先。

### `CalibrationPipeline` (`calibration.py`)

| パラメータ | 値 | 説明 |
|-----------|---|------|
| `HEAD_POSE_TOLERANCE_RAD` | 20° | 頭部姿勢ゲートの許容範囲 |
| `DynamicCalibration.max_history` | 50 | タップ履歴の最大件数 |

### キャリブレーション収集 (`main.py`)

| 定数 | 値 | 説明 |
|------|---|------|
| `CALIB_TOTAL` | 3.0 秒 | 各点の注視時間 |
| `CALIB_DISCARD` | 1.0 秒 | 最初に破棄する秒数（安定待ち） |

---

## 11. 既知の制限と今後の課題

### 現在の制限

1. **照明依存**: MediaPipe の顔検出は逆光・暗所で失敗しやすい
2. **眼鏡**: 反射光が虹彩ランドマークの精度を下げる
3. **単一ユーザー**: `num_faces=1` 固定
4. **キャリブの粗さ**: 5点では画面周辺部の精度が低い場合がある

### 今後の課題

1. **完全な Data Normalization (Zhang et al. 2015)**: `rvec/tvec` から虹彩 3D 座標を逆投影し、680mm 正規化カメラで再射影
2. **ユーザープロファイル保存**: キャリブ済みのアフィン行列 `A` をファイルに保存し次回起動時に再利用
3. **GazeCapture ベンチマークの外挿問題**: 最初の 5% プロトコルでのアフィン外挿誤差の軽減
