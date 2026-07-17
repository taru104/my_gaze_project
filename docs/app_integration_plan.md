# ハイブリッド手法の実アプリ統合 設計メモ

研究成果(`results/research_log.md`)を実アプリ(main.py/estimator.py/calibration.py)に
組み込むための設計。`hybrid_calibration.HybridCalibration` を核に据える。

---

## 座標系の壁 (最重要)

- グローバルモデルは GazeCapture の **cm座標**(カメラ基準)で学習されている。
- 実アプリは **正規化スクリーン座標 [0,1]²** を使う。→ 直接は繋がらない。

### 解決策の選択肢
1. **グローバルモデルを y_norm で再学習** (推奨・低コスト):
   cache に y_norm があるので、cm の代わりに y_norm を出力するグローバルMLPを学習。
   出力が正規化スクリーン座標になり、実アプリと同じ空間。デバイス差は個人キャリブが吸収。
   → 要検証: y_norm グローバルが cm 版と同等の頭部ロバスト性を保つか。
2. グローバル出力を中間表現とし、個人キャリブで実スクリーンへ写像 (アフィン)。
3. 合成データ(Blender)で実アプリのカメラ配置に合わせたグローバルを生成 (将来)。

まず選択肢1を実験(`train_global_mlp.py`の目的変数を y_norm に変えるだけ)。

---

## 統合アーキテクチャ

```
features.py: GazeFeatureExtractor
    既存: 2D虹彩特徴を返す
    変更: 7D(or 14D rich)特徴 [Lx,Ly,Rx,Ry,Pitch,Yaw,dist,...] を返すよう拡張
       ↓
calibration.py: CalibrationPipeline
    現行 TargetedPolyCalibration を HybridCalibration に置換 or 併設
    - collect_point() は同じI/F (feat, target を蓄積)
    - finalize() で local Ridge をfit + 事前学習済みグローバルモデルをロード
    - predict() で姿勢ゲート融合 (w=exp(-max(0,m-m_cal)/6))
       ↓
estimator.py: process_frame()
    raw_pred = calibration.predict(feat, head_vec)   # ハイブリッド予測
    その後は既存の OneEuroFilter2D 平滑化そのまま
```

### 必要な変更点
1. `features.py`: 7D(将来14D)特徴を返すメソッド追加 (抽出ロジックは
   `benchmarks/extract_rich_features.py` の `extract_rich` を移植)。
2. `cache/global_mlp_ynorm.joblib`: y_norm出力のグローバルモデルを事前学習して同梱。
3. `calibration.py`: `HybridCalibration` を使うモードを追加 (既存手法とフラグ切替)。
4. `main.py`: キャリブ時に feat をそのまま collect_point へ (target は既存の9点)。

---

## 実アプリでの期待効果

- 9点キャリブ(正面中心)後、ユーザが首を振っても視線が飛びにくくなる。
- キャリブ姿勢付近: ローカルの高精度。姿勢が外れる: グローバルの頑健性。
- 実測検証は `logs/*.csv` のユーザ実データ + 追加収集で行う(座標系を揃えて)。

---

## リスク・課題

- solvePnP頭部姿勢が実webカメラで不安定(実ログで pitch±90°のフリップを確認済)。
  → 姿勢magnitudeのクランプ/平滑化、または頭部姿勢フィルタ(HeadFilter)の強化が必要。
- グローバルの学習ドメイン(スマホ/タブレット前面カメラ)と実環境(PC webカメラ)の差。
  → 合成データ or 実データ微調整で埋める(タスク#3の外部データも活用)。
