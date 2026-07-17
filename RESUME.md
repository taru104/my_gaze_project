# RESUME — 続きからやるための単一の真実ソース

> **これは何**: セッションが切れても「続きからやって」で即再開できるための引き継ぎファイル。
> Claude はまずこのファイルを読めば、いま何をしていて次に何をすべきか完全に把握できる。
> **最終更新: 2026-07-17（✅H1本番[7D]実機で正面「超絶正確」を体感確認。mモード/姿勢矢印/操作説明追加。目標=GitHub star1000。ハッカソン31日）**
> 状態が変わったら Claude はこのファイルを必ず更新すること。詳細な実験履歴は `results/research_log.md`。

---

## 0. 次に起動したClaudeへ（最初にやること）

1. このファイル全体と `results/research_log.md` の末尾を読む。
2. **いま走ってるバックグラウンド抽出があるか確認**:
   ```bash
   powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select ProcessId,CommandLine | Format-List"
   ```
   - `extract_rich_features` が居れば本抽出が継続中。居なければ §3 の手順で再開。
3. **チェックポイント進捗を確認**:
   ```bash
   .venv/Scripts/python.exe -c "import numpy as np; d=np.load('cache/rich_features_checkpoint.npz'); print('frames:', len(d['X']), 'dim:', d['X'].shape[1])"
   ```
   - `cache/rich_features_cache.npz`（最終成果物）が在れば抽出は**完了済** → §4 の評価へ。
4. §4 の「次にやること」を上から実行。ユーザには yes/no を聞かず自律で回す（本人の明示指示）。

---

## 1. プロジェクトの目的とユーザ最重要要件

- **目的**: webカメラ視線推定の精度を現行パイプライン超えに引き上げる。
- **ユーザ最重要要件**: **顔を横に向けても視線がズレないこと（頭部姿勢ロバスト性）**。
  実ログ(logs/*.csv)では |pose|>20° が20〜81%と横向き利用が多い＝実際に効く。
- 指標: `Euc(cm) median`（低いほど良い）。頭部姿勢bin別に層別評価する。
- 作業スタイル: **自律で回す。yes/noを聞かない。理解/達成できるまでループ**（ユーザ明示指示）。
  外部データDLや新規パッケージ導入だけはタスクに残して起床後承認をもらう。

## 2. いまの勝ち筋（達成済みベスト）

**16D 姿勢ゲート・ハイブリッド**（新ベスト, 2026-07-16。本番 `HybridCalibration` で検証済）。

| pose | 現行ローカル | 7D hybrid | **16D rich hybrid** |
|------|------|------|------|
| 正面0-10° | 4.07 | 3.06 | **2.31** |
| 15-20° | 5.90 | 4.11 | **2.90** |
| 20-25° | 6.10 | 4.03 | **2.88** |
| 30-40° | 8.00 | 4.45 | **3.72** |
| overall | 6.679 | 4.508 | **3.339** |

→ 16Dは現行ローカル比 **50%改善**、7Dハイブリッド比 **26%改善**、横向きほど差が大きい。

**レシピ**: 16D特徴 → [ローカルRidge(正面キャリブ) + グローバルMLP(219k多姿勢学習, 16D)] を
姿勢ゲート `w=exp(-max(0,m-m_cal)/6)` でブレンド（tau=6, alpha=10）。
本番モデル `cache/global_mlp_16d.joblib`。`HybridCalibration` は次元非依存＝16D特徴＋16Dモデルでそのまま動く。

**検証済みの重要仮説**: 豊富な特徴は「グローバル学習」でのみ効く
（subject-held-out CV: 7D 5.29cm → 486D 3.77cm。今回16Dで decisive に確定）。
ローカルキャリブでは過学習して逆効果。だから厳選した少数の豊富特徴を全データで再抽出した。

## 3. 直近完了した実験（2026-07-16）— ✅ 16D 完遂・本番検証済

**7D → 16D 厳選リッチ特徴への拡張を全324kフレームで再抽出 → グローバル学習 → 決定的勝ち → 本番反映まで完了。**
成果物: `cache/rich_features_cache.npz`(train261k), `cache/rich_test_cache.npz`(test32k),
`cache/global_mlp_16d.joblib`(+.meta.json)。検証: `benchmarks/validate_hybrid_16d.py` → overall 3.339cm再現[OK]。
（以下は抽出の再開手順。再抽出が必要になった時の参照用に残す）

16D の内訳（`benchmarks/extract_rich_features.py` の `extract_rich`）:
```
[0:7]  Lx,Ly,Rx,Ry,Pitch,Yaw,dist        （既存7D）
[7]    roll                               （頭部ロール）
[8:10] L_EAR,R_EAR                        （眼開き; 残差分析でPitch方向の弱点に効く狙い）
[10:12] L_ivert,R_ivert                   （虹彩の眼内垂直位置; 縦視線の弱点狙い）
[12:14] L_idiam,R_idiam                   （虹彩径/眼幅; 距離代理）
[14:16] L_aspect,R_aspect                 （虹彩アスペクト比; NotebookLM最推奨・横向き扁平化を捉える）
```

### 進捗・再開方法
- チェックポイント: `cache/rich_features_checkpoint.npz`（`CK_EVERY` 毎に自動保存）。
- **抽出が止まってたら再開**（チェックポイントから自動継続する）:
  ```bash
  .venv/Scripts/python.exe benchmarks/extract_rich_features.py    # run_in_background で
  ```
  起動時に `[Resume] N frames, start~M` が出れば継続成功。約70fps、全体ETA約70分。
- 完了すると `cache/rich_features_cache.npz`（train用、split_code付き）が生成される。
- **test側キャッシュも必要**（被験者別ハイブリッド評価用、subj_id付き）:
  ```bash
  .venv/Scripts/python.exe benchmarks/extract_rich_test.py        # run_in_background で
  ```
  出力 `cache/rich_test_cache.npz`。※これはチェックポイント無し（最後に一括保存）。
  `extract_rich` を共有 import しているので**自動的に16D**（docstringの14Dは古い表記）。

## 3.5 ⚠️ 実カメラ検証の結果（2026-07-16 夕）— 勝ち筋が転移しなかった

ユーザ実録画 `logs/session_20260716_130217`（生ランドマーク→`reprocess_raw_landmarks.py`で16D化、
正解付き2172フレーム/9点）を **点ごと leave-one-point-out** で評価（`benchmarks/eval_live_16d.py`）:

| 手法 | median Euc(cm) |
|------|------|
| A) 7D + ローカルRidge（現行アプリ相当） | **4.10** |
| B) 16D + ローカルRidge | 4.05（≒差なし） |
| C) 16D globalMLP prior + アフィン補正 | **10.88** ← ほぼ使い物にならない |
| C2) 16D + prior_cm + ローカルRidge | 4.32（むしろ悪化） |
| baseline) 常に画面中央 | 12.36 |

**結論: `cache/global_mlp_16d.joblib` は実webカメラにそのまま転移しない。**
GazeCapture(スマホ前面カメラ・近距離)とPC webcam(≈50cm)のドメイン差が原因と思われる。
§2 の 3.339cm は GazeCapture 上の値であって、実カメラの値ではない。**混同しないこと。**
→ §4 の「16Dモデルをアプリに接続すれば良くなる」という前提は**崩れた**。接続しても良くならない。

### 🔥 さらに重大: アプリ自身のキャリブが plain Ridge より3倍悪い（2026-07-16夕 追加検証）

`calibration._compute_loo` は**点ごとLOO**（9点中1点を除外→held点の予測中央値と正解の誤差→
9点の|err_x|,|err_y|平均→`main.py`が`sqrt((ex*30.9)^2+(ey*17.4)^2)`）。
**この定義に完全に合わせて**同じデータで測り直すと:

| モデル（アプリと同一指標・同一データ） | loo_euc_cm |
|---|---|
| **アプリ本番 `TargetedPolyCalibration`** | **9.128** ← 今日のHUD実測値 |
| 7D + ローカルRidge | **3.120** |
| 16D + ローカルRidge | 3.435 |

### 🎯 犯人確定: `features.py` の X_feat/Y_feat が「視線」でなく「顔の画面内位置」を測っている

**過学習ではなかった。** `benchmarks/diagnose_local_calib.py` でモデルを総当たりしても
アプリ特徴では**全部9.6〜10.0cmに張り付く**（＝モデルは無関係）:

| アプリ特徴 `[X_feat,Y_feat,pitch]` 上のモデル | loo_euc_cm |
|---|---|
| TargetedPolyCalibration（アプリ本番） | 9.609 |
| PolyRidgeCalibration（2次10param・より複雑） | 9.633 |
| AffineCalibration（線形・最小） | 9.856 |
| plain Ridge（2特徴のみ） | 10.001 |

過学習ならモデルを単純化すれば改善するはずだが、**むしろ単純な方が悪い＝過学習ではない**。

**真因**（`features.py:302`）:
```python
cx, cy = w / 2.0, h / 2.0            # ← 画像の中心
X_feat = ((L_cx - cx) / L_diam + (R_cx - cx) / R_diam) / 2.0
```
虹彩直径で割るので**距離不変にはなっている**が、基準点が**画像中心**なので
**顔を平行移動しただけで視線が動かなくても X_feat が動く**。解くべき不変性を間違えている。
7D/16D 側の `_geo_normalize` は虹彩を**目頭・目尻の中点**基準・目の軸で回転・目幅で正規化＝
「眼球の中で虹彩がどこにあるか」＝本当の視線信号で、頭の平行移動に不変。

`benchmarks/prove_feature_bug.py`（同一生ランドマークから両特徴を計算＝特徴以外完全同一）:

| 特徴 | 顔の画面内位置(鼻先)との相関 | 正解の視線との相関 |
|---|---|---|
| **アプリ X_feat** | **0.976** | 0.478 |
| 目頭基準 Lx | 0.141 | 0.682 |
| **アプリ Y_feat** | **0.546** | **0.123** ←視線信号がほぼ無い |
| 目頭基準 Ly | 0.229 | 0.799 |

→ アプリの X_feat は実質**顔位置センサー**。同一モデル(plain Ridge)・同一指標での直接対決:
アプリ方式2D **10.051cm** vs 目頭基準4D **4.563cm**。

**次元数の問題ではない**: 片目2D `[Lx,Ly]` だけでも 4.503cm でアプリの2Dの倍以上良い。
中身が正しいかどうかの問題。（ローカルで 7D 3.120 < 16D 3.435 なのは §2 の既知の性質と整合）

### 測り方の注意
- 指標が3種類あって混同しやすい。比較時は必ず定義を揃えること:
  (a) アプリHUD `loo_euc_cm`＝点ごとLOO・点内中央値・軸別平均→合成、
  (b) 点ごとLOOのフレーム単位 median Euclidean（`eval_live_16d.py`の表示: 7D 4.10cm）、
  (c) ランダムk-fold＝**甘い**（同一点の相関フレームが訓練に混じる。7D 2.81cm と出る）。
- ~~アプリHUDはフレーム単位LOO~~ は**誤り**（2026-07-16夕に訂正）。アプリも点ごとLOOである。
- 実測の弱点は隅と横向き: point6(0.9,0.1) 6.12cm / point2(0.1,0.9) 5.81cm。
  |yaw| 20-30° 5.92cm, 30°+ 5.78cm（正面0-10°は4.16cm）。

## 4. 次にやること（この順で自律実行）

### ✅ 完了（2026-07-17）: 7Dローカルを本番実装
- `rich16d.py` 新設（16D特徴の単一定義。ライブ/再処理/評価が共有）。
- `features.extract` → 7D出力（`rich_16d_from_lms[:7]`、目頭基準）。X_feat/Y_feat等はdebug/ログに残す。
- `calibration.py` → `RidgeCalibration`(次元非依存) に差し替え。`TargetedPolyCalibration`廃止。
- `estimator.py` → 頭部姿勢ゲート解除（フリーズ廃止）。`main.py` → タップ正解をCSV/生ログに保存。
- 検証 `benchmarks/verify_app_pipeline_7d.py`: アプリ`CalibrationPipeline`が7Dで **loo_euc_cm 3.120cm**
  （旧9.128cm）。import/init/calib/predict の sanity 全通過。詳細は `research_log.md` 2026-07-17。

### 🔴 実機で要確認（ユーザと一緒に）
- 実webカメラで9点キャリブし直し、HUDの `loo_euc_cm` が3cm台になるか。オフライン1セッションのみの検証なので実測必須。
- 毎フレーム solvePnP を2回通る（features自前 + rich16d内）ので fps 低下がないか。落ちるなら rich16d に姿勢を渡して共用化。

### ✅ 探索結論(2026-07-17): Huber を本番採用、アプリ実効 loo_euc_cm 2.08cm 確定
方針: **次元は増やさない（ユーザ指示）。7D以下で中身を詰める。**
- **7D + Huber が最良**。アプリ経路 loo_euc_cm = **2.084cm**（Ridge 3.120 / 旧アプリ実測 9.128 から4.4倍改善）。
  `calibration.HuberCalibration` を本番採用済(`CalibrationPipeline`)。検証 `verify_app_pipeline_7d.py`。
- **小細工は悪化**(外れ値除去/per-eye/アンサンブル/RBF全frame/後処理)。効く: Huber, 9点集約, 虹彩4D 2次, 姿勢ゲート。

### 🎯 1cmの芽(段階4-6): H4 yawゲートブレンド ※要追加データ検証、まだアプリ未搭載
- **虹彩4D 2次多項式(9点集約)=正面1.08cm だが横向き30+で19.78cm崩壊**(姿勢を入れないため)。単体では不可。
- **H4 = 正面:虹彩4D2次 + 横向き:7D Huber を yawゲート w=exp(-|Δyaw|/8°) でブレンド** →
  実効**1.07cm** かつ 30+ 5.25cm(崩壊せず)。「1cm と 横向きロバスト」を両立。frame median 2.63で全手法最小。
- **⚠️過適合リスクで未搭載**: 1セッション・tau=8°チューニング・正面n=101/横向きn=254と少。
  → 追加データで再現したら `HybridPolyCalibration` として実装。`benchmarks/explore_hybrid_pose.py` が H4 実装の雛形。
- **1cmへの本命=キャリブ点数**: 3点11.4→7点3.8→9点3.6cm。点数増＝追加録画。手法だけでは1セッション9点が限界。
- 制約: 実カメラ用データは1セッション(本人9点2172フレーム)のみ。過去CSVは7D再計算不可(生ランドマーク無し)。
- 🔁**監視ループ稼働中**: `benchmarks/watch_new_sessions.py` が新録画を自動で16D化→7D Huber評価→全セッション合算精度。
  ユーザが main.py で追加録画すれば結果が `results/exploration_log.md` に自動追記される。

### 🧱 段階7で判明した横向きの真の壁（最重要・要ユーザ対応）
- H4のゲートtauは4-24°で~1cmと**頑健(過適合でない)**。tau=10°でサブcm(0.99cm)。
- **だが時間分割ホールドアウト(姿勢が時間変動)では M2もH4も横向き20-30°で12cm・30+で20cmに崩壊**。
  点ごとLOO(同姿勢の空間内挿)の4-5cmと大差。→ **横向きの壁はモデルでなく「虹彩/姿勢推定の時間的不安定性」**。
- **実利用示唆**: キャリブ時と頭部姿勢が変わると横向きが大きく劣化。正面固定なら1cm、頭を振ると崩れる。
- **横向き1cmの処方箋(データ/特徴側、要ユーザ)**:
  1. 多姿勢キャリブ(横向き状態でもキャリブ点を取る)。2. E4 solvePnP安定化。3. 虹彩楕円フィッティング。
- **回帰手法の探索は段階1-7で飽和**。以降は追加データと特徴改良が主戦場。

### その次（保留中の長期テーマ）
- **正解データを増やす**: タップ正解保存を実装済(2026-07-16)。通常利用でデータが貯まる設計に。
- **ドメインギャップ**: global MLP は実カメラに転移せず(§3.5)。本人データでの学習/FTが要る。データ量が全く足りない。
- E2虹彩楕円 / E4 solvePnP安定化 / 合成データ(Blender)。※ただし「次元を増やす」系は当面保留（ユーザ指示）。

<details><summary>旧メモ（16D統合計画・前提が崩れたので保留）</summary>

1. **ライブ特徴を16D化**: `features.GazeFeatureExtractor.extract` を `extract_rich` と同一の16Dを
   出すように拡張（既存部品を流用。mediapipeランドマークからの計算は extract_rich が参照実装）。
2. **estimatorにハイブリッドを接続**: `GazeEstimator`/`CalibrationPipeline` に
   `HybridCalibration(joblib.load('cache/global_mlp_16d.joblib'))` を組み込む。9点キャリブ→`add/fit`、
   毎フレーム→`predict`。
3. **座標系ブリッジ**: `global_mlp_16d` はGazeCapture cm(デバイス非依存)を出力。アプリは正規化[0,1]。
   二段構成: 「グローバル=cm prior → 個人キャリブで cm→画面座標アフィン」。
   設計 `docs/app_integration_plan.md`。y_norm直接グローバルは却下済(デバイス依存で実用不可)。
4. **ユーザ実キャリブデータで検証**: `logs/*.csv` の calibレコード(3セッション766〜1632点)で
   実カメラ精度を測る。実データはpitch±80-90°の非現実値・solvePnPフリップに注意(下記E4)。
   ※この段階は実webカメラでの反復が要る＝ユーザが起きている時に一緒にやるのが効率的。

**並行して精度をさらに詰める実験**（`results/next_experiments_from_notebooklm.md`）:
- E2: 虹彩楕円フィッティング特徴（16Dに追加候補。横向き強化）。
- E4: solvePnP頭部姿勢の安定化（実ログのpitch±80-90°フリップ対策。実データ精度の底上げ）。
- 合成データ（Blender）で横向き・極端姿勢を増やす: `docs/synthetic_gaze_dev_plan.md`。

**[起床後に承認が要るもの]** 外部データセットDL(ETH-XGaze等)、追加パッケージ(PyTorch等)。
</details>

## 4.5 ユーザ本人にしか出来ない開発（前AIからの引き継ぎ＝要お願い事項）

Claude だけでは進められず、ユーザの手が要る部分。壁にぶつかったらここに追記する。
1. **実webカメラでの反復検証**（最重要）: 精度は実カメラでキャリブし直さないと確定しない。
   オフラインは1セッションのみ。ユーザが実機でキャリブ→タップ利用してデータを増やす必要がある。
   → **追加録画のお願い**: 色々な頭部姿勢(特に横向き±30°超)・距離・時間帯で9点キャリブ+通常利用。
   生ログ`logs/session_*_landmarks.bin`が貯まるほど探索が効く。今は本人1人9点2172フレームだけ。
2. **外部データDL・パッケージ導入の承認**: ETH-XGaze等の横向き豊富データDL、PyTorch等の追加は
   ユーザ承認が要る（容量・環境影響のため）。合成データ(Blender)方針も本人GO待ち。
3. **物理計測**: 画面の物理サイズ(現状 30.9×17.4cm 決め打ち)、カメラ位置、視距離。
   cm換算精度に直結するので、実機の実値をユーザに確認してもらうと誤差評価が正確になる。
4. **主観評価**: 「実際に使ってカーソルが目に追従して快適か」は数値と別。本人の体感フィードバックが要る。

## 5. ファイル地図

| 種別 | パス | 内容 |
|------|------|------|
| **研究ログ** | `results/research_log.md` | 全実験の詳細履歴（読めば経緯が全部分かる） |
| **16D単一定義** | `rich16d.py` | ライブ/再処理/評価が共有する16D特徴。`rich_16d_from_lms` |
| **本番キャリブ** | `calibration.py` | `RidgeCalibration`(7D/16D非依存) + `CalibrationPipeline` |
| **7D検証** | `benchmarks/verify_app_pipeline_7d.py` | アプリ経路で7Dが3.12cm出るか |
| **精度探索** | `benchmarks/explore_accuracy.py` | 1cm目標の手法総当り→`results/exploration_log.md` |
| 犯人特定 | `benchmarks/diagnose_local_calib.py` `prove_feature_bug.py` `simulate_feature_fix.py` | §3.5の証拠 |
| 本番実装 | `hybrid_calibration.py` | 勝ち筋 `HybridCalibration`（姿勢ゲート融合） |
| 特徴抽出(train) | `benchmarks/extract_rich_features.py` | 16D全フレーム抽出・チェックポイント対応 |
| 特徴抽出(test) | `benchmarks/extract_rich_test.py` | test被験者16D・subj_id付き |
| 評価 | `benchmarks/rich_hybrid_eval.py` | 7D vs 16D グローバル比較（16D動的対応済） |
| 本番モデル学習 | `benchmarks/train_global_16d.py` | 勝った16D global MLPを保存→`cache/global_mlp_16d.joblib` |
| 本番検証 | `benchmarks/validate_hybrid_16d.py` | 本番`HybridCalibration`で16Dが3.339cm再現するか検証 |
| **実カメラ評価** | `benchmarks/eval_live_16d.py` | ユーザ実録画で点ごとLOO評価（§3.5の数字はこれ） |
| 生ログ | `raw_landmark_logger.py` | 全478ランドマーク(x,y,z)+ターゲットをフレーム毎に追記保存(main.py統合済) |
| 生ログ再処理 | `benchmarks/reprocess_raw_landmarks.py` | 生ログ→任意次元特徴を再計算。現状16D。実画像で数値一致検証済 |
| 次の実験計画 | `results/next_experiments_from_notebooklm.md` | NotebookLM知見の実験案 |
| 相談用プロンプト | `results/notebooklm_briefing.md` | NotebookLMに追加相談する時の材料 |
| 合成データ計画 | `docs/synthetic_gaze_dev_plan.md` | Blender合成データ開発構想 |
| アプリ統合設計 | `docs/app_integration_plan.md` | 座標系課題・二段構成設計 |

### キャッシュ（cache/）
- `sota_7d_cache.npz` — test 26被験者の7D。 `gazeCapture_features_cache.npz` — train 264k。
- `rich_features_checkpoint.npz` / `rich_features_cache.npz` — 16D抽出の途中/完成。
- `rich_test_cache.npz` — test被験者の16D（生成予定/済）。
- `global_mlp*.joblib` — 学習済みグローバルMLP。

## 5.5 データ記録の方針（2026-07-16 追加）

**教訓**: 以前は加工後の値(pitch/X_feat等)しかログしておらず、後から16D等を作りたくても
再計算できず詰んだ。→ **今後は生ランドマークを丸ごと残す**。
- ライブアプリ(`main.py`)は起動毎に `logs/session_<id>_landmarks.bin`(+`.meta.json`)へ
  **全478ランドマーク(x,y,z)+画像サイズ+正解ターゲット**をフレーム毎に追記(クラッシュ耐性・各flush)。
- 人間可読CSV(`session_<id>.csv`)は従来どおり併存。両者は `frame_idx` で対応。
- 将来どんな高次元特徴が必要になっても、`reprocess_raw_landmarks.py` で過去録画から再計算できる。
  実画像で `extract_rich`(画像版)と数値一致を検証済 → 生データの十分性は確認済。
- 1フレーム≈5.7KB(478×3×4B+ヘッダ)。長時間録画でサイズは増えるが、生の価値を優先。

## 6. 環境の注意（ハマりどころ）

- **Python は必ず** `.venv/Scripts/python.exe`（相対パスでOK、cwdはプロジェクト直下）。
- **PowerShellで `cd`+コマンド連結は許可拒否される** → 絶対パスで直接実行 or Bashツール利用。
- 長いPythonコードは一旦 `.py` に書き出してから実行（コマンド長制限）。
- **stdoutのUTF-8化は `sys.stdout.reconfigure(encoding="utf-8")` を使う**。
  `io.TextIOWrapper(sys.stdout.buffer, ...)` で包み直すと元stdoutのGC時に下層バッファが
  閉じられ `I/O operation on closed file` でバックグラウンド実行が落ちる（2026-07-16に遭遇・修正済）。
  ※ `extract_rich_features.py` はまだ旧パターン(line22)。次に止まったら reconfigure に直すと安全。
- 長時間ジョブは必ず `run_in_background: true` で起動し、チェックポイント対応にする。
- 落ちても状態を失わないよう、抽出はチェックポイント、実験結果は `research_log.md` に逐次記録。

## 7. このファイルの運用ルール（Claudeへ）

- 状態が進んだら（抽出完了・比較実行・新ベスト更新など）**必ずこの RESUME.md を更新**する。
- 「最終更新」の日付と §3/§4 を最新の実態に合わせる。
- 新しい確定知見は `research_log.md` に、次アクションはここ §4 に反映する。
- こうしておけば、ユーザが「続きからやって」と言うだけで即座に再開できる（今回の混乱を二度と起こさない）。
