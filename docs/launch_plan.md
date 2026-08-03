# Tgaze をどう売り出すか — star 0 の診断と打ち手

> 調査日 2026-08-02。star数・競合数値はすべて GitHub API での実測。
> 憶測と実測を混ぜないため、**実測値**と**判断**を分けて書く。

## 0. 結論(先に)

**star が付かない理由は、品質ではなく「誰も見ていない」こと。**
GitHub に新規リポジトリの自然流入はほぼ存在しない。star は「どこかに投稿された瞬間」に生まれる。
README に動画を足したのは正しい改善だが、**訪問者が0のページを改善しても訪問者は0のまま**。
動画は「来た人を変換する」ための施策であって、「人を連れてくる」施策ではない。

やるべき順序は **(1) 変換率の下地を整える → (2) 露出を作る**。(2)を一度もやっていないのが現状。

---

## 1. 競合の実測（2026-08-02, GitHub API）

| repo | stars | 中身 |
|---|---:|---|
| brownhci/WebGazer | 3,876 | JS/ブラウザ。この分野の巨人。精度は粗い |
| antoinelame/GazeTracking | 2,615 | Python。**画面座標を出さない**（瞳孔位置と左右判定のみ） |
| NativeSensors/EyeGestures | 620 | Python。アクセシビリティ訴求・製品サイトあり |
| Ahmednull/L2CS-Net | 517 | CNN で視線"角度"。画面座標へのマッピングは無い |
| ck-zhang/EyeTrax | 318 | **最も直接的な競合**。Python・画面座標・pip install |
| xucong-zhang/ETH-XGaze | 230 | データセット/モデル |

**ここから読み取れること**:
- GazeTracking は Tgaze より機能が少ないのに 2,615 star。**star は精度ではなくパッケージングと露出で決まる**。
- このニッチの現実的な天井は数百 star。**近期の現実的目標は 50〜300 star**であって 1,000 ではない
  （1,000超はWebGazer/GazeTrackingの2件のみ＝長年の被引用と露出の蓄積）。
- 「画面座標を実用精度で出す Python ライブラリ」は EyeTrax(318) がほぼ唯一。**ここに空きがある**。

## 2. 最近の技術動向（2026）と Tgaze の位置

- 主流は appearance-based CNN（ETH-XGaze系）。精度は高いが **GPU前提・出力は視線角度**で、
  「画面のどこ」に落とすには結局ユーザ較正が要る。→ **Tgaze が解いている問題は消えていない**。
- 2026 の同じ土俵の論文: **EMC-Gaze**（arXiv 2603.12388, ランドマークのみ+セッション毎較正）。
  MPIIFaceGaze LOPO 16-shot で **8.82° (subject-macro RMSE)**。
  → Tgaze は同プロトコルで **5.73°**（`experiments/REPORT8_metrics.md`）。
  ⚠️ ただし論文側は cross-dataset 転移の可能性が残り、**単純な優劣比較はできない**。
  「GPU無しの16D線形がこの水準に到達」までが言える範囲。
- 応用側の伸び: **アクセシビリティ(AAC/ALS, OptiKey系)**、UXリサーチ/アテンション解析、リモート実験。
  → **アクセシビリティが最も感情に訴え、コミュニティが実在する**（後述の awesome-eye-tracking 等）。

## 3. 変換率の下地（訪問者を star に変える）

実測した減点と、その状態。

| # | 問題 | 影響 | 状態 |
|---|---|---|---|
| 1 | リポジトリ名 `my_gaze_project` | 検索されない。習作に見える | ✅ 2026-08-03 `tgaze` へ改名（旧URLは自動リダイレクト） |
| 2 | GitHub の About(description) が**空** | 検索順位・第一印象に直結 | ✅ 2026-08-03 設定。トピックも5→12個へ |
| 3 | LICENSE ファイルが無い(=「No license」表示) | 企業/慎重な開発者が避ける | ✅ 2026-08-02 追加 |
| 4 | インストール手段が無い | 競合は `pip install eyetrax` | 🔶 `tgaze` パッケージAPI追加済。PyPI公開は次段 |
| 5 | 5行で動くサンプルが無い | 「試すコスト」が高い | ✅ `examples/quickstart.py` 追加 |
| 6 | リポジトリ容量 171MB | clone が重い | 要判断（過去コミットの `cache/*.npz` 142MB。履歴書換=force push） |
| 7 | 非商用ライセンス(CC BY-NC-SA) | **star を減らす**。企業利用を排除 | **ユーザの経営判断**（商用化を狙うなら妥当なトレードオフ） |

### 1 と 2 の具体的な文言（そのまま使える）

- **リポジトリ名**: `tgaze`
- **Description**:
  > Accurate webcam eye tracking that outputs where you look **on screen** — CPU-only, ~2.6 cm after a 9-point calibration.
- **Topics 追加候補**: `eye-tracker` `gaze` `computer-vision` `opencv` `accessibility` `scikit-learn` `webcam`
- **Website欄**: デモ動画のURL

## 4. 露出を作る（ここが本丸・一度もやっていない）

**投稿の中身より「動くGIFが最初の3秒で見えるか」が効く。** 動画は既にある＝素材は揃っている。

| 場所 | 狙い | 備考 |
|---|---|---|
| **Zenn / Qiita（日本語）** | 最も転換率が高い。ユーザの主戦場 | 「webカメラだけで視線を2.6cmまで追い込んだ話」＋負の結果も書くと刺さる |
| **Show HN** | 単発で最大。ただし一発勝負 | タイトル例: `Show HN: Tgaze – webcam eye tracking that outputs screen coordinates (CPU-only)` |
| **r/Python / r/computervision** | 安定して数十star | 週末の朝(米国東部)が伸びやすい |
| **X / Twitter** | 20秒動画。研究者アカウントに届く | MediaPipe/gaze 界隈にメンション |
| **awesome 系へPR** | 継続的な流入 | `codeberg.org/eyes-on-disabilities/awesome-eye-tracking`、awesome-mediapipe 等 |
| **アクセシビリティ系コミュニティ** | 最も強い動機を持つ層 | OptiKey/AAC 界隈。「視線でカーソル」は実需 |

## 5. 訴求すべき差別化（すべて検証済の事実のみ）

1. **画面座標を直接出す** — L2CS/ETH-XGaze は角度どまり、GazeTracking は左右だけ。
2. **GPU不要・CPUリアルタイム** — 導入障壁が桁違いに低い。
3. **較正が軽い** — 9点で個人 2.6cm。較正ゼロでも他人に約 4.2cm/6.2°。
4. **評価が正直** — 負の結果(exp62リンバス楕円)も、自分の指標バグ(縦1.68倍の水増し)も公開している。
   **この分野では希少**で、研究者・実務者の信頼を最も強く買う。ここは隠さず前面に出す。

## 6. やらない方がいいこと

- star を買う/相互star。バレるし、質の高い流入が死ぬ。
- 数字の誇張。「SOTA越え」と書くと、条件の違い（cross-dataset 疑い）を突かれた瞬間に信用を全部失う。
  **Tgaze の最大の資産は正直さ**なので、ここを削ってはいけない。
