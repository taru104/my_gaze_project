# 投稿用ドラフト（そのまま出せる状態）

> `docs/launch_plan.md` の「露出を作る」の実行物。数値はすべて 2026-08-02 時点の実測。
> **✅ 2026-08-03 完了**: リポジトリ名を `tgaze` に変更 / About欄を設定 / トピック12個を設定。
> 旧URL `my_gaze_project` はGitHubが自動リダイレクトする。**投稿の下地は整った**。
> 残るは デモGIFの用意 → 投稿。

---

## A. Zenn / Qiita（日本語・最優先）

**タイトル案（どれか1つ）**
1. `webカメラだけで視線を追う精度を上げ続けたら、自分の評価指標がバグっていた話`
2. `GPUなしで視線推定 5.73° — 66回の実験でわかったこと（負の結果込み）`
3. `「体感は良いのに数値は悪い」をどう扱ったか — webカメラ視線推定の開発記`

→ **1 を推奨**。技術記事は「失敗の告白」が最も伸びる。かつ内容が本当にそれ。

**本文構成**

```markdown
## 作ったもの

webカメラだけで「画面のどこを見ているか」を推定する Tgaze を作っています。
9点キャリブ後で約2.6cm、GPU不要でリアルタイム。

（ここにデモ動画/GIF）

既存のOSSは「左を見てる/右を見てる」か「視線の角度」までで止まるものが多く、
"画面上の座標" を実用精度で出すものは意外と少ない。そこを狙っています。

## 1. 最初の犯人は「次元」でも「モデル」でもなかった

精度が9cmから改善しなくて、モデルを総当たりしました。線形、多項式、Ridge、
どれも9.6〜10.0cmに張り付く。**単純なモデルほど悪い**——過学習なら逆のはずです。

原因は特徴量でした。虹彩の位置を**画像の中心**基準で正規化していたんです。

```python
cx, cy = w / 2.0, h / 2.0        # ← 画像の中心
X_feat = ((L_cx - cx) / L_diam + (R_cx - cx) / R_diam) / 2.0
```

これだと**顔を平行移動しただけで、視線が動いていなくても値が動く**。
実際に相関を測ったら、顔の画面内位置との相関 0.98、肝心の視線との相関 0.48。
視線センサーのつもりが顔位置センサーでした。

目頭・目尻の中点を基準にして目の幅で正規化したら、9cm → 1.4cm。
**解くべき不変性を間違えていた**だけでした。

## 2. 数値と体感が食い違ったとき、どちらを取るか

目の画像パッチを特徴に足したら、点精度の数値は**わずかに悪化**しました(1.63→2.07cm)。
数値だけ見れば却下です。でも実際に使うと明らかに安定している。

そこで両方のログを取って「何が違うのか」を測り直しました。犯人は
**カーソルが画面外に吹っ飛ぶ頻度**でした。11.9% → 6.3%。ほぼ半減。

点精度という単一の指標は、"たまに大暴走する" を評価できていなかった。
体感の方が正しかったので、画像版をデフォルトにしました。

## 3. そして、自分の指標がバグっていた

MPIIFaceGaze で評価するとき、正規化誤差をcmに直していました。

```python
err_cm = sqrt(dx_norm**2 + dy_norm**2) * 30.0   # 画面幅30cm
```

**両方の軸に30を掛けている。** 実際の画面は 286.5 × **179.0** mm です。
縦方向の誤差を 30/17.9 = **1.68倍に水増し**していました。

軸ごとに正しい実サイズで換算し直したら、報告していた 4.21cm は実際には **3.19cm**。
1年近くこの指標で意思決定していたことになります。
幸い過小評価の方向でしたが、逆だったら全部やり直しでした。

**誰も監査しない指標は、静かにあなたの信念を決める。**

## 4. 「度」で測って、ようやく文献と比較できた

cmは画面サイズ依存なので論文と比べられません。MPIIの3D注釈と monitorPose を使い、
予測した画面座標をカメラ座標系に逆投影して角度誤差を出しました
（逆投影の正しさは、正解の3D注視点と 0.0mm 一致することで確認）。

| プロトコル | 誤差 |
|---|---|
| 同一人物・較正あり | 3.19 cm / 3.65° |
| 他人・較正ゼロ | 4.19 cm / 6.24° |
| 他人・16サンプル較正 | 3.51 cm / 5.73° |

同じ土俵（ランドマークのみ・少数較正）の2026年の論文 EMC-Gaze が
MPII LOPO 16-shot で 8.82°。こちらは 5.73° でした。

ただし**「SOTAを超えた」とは書きません**。論文側は自前データで学習したモデルを
MPIIに転移した可能性が読み取れ、その場合こちらの方が条件が有利です。
言えるのは「GPUなしの16次元線形モデルがこの水準に到達した」までです。

## 5. 効かなかったもの（こっちが本題かもしれない）

- **リンバス（虹彩輪郭）楕円フィット**: 放射状レイ＋サブピクセル勾配で虹彩を測り直したが、
  MediaPipeの学習済みランドマークに負けた。理由が面白くて、MediaPipeの虹彩中心は
  目頭・目尻と**同相に揺れる**（相関0.997）ので、目頭基準で正規化した瞬間に
  揺れが相殺される。毎フレーム独立に測る楕円中心はこの恩恵を失う。
- **解像度を上げる**: 24×16 → 48×32 で改善、48×32 → 96×64 で**平坦**。手作り特徴の天井。
- **姿勢の交互作用項・輻輳角**: 過学習 or 定義上ゼロ情報で不変。

負の結果を出さない研究記事は信用できないので、全部リポジトリに置いています。

## リポジトリ

https://github.com/taru104/tgaze

（スター貰えると続ける励みになります）
```

---

## B. Show HN

**タイトル**（80字以内・"Show HN:" を含める）
```
Show HN: Tgaze – webcam eye tracking that outputs screen coordinates, CPU-only
```

**本文（最初のコメントとして自分で投稿する）**
```
I've been building a webcam gaze tracker that outputs the point on the screen you're
looking at, rather than a gaze angle or a "looking left/right" label. It runs on CPU
in real time (MediaPipe landmarks + a 16-D geometric feature + an eye-image patch
compressed to 16 PCA components, then robust linear regression).

After a 9-point calibration it lands around 2.6 cm on screen. On MPIIFaceGaze with
leave-one-person-out and 16 calibration samples it gets 5.73° subject-macro angular
RMSE; with zero calibration, 6.24°.

Two things I found along the way that might be more interesting than the numbers:

1. An early feature normalized the iris against the image center, which quietly made
   it a face-position sensor — correlation 0.98 with head location, 0.48 with actual
   gaze. Normalizing against the eye corners instead took center error from ~9 cm to
   ~1.4 cm. The model was never the problem.

2. My own cm metric was wrong. I scaled normalized error by a flat 30 cm on screens
   that are 28.6 x 17.9 cm, inflating vertical error by 1.68x. Fixing it moved a
   reported 4.21 cm to an actual 3.19 cm. I'd been making decisions on that number
   for months.

The negative results are in the repo too (limbus ellipse fitting loses to MediaPipe's
learned landmarks; eye-patch resolution stops paying off past 48x32).

Note on licensing: it's CC BY-NC-SA, so noncommercial only — I'd rather say that
up front than have someone find out later.
```

---

## C. Reddit（r/Python, r/computervision）

**タイトル**
```
I built a webcam eye tracker that outputs screen coordinates (CPU-only, ~2.6cm after calibration)
```

**本文**
```
Most open-source webcam gaze projects give you a gaze angle or a left/right label.
I wanted the actual point on the screen, so I built Tgaze.

- ~2.6 cm on screen after a 9-point calibration
- ~4.2 cm / 6.2° for a person it has never seen, with zero calibration
- CPU real-time, no GPU, nothing leaves your machine
- MediaPipe landmarks -> 16-D geometry + eye-image patch (PCA-16) -> Huber regression

```python
from tgaze import GazeTracker
tracker = GazeTracker()
tracker.calibrate()
x, y = tracker.predict(frame)
```

The part I'd actually like feedback on: my point-accuracy metric and my subjective
experience disagreed. Adding the eye-image patch made the accuracy number slightly
worse but the tracker clearly steadier. Measuring properly showed it halved the rate
of wild off-screen jumps — something the accuracy number couldn't see. I shipped the
version that felt better. Curious how others handle metric-vs-feel disagreements.

Repo (negative results included): https://github.com/taru104/tgaze
It's CC BY-NC-SA (noncommercial), flagging that up front.
```

---

## D. X / Twitter（スレッド）

```
1/ webカメラだけで「画面のどこを見ているか」を推定する Tgaze を作っています。
9点キャリブで約2.6cm、GPU不要・リアルタイム・ローカル完結。
[動画]

2/ 精度が9cmで止まっていた原因はモデルではなく特徴量でした。
虹彩を「画像の中心」基準で正規化していて、実質は顔位置センサーだった。
顔位置との相関0.98、視線との相関0.48。
目頭・目尻基準に変えたら 9cm → 1.4cm。

3/ さらに、自分の評価指標がバグっていました。
正規化誤差の両軸に画面幅30cmを掛けていたが、実画面は28.6×17.9cm。
縦誤差を1.68倍に水増ししていた。
報告していた4.21cmは、実際は3.19cmだった。

4/ 度で測り直したら MPIIFaceGaze LOPO 16-shot で 5.73°。
同じ土俵の2026年の論文が8.82°。
ただし条件が完全一致か確認できないので「SOTA超え」とは書きません。

5/ 効かなかった手法も全部公開しています。
リンバス楕円フィットはMediaPipeの学習済みランドマークに負けました。
https://github.com/taru104/tgaze
```

---

## E. awesome 系へのPR

- `codeberg.org/eyes-on-disabilities/awesome-eye-tracking` — アクセシビリティ文脈で最有力
- `awesome-mediapipe` 系 — MediaPipe応用として

**追加する1行の例**
```
- [Tgaze](https://github.com/taru104/tgaze) — Webcam gaze tracking that outputs
  on-screen coordinates (not just gaze angle). CPU-only, ~2.6 cm after a 9-point
  calibration. Noncommercial license.
```

---

## 投稿時の注意（守らないと逆効果）

- **数字を盛らない**。「SOTA超え」と書いた瞬間に条件差を突かれ、正直さという最大の資産を失う。
- **非商用ライセンスを最初に自分から言う**。後で発覚する方が心証が悪い。
- 動画/GIFを**最初の3秒**に置く。文章より先に見られる。
- コメントには全部返す。初速の議論がそのまま順位になる。
