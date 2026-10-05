<p align="center">
  <img src="assets/logo1_2.1.1_50.png" alt="Yakumo.AffectEngine logo" />
</p>

# Yakumo.AffectEngine
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
Emotion analysis library for .NET 8 — understands **Japanese and English** feelings in text.  
Built on GoEmotions RoBERTa (ONNX) with DirectML support for fast GPU/CPU inference.

🧠 Yakumo.AffectEngine is a **static AI**: pretrained deep-learning models run frozen-weight
inference fully on your machine — no runtime learning, no cloud, reproducible results.
Tunable via configuration and user dictionaries (proper nouns, phrases, and translation fixes).

.NET 8 向け感情分析ライブラリ。**日本語・英語**のテキストから感情を読み取ります。  
GoEmotions RoBERTa (ONNX) を採用し、DirectML で GPU/CPU 高速推論に対応。

🧠 本ライブラリは**静的AI**です — 学習済み深層学習モデルによる固定重み推論を
完全ローカルで実行します。実行時学習なし・クラウド不要・再現性のある結果。
設定ファイルによるチューニングと、ユーザー辞書（固有名詞・フレーズ・誤訳補正）に対応。


---

📝 Note: A ready-to-run package (sample app + installer) is available on the [Releases](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) page.
📝 注記: すぐに使えるパッケージ（サンプルアプリ + インストーラー同梱の ZIP）は [Releases](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) ページから入手できます。

## Documentation / ドキュメント

- 📖 [Public API Reference (English)](Yakumo.Affect/YAKUMO_NLI_API_en.md)
- 📖 [公開 API リファレンス（日本語）](Yakumo.Affect/YAKUMO_NLI_API_jp.md)

## Features / 機能

- 🎭 **14-label multi-label emotion classification** (joy, sadness, anger, fear, disgust, surprise, neutral, and more)  
  **14ラベル多ラベル感情分類**（喜び・悲しみ・怒り・恐れ・嫌悪・驚き・中立 など）
- 🌐 **Bilingual support** — Japanese input auto-translated to English for inference  
  **日英バイリンガル対応** — 日本語入力は自動翻訳して推論
- ⚡ **DirectML GPU acceleration — no CUDA, no NVIDIA lock-in.** Runs on any DirectX 12 capable GPU (AMD Radeon / Intel Arc / NVIDIA), with automatic CPU fallback  
  **DirectML GPU アクセラレーション — CUDA 不要・NVIDIA 縛りなし。** DirectX 12 対応 GPU なら AMD Radeon / Intel Arc / NVIDIA いずれでも動作。利用できない場合は CPU へ自動フォールバック
- 🔌 **Plugin-based filter architecture** via `IEmotionFilter` interface  
  `IEmotionFilter` インターフェースによる**プラグイン式フィルタ構成**
- 🛡️ **Polarity Gate** (optional) — resolves positive/negative emotion conflicts for higher coherence  
  **Polarity Gate**（オプション） — ポジティブ/ネガティブ感情の矛盾を解消し、分類精度を向上
- 📦 **No cloud dependency** — fully local inference via ONNX Runtime  
  **クラウド不要** — ONNX Runtime による完全ローカル推論

---

## Accuracy / 精度

**Macro-F1 = 0.6976** ／ **Top-3 = 0.8591** — 14 labels, 504 Japanese sentences / 14ラベル・日本語504文

<details open>
<summary><b>English</b></summary>

### Score

**Macro-F1 = 0.6976** ／ **Top-3 = 0.8591**

| Metric | Meaning |
|---|---|
| Macro-F1 | F1 of the top-ranked emotion, computed per label and averaged over the 14 labels |
| Top-3 | Share of sentences whose correct label appears among the top 3 emotions in the result |

The 14 labels are the Yakumo 14: GoEmotions' 28 labels (raw28) consolidated into 14.
See [Label Reference](Yakumo.Affect/YAKUMO_NLI_API_en.md#11-label-reference) for the mapping.

### How it was measured

| | |
|---|---|
| Classifier | `SamLowe/roberta-base-go_emotions` (GoEmotions, 28 labels) consolidated into the Yakumo 14 |
| Evaluation set | 504 Japanese sentences with **explicit** emotion expressions, 36 per label (balanced) |
| How the set was handled | Never used for tuning and not published. Created with a different model from the development data, and measured **once**, after all tuning was finished |
| Device | CPU, fp32 — GPU (DirectML) and fp16 produce the same scores |
| Configuration | Measured with the bundled defaults (Polarity Gate enabled). The Polarity Gate is bundled and enabled by default. |
| Verified environment | `transformers 4.53.3` / `torch 2.8.0+cpu` / `sentencepiece 0.2.1` / `sacremoses 0.1.1` / Python 3.11.5 |

Please read these before quoting the number:

- **This is a 14-label score.** Do not compare it with results from classifiers that use a
  different number of labels (such as 7-class) — it is a different task.
- **The sentences express emotion explicitly.** Scores are much lower for implicit expressions
  (see *Known limitation* below).
- **The set is balanced (36 per label)**, not a real-world frequency distribution.
- **Measured with the Polarity Gate plugin enabled** (it is bundled and enabled by default).
  If you remove its DLL, Top-3 and other figures may change.
- **User dictionaries had almost no effect here** — they matched under 1% of the sentences.
  This reflects general Japanese performance, not domain tuning.
- **The Python libraries used by the translation server (`transformers` and others) are not
  pinned to a version.** The translation model weights are pinned to a revision, but a different
  library version can still change translation output, and therefore the score.
  The verified versions are listed in the table above.

### Known limitation

Sentences with **no emotion words, where the behaviour itself carries the feeling**
(for example *"I covered my mouth the moment I opened the bag"*) are **not handled well**.

This is structural. The GoEmotions dataset this engine builds on
([Demszky et al., 2020](https://ar5iv.labs.arxiv.org/html/2005.00547)) instructed its raters that
*"if raters were not certain about any emotion being expressed, they were asked to select Neutral"*
([Section 3.3 of the paper](https://ar5iv.labs.arxiv.org/html/2005.00547#S3.SS3)), and when
preparing the training data the authors *"filter out emotion labels selected by only a single
annotator"* ([Section 5.1 of the paper](https://ar5iv.labs.arxiv.org/html/2005.00547#S5.SS1)).
Text whose emotion is subtle is therefore pulled toward Neutral by the dataset's own design —
and a model trained on it inherits that.

**How it behaves when it fails:** it tends to return **neutral** (withholding judgement) rather
than confidently reporting the wrong emotion. On a separate set of 504 sentences containing only
implicit expressions (also never used for tuning), the engine returned neutral for about 68% of
the sentences and reported a wrong emotion for about 13%.

These are known issues, work is underway, and they will be addressed in a future update.

### Effect of user dictionaries

On a 70-sentence idiom test set, enabling the bundled dictionary raised top-1 accuracy
from **37.0% to 57.0%** (Macro-F1 0.3511 → 0.5353).

</details>

<details open>
<summary><b>日本語</b></summary>

### スコア

**Macro-F1 = 0.6976** ／ **Top-3 = 0.8591**

| 指標 | 意味 |
|---|---|
| Macro-F1 | 1位と判定した感情の F1 をラベルごとに求め、14ラベルで平均した値 |
| Top-3 | 分析結果の上位3件の中に正解ラベルが含まれていた文の割合 |

14ラベルとは、GoEmotions の28ラベル（raw28）を14ラベルに集約した Yakumo 14ラベルです。
対応関係は [ラベルリファレンス](Yakumo.Affect/YAKUMO_NLI_API_jp.md#11-ラベルリファレンス) を参照してください。

### 測定条件

| | |
|---|---|
| 分類器 | `SamLowe/roberta-base-go_emotions`（GoEmotions 28ラベル）を Yakumo 14ラベルへ集約 |
| 評価セット | 感情が**明示的に**表れた日本語504文・14ラベル各36件（均等） |
| 評価セットの扱い | 調整には一切使っておらず、公開もしていません。開発用のデータとは別のモデルで作成し、すべての調整を終えた後に**1回だけ**測定しました |
| 実行環境 | CPU・fp32 — GPU（DirectML）・fp16 でも同じ数値 |
| 構成 | 同梱の既定構成（Polarity Gate 有効）で測定しています。Polarity Gate は同梱されており、既定で有効です。 |
| 検証済み構成 | `transformers 4.53.3` ／ `torch 2.8.0+cpu` ／ `sentencepiece 0.2.1` ／ `sacremoses 0.1.1` ／ Python 3.11.5 |

この数値を引用する前にお読みください。

- **14ラベルでの数値です。** ラベル数の異なる分類（7分類など）の数値とは課題が異なるため、
  並べて比較しないでください。
- **感情が明示的に表れた文での数値です。** 暗示的な表現では大きく下がります（下記「既知の限界」）。
- **均等構成**（各36件）であり、実使用での出現頻度分布ではありません。
- **Polarity Gate プラグインを有効にした構成**での数値です（同梱・既定で有効）。
  DLL を取り除いた構成では、Top-3 などの数値が変わりえます。
- **ユーザー辞書はここではほとんど作用していません** — 一致したのは全体の1%未満です。
  したがってこれは特定ドメインへの適応ではなく、汎用日本語での性能です。
- **翻訳サーバー（Python）が使うライブラリ（`transformers` など）のバージョンは固定していません。**
  翻訳モデルの重みはリビジョンを固定していますが、ライブラリの版が違うと翻訳結果が変わり、
  数値も変わりえます。検証済みの版は上の表のとおりです。

### 既知の限界

**感情語を使わず、行動が感情を表す文**
（例:「袋を開けた瞬間、思わず口を押さえた」）は**うまく扱えません**。

これは構造的な理由によるものです。本エンジンが基盤とする GoEmotions データセット
（[Demszky ら, 2020](https://ar5iv.labs.arxiv.org/html/2005.00547)）では、注釈者に対して
「**いずれかの感情が表れていると確信できない場合は Neutral を選ぶこと**」と指示されており
（[同論文 3.3節](https://ar5iv.labs.arxiv.org/html/2005.00547#S3.SS3)）、
また学習データの作成時に**1名の注釈者しか選ばなかった感情ラベルは除外**されています
（[同論文 5.1節](https://ar5iv.labs.arxiv.org/html/2005.00547#S5.SS1)）。
**感情の表れ方が微妙なテキストは、データセットの設計そのものによって Neutral に寄ります。**
それで学習したモデルは、その性質を受け継ぎます。

**失敗したときの挙動**: 誤った感情を断定するよりも、**中立を返す**（判定を保留する）傾向があります。
暗示的な表現だけを集めた別の504文（こちらも調整には未使用）では、
約68%の文で中立を返し、誤った感情を断定したのは約13%でした。

これら既知の問題については対策中であり、今後のアップデートで対応していきます。

### ユーザー辞書の効果

慣用句70文のテストセットでは、同梱辞書を有効にすると Top-1 正答率が
**37.0% → 57.0%** に向上しました（Macro-F1 0.3511 → 0.5353）。

</details>

---

## Requirements / 動作要件

### Runtime / ランタイム
- [.NET 8.0 Runtime](https://dotnet.microsoft.com/download/dotnet/8.0)
- Python 3.9 or later — required for Japanese translation. This is a **minimum**, not the version the published scores were measured with (that was 3.11.5; see [Accuracy](#accuracy--精度)) / 日本語翻訳に必要。これは**下限**であって、公表スコアを測定した版ではありません（測定は 3.11.5。[精度](#accuracy--精度)を参照）
- [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) — required by PyTorch / ONNX Runtime. The installer can set it up automatically / PyTorch・ONNX Runtime の実行に必要（インストーラーが自動導入できます）

### Python packages / Python パッケージ
```
pip install -r requirements.txt
```

Installing by hand instead? `sacremoses` is **required**, not optional — the Marian
tokenizer uses it for normalization, and translation output differs without it.

手動で入れる場合、`sacremoses` は**必須**です。Marian トークナイザーの正規化に
使われるため、無いと翻訳結果が変わります。

```
pip install flask transformers torch sentencepiece protobuf sacremoses
```


### Models / モデル (auto-downloaded by install.ps1 / install.ps1 が自動ダウンロード)
- `SamLowe/roberta-base-go_emotions` — emotion classification model (ONNX) / 感情分類モデル
- `Helsinki-NLP/opus-mt-ja-en` — Japanese → English translation via Flask server (default) / 日英翻訳サーバー（既定）

Both are **pinned to a specific revision**, so every installation gets exactly the weights the
published scores were measured with — even if the upstream repository is updated later.

いずれも**リビジョンを固定**しています。上流が更新されても、
**公表スコアを測定したものと同一の重み**が必ず導入されます。

The translation model is selectable at install time (`opus` / `nllb` / `mt5`).
`opus` is the default and the only supported choice — it is lightweight and permissively
licensed. `nllb` is **non-commercial** (CC-BY-NC-4.0), and `mt5` is **experimental and
currently unusable** — that path is kept only for a future replacement model.

翻訳モデルはインストール時に選択できます（`opus` / `nllb` / `mt5`）。
既定かつサポート対象は `opus` です（軽量かつライセンス面で扱いやすいため）。
`nllb` は **非商用限定**（CC-BY-NC-4.0）、`mt5` は **実験的で現状まともに動きません**
（将来の差し替え用に経路のみ残しています）。

### Optional / オプション
- DirectML-compatible GPU — for faster inference. Any DirectX 12 capable device works; **CUDA is not required**  
  DirectML 対応 GPU — 高速推論に使用。DirectX 12 対応であればよく、**CUDA は不要**です
  - Verified on AMD Radeon RX 6000 series / 動作確認: AMD Radeon RX 6000 系
  - Other DirectX 12 GPUs (Intel Arc, NVIDIA, older Radeon) are expected to work but are untested  
    その他の DirectX 12 対応 GPU（Intel Arc・NVIDIA・旧世代 Radeon 等）も動作する見込みですが未検証です

### 🛠️ Setup / セットアップ
- Download the ZIP from the [Releases](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) page and run `install.bat` — Python packages and ONNX models are set up automatically. (.NET 8.0 Runtime and Python 3.9+ must be installed beforehand.)
- [Releases](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) ページから ZIP を取得して `install.bat` を実行してください。Python パッケージの導入と ONNX モデルの取得は自動で行われます（.NET 8.0 Runtime と Python 3.9+ は事前にインストールしておいてください）。
---

## Quick Start / クイックスタート

```csharp
using Yakumo.Affect;

// affect.config is loaded automatically / affect.config は自動で読み込まれます
using var engine = new AffectCore(language: "jp", debugMode: false);

var result = await engine.AnalyzeTextWithAutoModelAsync(
    "欲しかったグラボが激安で買えて本当に嬉しい！",
    SpeakerRole.User);

Console.WriteLine(result.TopEmotion);   // 喜び
Console.WriteLine(result.TopScore);     // 0.809
Console.WriteLine(result.IsSurprised);  // False

foreach (var kv in result.TopK)
    Console.WriteLine($"{kv.Key} = {kv.Value:F3}");
    // 喜び = 0.809
    // 期待 = 0.041
    // 欲望 = 0.034
```

Run `install.bat` first — it sets up the ONNX models, `affect.config`, and the Python
translation server that Japanese input requires.  
先に `install.bat` を実行してください。ONNX モデル・`affect.config`・日本語入力に必要な
Python 翻訳サーバーが用意されます。

> 📊 **The scores above are measured with the default setup and are not fixed values.**
> `TopK` returns the top *k* labels (3 by default; pass `k` to change it), and the scores
> shift as you tune the engine for your own domain — label weights in `affect.config`
> and entries in `affect.dict.json` are both meant to be adjusted.
> Because label weights are multipliers, **a score can exceed 1.0** — it is not a probability.
> Treat the **ranking** as the primary output and the numbers as relative confidence.
>
> 📊 **上記のスコアは既定構成での実測値であり、固定値ではありません。**
> `TopK` は上位 *k* 件を返します（既定は3件。引数 `k` で変更可）。スコアは自分のドメインに
> 合わせた調整で変動します（`affect.config` のラベル重み、`affect.dict.json` のエントリは
> どちらも調整される前提のものです）。
> ラベル重みは乗数なので、**スコアは 1.0 を超えることがあります**。確率ではありません。
> **順位**を主たる出力、数値は相対的な確信度として扱ってください。

An interactive sample is included in the release ZIP under `samples/`.  
リリース ZIP の `samples/` に対話型のサンプルが同梱されています。

→ [API Reference (English)](Yakumo.Affect/YAKUMO_NLI_API_en.md) ／ [API リファレンス（日本語）](Yakumo.Affect/YAKUMO_NLI_API_jp.md)

---

## License / ライセンス

The core library (`Yakumo.Affect`) is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.  
コアライブラリ（`Yakumo.Affect`）は **MIT ライセンス** で提供されます。

The optional Polarity Gate plugin (`Yakumo.Affect.PolarityGate.dll`) is distributed as **binary only** under the **Apache License 2.0**.  
See [NOTICE_PolarityGate](Yakumo.Affect.PolarityGate/NOTICE_PolarityGate) for details.  
オプションの Polarity Gate プラグイン（`Yakumo.Affect.PolarityGate.dll`）は **バイナリのみ配布** で、**Apache License 2.0** が適用されます。

### Third-party notices / サードパーティ表示

#### 1. Bundled components / 同梱しているコンポーネント

These binaries ship inside the release package. Their full license texts are reproduced in [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).  
以下はリリースパッケージに同梱しているバイナリです。ライセンス全文は [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) に収録しています。

| Component | Files | License |
|---|---|---|
| ONNX Runtime (`Microsoft.ML.OnnxRuntime.DirectML`) | `Microsoft.ML.OnnxRuntime.dll`, `runtimes/*/native/onnxruntime*.dll` | MIT |
| Microsoft.ML.Tokenizers | `Microsoft.ML.Tokenizers.dll` | MIT |
| .NET runtime libraries | `System.Numerics.Tensors.dll` | MIT |
| Google Protocol Buffers | `Google.Protobuf.dll` | BSD-3-Clause |
| Yakumo.Affect.PolarityGate | `Yakumo.Affect.PolarityGate.dll` | Apache 2.0 |

#### 2. Downloaded at install time / インストール時に取得するもの

The following are **not bundled**. `install.ps1` downloads them from their original distributors when you run it, so each is governed by the license published by its distributor. Please check the distributor for the authoritative and current terms.  
以下は **同梱していません**。`install.ps1` の実行時に各配布元から取得するため、適用される条件は各配布元が公開するライセンスに従います。正確かつ最新の条件は配布元をご確認ください。

**ONNX models / ONNX モデル**

| Model | Distributor |
|---|---|
| `SamLowe/roberta-base-go_emotions` | [HuggingFace](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
| `sentence-transformers/all-MiniLM-L6-v2` | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `Helsinki-NLP/opus-mt-ja-en` | [HuggingFace](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en) |
| `facebook/nllb-200-distilled-600M` ⚠️ | [HuggingFace](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| `google/mt5-small` | [HuggingFace](https://huggingface.co/google/mt5-small) |

**Python packages / Python パッケージ** (see `requirements.txt`)

`flask` / `transformers` / `torch` / `sentencepiece` / `protobuf` / `sacremoses` / `hf_xet` / `optimum` / `onnx` — all obtained from [PyPI](https://pypi.org/).

> ⚠️ `facebook/nllb-200-distilled-600M` is published under **CC-BY-NC-4.0 (non-commercial use only)**. The default translation model is `opus`; if you switch to `nllb`, please check its terms before using it commercially.  
> ⚠️ `facebook/nllb-200-distilled-600M` は **CC-BY-NC-4.0（非商用限定）** で公開されています。デフォルトの翻訳モデルは `opus` です。`nllb` に切り替えて商用利用する場合は、事前に条件をご確認ください。

---

## Project status / プロジェクトの体制

Yakumo.AffectEngine is built and maintained by **a single developer**.
Issues and questions are welcome, but replies may take time, and feature requests
may not be picked up. Pull requests are not accepted — please fork and modify for
your own use.

**Nothing here depends on the author staying available.** Inference runs entirely on
your machine: no account, no API key, no server to keep alive. The code is MIT
licensed, so if this project stops, your copy keeps working and you are free to fork it.

Yakumo.AffectEngine は**個人開発**です。
Issue や質問は歓迎しますが、返信に時間がかかることがあり、
ご要望にお応えできない場合もあります。プルリクエストは受け付けていません。
フォークして自由に改変してください。

**ただし、このライブラリの動作は作者の存在に依存しません。**
推論は完全にお手元で動き、アカウントも API キーも、維持すべきサーバーもありません。
MIT ライセンスなので、仮に開発が止まっても手元のものは動き続け、フォークも自由です。

---

## Disclaimer / 免責事項

This library is provided "as is", without warranty of any kind, express or implied.  
The author(s) shall not be held liable for any damages arising from the use of this software.  
Use at your own risk.

本ライブラリは現状有姿で提供されます。  
本ソフトウェアの使用によって生じたいかなる損害についても、開発者は一切の責任を負いません。  
ご利用は自己責任でお願いします。