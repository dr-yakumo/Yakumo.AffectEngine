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

## Requirements / 動作要件

### Runtime / ランタイム
- [.NET 8.0 Runtime](https://dotnet.microsoft.com/download/dotnet/8.0)
- Python 3.9 or later — required for Japanese translation / 日本語翻訳に必要
- [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) — required by PyTorch / ONNX Runtime. The installer can set it up automatically / PyTorch・ONNX Runtime の実行に必要（インストーラーが自動導入できます）

### Python packages / Python パッケージ
pip install flask transformers torch sentencepiece protobuf


### Models / モデル (auto-downloaded by install.ps1 / install.ps1 が自動ダウンロード)
- `SamLowe/roberta-base-go_emotions` — emotion classification model (ONNX) / 感情分類モデル
- `Helsinki-NLP/opus-mt-ja-en` — Japanese → English translation via Flask server (default) / 日英翻訳サーバー（既定）

The translation model is selectable at install time (`opus` / `nllb` / `mt5`).
`opus` is the default — it is lightweight and permissively licensed.
翻訳モデルはインストール時に選択できます（`opus` / `nllb` / `mt5`）。
既定は `opus` です（軽量かつライセンス面で扱いやすいため）。

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
    "やった！欲しかったグラボが激安で買えた！",
    SpeakerRole.User);

Console.WriteLine(result.TopEmotion);   // 喜び
Console.WriteLine(result.TopScore);     // 0.812
Console.WriteLine(result.IsSurprised);  // False

foreach (var kv in result.TopK)
    Console.WriteLine($"{kv.Key} = {kv.Value:F3}");
    // 喜び = 0.812
    // 期待 = 0.078
    // 信頼 = 0.065
```

Run `install.bat` first — it sets up the ONNX models, `affect.config`, and the Python
translation server that Japanese input requires.  
先に `install.bat` を実行してください。ONNX モデル・`affect.config`・日本語入力に必要な
Python 翻訳サーバーが用意されます。

> 📊 **The scores above are from the default setup and are not fixed values.**
> `TopK` returns the top *k* labels (3 by default; pass `k` to change it), and the scores
> shift as you tune the engine for your own domain — label weights in `affect.config`
> and entries in `affect.dict.json` are both meant to be adjusted.
> Treat the **ranking** as the primary output and the numbers as relative confidence.
>
> 📊 **上記のスコアは既定構成での値であり、固定値ではありません。**
> `TopK` は上位 *k* 件を返します（既定は3件。引数 `k` で変更可）。スコアは自分のドメインに
> 合わせた調整で変動します（`affect.config` のラベル重み、`affect.dict.json` のエントリは
> どちらも調整される前提のものです）。
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

`flask` / `transformers` / `torch` / `sentencepiece` / `protobuf` / `sacremoses` / `hf_xet` / `optimum` / `onnx` / `onnxconverter-common` — all obtained from [PyPI](https://pypi.org/).

> ⚠️ `facebook/nllb-200-distilled-600M` is published under **CC-BY-NC-4.0 (non-commercial use only)**. The default translation model is `opus`; if you switch to `nllb`, please check its terms before using it commercially.  
> ⚠️ `facebook/nllb-200-distilled-600M` は **CC-BY-NC-4.0（非商用限定）** で公開されています。デフォルトの翻訳モデルは `opus` です。`nllb` に切り替えて商用利用する場合は、事前に条件をご確認ください。

---

## Contributing / コントリビューション

This repository does not accept pull requests.  
Feel free to fork and modify for your own use.  
このリポジトリはプルリクエストを受け付けていません。  
フォークして自由にご利用ください。

---

## Disclaimer / 免責事項

This library is provided "as is", without warranty of any kind, express or implied.  
The author(s) shall not be held liable for any damages arising from the use of this software.  
Use at your own risk.

本ライブラリは現状有姿で提供されます。  
本ソフトウェアの使用によって生じたいかなる損害についても、開発者は一切の責任を負いません。  
ご利用は自己責任でお願いします。