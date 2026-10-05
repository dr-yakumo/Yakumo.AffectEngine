# Yakumo Affect Engine — 公開 API リファレンス

> 🌐 **Language**: 日本語 | [English](YAKUMO_NLI_API_en.md)

> **対象バージョン**: v1.2
> **最終更新**: 2026-10-05
> **名前空間**: `Yakumo.Affect`
> **対象読者**: ライブラリ利用者（外部開発者）

本ドキュメントは、ライブラリ利用者向けの **公開 API リファレンス** となります
内部実装・レガシー推論パス(Obsoleteによる非推薦クラスなど)の詳細は対象外となっております（[10. レガシー / 非推奨 API](#10-レガシー--非推奨-api) に一覧のみ記載）。

---

## 目次

1. [概要](#1-概要)
2. [クイックスタート](#2-クイックスタート)
3. [AffectCore — 感情分析エンジン](#3-affectcore--感情分析エンジン)
4. [SpeakerRole (enum)](#4-speakerrole-enum)
5. [結果クラス](#5-結果クラス)
6. [AffectAnalysisResult への変換](#6-affectanalysisresult-への変換)
7. [TranslationService — 翻訳辞書カスタマイズ](#7-translationservice--翻訳辞書カスタマイズ)
8. [設定 (affect.config)](#8-設定-affectconfig)
9. [拡張ポイント: IEmotionFilter](#9-拡張ポイント-iemotionfilter)
    - [9.5 拡張ポイント: Classified イベント](#95-拡張ポイント-classified-イベント)
10. [レガシー / 非推奨 API](#10-レガシー--非推奨-api)
11. [ラベルリファレンス](#11-ラベルリファレンス)
12. [よくある使用パターン](#12-よくある使用パターン)

---

## 1. 概要

Yakumo Affect Engine は、日本語・英語テキストを対象とした **14 ラベル多ラベル感情分析** ライブラリです (.NET 8.0 / C# 12.0)。

- 推論エンジン: GoEmotions RoBERTa (ONNX Runtime、DirectML GPU 対応)
- 日本語入力は内蔵の翻訳サービス（ローカル Python プロセス）で英語に変換してから推論
- 28 ラベルの GoEmotions 出力を Yakumo 独自の 14 ラベルに集約

### はじめに（入手とセットアップ）

本ライブラリは [**Releases ページ**](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) から
ZIP をダウンロードして使います。展開して `install.bat` を実行すると、ONNX モデルの取得・
Python 環境の構築・`affect.config` の配置までが自動で行われます。

- 事前に必要なもの: [.NET 8.0 Runtime](https://dotnet.microsoft.com/download/dotnet/8.0) / Python 3.9 以降
- 詳細は同梱の [README](https://github.com/dr-yakumo/Yakumo.AffectEngine#readme) および `install.ps1` を参照してください

---

## 2. クイックスタート

```csharp
using Yakumo.Affect;

// 日本語モードで初期化（設定は affect.config を自動読み込み）
using var engine = new AffectCore(language: "jp", debugMode: false);

// 感情分析（設定ファイルの Model に応じたエンジンを自動選択）
var result = await engine.AnalyzeTextWithAutoModelAsync(
    "欲しかったグラボが激安で買えて本当に嬉しい！",
    SpeakerRole.User);

Console.WriteLine(result.TopEmotion);   // "喜び"
Console.WriteLine(result.TopScore);     // 0.809
Console.WriteLine(result.IsSurprised);  // false

foreach (var kv in result.TopK)
    Console.WriteLine($"{kv.Key} = {kv.Value:F3}");
    // 喜び = 0.809
    // 期待 = 0.041
    // 欲望 = 0.034
```

> 📊 **スコアは既定構成での実測値であり、固定値ではありません。**
> `affect.config` のラベル重みや `affect.dict.json` のエントリを調整すると変動します。
> ラベル重みは乗数なので、**スコアは 1.0 を超えることがあります**。確率ではありません。
> **順位**を主たる出力、数値は相対的な確信度として扱ってください。

> 💡 実際に動く対話型サンプルが `Yakumo.Affect.Sample/Program.cs` にあります。

> ⚠️ **実行時アセットが必要です。** 上記のコードを動かすには、[Releases](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) の ZIP に含まれる `install.bat` / `install.ps1` が用意するアセットが要ります。
> ライブラリを参照するだけでは動きません。インストーラーを使わない場合は、以下を手動で配置してください。
>
> | アセット | 用途 |
> |---|---|
> | `affect.config` | エンジン設定。`affect.config.default` からコピーする |
> | `libs/models/goemo-roberta-base/` | ONNX 分類モデル（`model.onnx`・トークナイザ一式） |
> | Python 翻訳サーバー | **日本語入力に必須。** エンジンが自動起動する。Python 3.9+ と `requirements.txt` のパッケージが必要 |
> | `Yakumo.Affect.PolarityGate.dll` | 任意。実行ファイルと同じディレクトリに置くと極性フィルタが有効になる |
>
> パスは実行ファイルのディレクトリ（`AppContext.BaseDirectory`）を基準に解決されます。

> 📊 **スコアは固定値ではありません。** 自分のドメインに合わせた調整で変動します
> （`affect.config` のラベル重み、`affect.dict.json` のエントリはどちらも調整される前提のものです）。
> **順位**を主たる出力、数値は相対的な確信度として扱ってください。

---

## 3. AffectCore — 感情分析エンジン

**実装**: `IDisposable`
感情分析のエントリポイントです。使い終わったら `Dispose()`（または `using`）でモデルセッションを解放してください。

> ⚠️ **注意: `AnalyzeText` / `AnalyzeTextAsync` / `AnalyzeTexts` / `AnalyzeTextsAsync` は使用しないでください。**
> これらは旧 BERT センチメントモデル専用のレガシー API で、**GoEmotions モデルには接続されません**（呼ぶと精度の低い旧パスで推論されます）。
> 通常は必ず **`AnalyzeTextWithAutoModelAsync` / `AnalyzeTextsWithAutoModelAsync`** を使用してください。

### 3.1 コンストラクタ

```csharp
public AffectCore(
    string language = "jp",
    bool debugMode = false,
    double temperature = 0.3,
    string scoreMode = "ent_minus_neu",
    string aggregate = "max")
```

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `language` | `string` | `"jp"` | `"jp"` または `"en"`。それ以外は `ArgumentException` |
| `debugMode` | `bool` | `false` | `true` で詳細ログをコンソール出力 |
| `temperature` | `double` | `0.3` | Softmax 温度（NLI 系パスのみ有効） |
| `scoreMode` | `string` | `"ent_minus_neu"` | NLI スコア計算方式（NLI 系パスのみ有効） |
| `aggregate` | `string` | `"max"` | プロンプト集約方法（NLI 系パスのみ有効） |

> `temperature` / `scoreMode` / `aggregate` は `affect.config` の `[nli_model]` セクションに同名キーがある場合、**設定ファイルの値で上書き** されます。GoEmotions パスではこれら 3 つは使用されません。

> 📌 **`language` はインスタンス生成時に固定される動作モードです。** 入力テキストの言語を自動判定する機能ではなく、分析メソッド呼び出しごとに切り替えることもできません。
>
> | | `"jp"` モード | `"en"` モード |
> |---|---|---|
> | 翻訳 | 入力の内容にかかわらず毎回翻訳ステップを経由 | 翻訳なし（入力をそのまま推論） |
> | 結果のラベル名 | 日本語表示名（「喜び」等） | 英語正準名（`joy` 等） |
>
> 日本語と英語のテキストが混在するワークロードでは、言語ごとに別のインスタンス（`CreateJapanese()` / `CreateEnglish()`）を使い分けてください。

### 3.2 ファクトリメソッド

```csharp
public static AffectCore CreateDefault()   // 日本語 + デフォルト設定
public static AffectCore CreateJapanese()  // CreateDefault と同等
public static AffectCore CreateEnglish()   // 英語設定
```

### 3.3 プロパティ

| プロパティ | 型 | 説明 |
|---|---|---|
| `Language` | `string` | `"jp"` または `"en"` |
| `DebugMode` | `bool` | デバッグモード状態 |
| `ModelName` | `string` | `affect.config` の `[nli_model] Model` 値（例: `"goemo-roberta-base"`） |

### 3.4 分析メソッド（推奨エントリポイント）

```csharp
// 単一テキスト
public Task<ClassificationResult> AnalyzeTextWithAutoModelAsync(
    string text,
    SpeakerRole role = SpeakerRole.User,
    int k = 3,
    double threshold = 0.25,
    string[]? properNouns = null)

// 複数テキスト
public Task<NliAnalysisResults> AnalyzeTextsWithAutoModelAsync(
    string[] texts,
    SpeakerRole role = SpeakerRole.User,
    int k = 3,
    double threshold = 0.25)
```

| パラメータ | 説明 |
|---|---|
| `text` / `texts` | 分析対象テキスト。`"jp"` モードでは翻訳を経由し、`"en"` モードではそのまま推論されます（[3.1節](#31-コンストラクタ)参照） |
| `role` | 発話者ロール（[4章](#4-speakerrole-enum)）。ロール別の重みが適用されます |
| `k` | 返却する上位感情の数 (Top-K) |
| `threshold` | 驚き判定閾値。`affect.config` の `SurpriseThreshold` があればそちらが優先 |
| `properNouns` | 指定した語を翻訳前に `"entity"` に置換し、固有名詞の誤訳・スコアへの影響を抑制 |

`affect.config` の `[nli_model] Model` の値によって内部エンジンが自動選択されます:

| Model 設定値 | 推論エンジン | 状態 |
|---|---|---|
| `goemo-roberta-base` | GoEmotions 直接分類 | **推奨（現行）** |
| `roberta-large-mnli` | RoBERTa-MNLI NLI | レガシー（研究・フォールバック用） |
| その他 | BERT センチメント | レガシー |

### 3.5 モデル判定 / 低レベルメソッド

```csharp
// 現在の設定が GoEmotions モデルかを判定
public bool ShouldUseGoEmotionsModel()

// GoEmotions パスを直接呼び出す（通常は AnalyzeTextWithAutoModelAsync 経由を推奨）
public Task<ClassificationResult> ClassifyByGoEmotionsAsync(
    string text,
    SpeakerRole role = SpeakerRole.User,
    int k = 3,
    double threshold = 0.25,
    string[]? properNouns = null)
```

### 3.6 Dispose

```csharp
public void Dispose()
```

保持している全 ONNX セッションを解放します。`using` 宣言の利用を推奨します。

---

## 4. SpeakerRole (enum)

```csharp
public enum SpeakerRole
{
    User,    // ユーザー（人間）の発話 — デフォルト
    AI,      // AI（人工知能）の応答
    System   // システムメッセージ
}
```

ロールに応じて一部ラベルのスコアに重みが掛かります（AI/System 発話では「驚き」が抑制され「中立」が優遇されます）。

| ラベル | User | AI | System |
|---|---|---|---|
| 驚き / surprise | 1.0 | **0.3** | **0.1** |
| 中立 / neutral | 1.0 | **1.5** | **2.0** |
| 喜び / joy | 1.0 | **0.8** | 1.0 |

---

## 5. 結果クラス

### 5.1 ClassificationResult

`AnalyzeTextWithAutoModelAsync` の戻り値です。

| プロパティ | 型 | 説明 |
|---|---|---|
| `TopEmotion` | `string` | 最高スコアの感情ラベル（言語別表示名）。TopK が空なら中立 |
| `TopScore` | `double` | 最高スコア値 |
| `Scores` | `Dictionary<string, double>` | 全ラベルのスコア辞書（sigmoid ベース × 重み。内部の昇格補正により 1.0 を超えることがあります） |
| `TopK` | `List<KeyValuePair<string, double>>` | 上位 K 件（フィルタ適用後・スコア降順） |
| `IsSurprised` | `bool` | 驚き判定フラグ |
| `SurpriseScore` | `double` | 驚きラベルのスコア |
| `Threshold` | `double` | 驚き判定に使用した閾値 |
| `Text` | `string` | 推論に使用したテキスト（日本語入力時は翻訳後の英文） |
| `OriginalText` | `string` | 入力原文 |
| `Language` | `string` | エンジン生成時の `language` モードのコピー（[3.1節](#31-コンストラクタ)参照）。呼び出しごとに変化せず、入力テキストの言語判定結果でもありません |
| `Role` | `SpeakerRole` | 発話者ロール |

```csharp
var result = await engine.AnalyzeTextWithAutoModelAsync("最悪だ、電車が遅れた");

Console.WriteLine(result.TopEmotion);              // "嫌悪"
if (result.Text != result.OriginalText)
    Console.WriteLine($"翻訳: {result.Text}");     // "This is awful. ..."
```

### 5.2 NliAnalysisResults

`AnalyzeTextsWithAutoModelAsync`（複数テキスト）の戻り値です。

| メンバー | 型 | 説明 |
|---|---|---|
| `Results` | `List<ClassificationResult>` | 各テキストの分析結果 |
| `TotalCount` | `int` | 結果件数 |
| `SurpriseCount` | `int` | 驚き判定されたテキスト数 |
| `MostCommonEmotion` | `string` | 最頻出の Top 感情 |
| `ProcessingTime` | `TimeSpan` | 処理時間 |
| `ModelUsed` | `string` | 使用モデル名 |
| `AnalyzedAt` | `DateTime` | 分析日時 |
| `GetSummary()` | `string` | サマリー文字列を生成 |

---

## 6. AffectAnalysisResult への変換

`ClassificationResult` はエンジン内部の状態（`Role` 等）を含みます。分析結果を保存・シリアライズ・他レイヤーへ受け渡す場合は、疎結合な DTO である `AffectAnalysisResult` への変換を推奨します。

```csharp
using Yakumo.Affect;

ClassificationResult raw = await engine.AnalyzeTextWithAutoModelAsync(text);

// 単一結果の変換
AffectAnalysisResult dto = raw.ToAnalysisResult();

// 複数結果の一括変換
NliAnalysisResults batch = await engine.AnalyzeTextsWithAutoModelAsync(texts);
List<AffectAnalysisResult> dtos = batch.ToAnalysisResults();

// 特定ラベルのスコア取得（未存在ラベルは 0.0）
double joy = dto.GetEmotionScore("喜び");
```

`AffectAnalysisResult` のプロパティは `ClassificationResult` とほぼ同一です（`Role` を持たない点、`GetEmotionScore(string)` を持つ点が異なります）。

---

## 7. TranslationService — 翻訳辞書カスタマイズ

**シングルトン**: `TranslationService.Instance`

日本語→英語翻訳はエンジンが内部で自動実行するため、通常は直接呼ぶ必要はありません。
外部利用者向けの主な用途は **翻訳辞書のカスタマイズ** です。定型句・固有名詞・誤訳パターンを登録することで、翻訳品質（＝感情分析精度）を改善できます。

### 7.1 辞書登録 API（フルーエント）

```csharp
var ts = TranslationService.Instance;

ts.AddPhrase("眠いなぁ", "I'm sleepy.")            // フレーズ辞書（完全一致で翻訳APIをスキップ）
  .AddProperNoun("株式会社サンプル", "Sample Corporation")  // 固有名詞マップ（翻訳前に部分一致置換）
  .AddCorrection("bird skin", "goosebumps");       // 翻訳後補正（誤訳パターンの修正）

// JSON ファイルから一括読み込み
ts.LoadDictionaryFromJson("affect.dict.json");
```

**適用優先順位**: フレーズ辞書（完全一致）> 翻訳キャッシュ > HTTP 翻訳 API

### 7.2 辞書 JSON フォーマット

`affect.dict.json.example` をコピーして編集してください:

```json
{
  "version": 1,
  "phraseDict":  { "眠いなぁ": "I'm sleepy." },
  "properNouns": { "株式会社サンプル": "Sample Corporation" },
  "corrections": { "bird skin": "goosebumps" }
}
```

> ⚠️ **同梱の辞書は既定モデル `opus` 専用です。**
> エントリは `Helsinki-NLP/opus-mt-ja-en` が**実際に出す誤訳**に対して作られています。
>
> ```
> 鳥肌が立つ  →  opus は "bird skin" と直訳する  →  corrections で goosebumps に補正
> ```
>
> **`[nli_translation] Model` を変更すると、同梱辞書はほとんど機能しなくなります。**
>
> なお、このエントリ群は **`opus-mt-ja-en` の特定リビジョンが出す誤訳**に合わせて作られています。
> インストーラーはそのリビジョンを固定するため、上流が更新されても補正は空振りしません。
> **モデルを自前で差し替える場合は、補正辞書も作り直しが必要**とお考えください。
> 別のモデルは別の壊れ方をするため、`bird skin` のような文字列がそもそも現れません。
> モデルを変える場合は、**そのモデルの出力を見て辞書を作り直してください。**

**`properNouns` の置換先は英語でなくても構いません。**
翻訳が壊れる言い回しを、同義の平易な日本語へ書き換える用途にも使えます。

```json
"properNouns": {
  "株式会社サンプル": "Sample Corporation",
  "肩を落とす": "がっかりする"
}
```

日本語のまま渡すと翻訳側が再翻訳しないため壊れません
（英語を混ぜると再翻訳されて壊れることがあります）。
`text.Replace` による部分一致なので、**活用形ごとに登録が必要**です
（`肩を落とす` は `肩を落とした` に一致しません）。

### 7.3 翻訳 API（直接利用する場合）

```csharp
public Task<string> TranslateAsync(string japaneseText, string[]? properNouns = null)
public string Translate(string japaneseText, string[]? properNouns = null)  // 同期版（後方互換）
```

初回アクセス時にローカルの Python 翻訳サーバーを自動起動します（`BasePort` と `BasePort+1` の 2 ポートを使用。デフォルト 5000/5001）。初回はモデルロードのため数十秒〜数分かかることがあります。

---

## 8. 設定 (affect.config)

設定は INI 形式です。`affect.config.default` を `affect.config` にコピーして編集してください（`install.ps1` 実行時は自動コピー）。

**読み込み優先順位（低 → 高）**:

1. `AIClient.Config`（共通設定）
2. `affect.config`（旧名 `NLI.Config` も後方互換で有効）
3. 環境変数 `AFFECT_CONFIG` の指すファイル（旧名 `NLI_CONFIG` も有効）

### 8.1 主要キー早見表

```ini
[nli_model]
Model = goemo-roberta-base   ; 推奨。roberta-large-mnli / bert_sentiment はレガシー
ModelPrecision = fp32        ; fp32 | fp16（fp16 は GPU 使用時のみ推奨）
UseDirectML = false          ; true で DirectML GPU 推論（不可時は CPU に自動フォールバック）
DirectMLDeviceId = 0
SurpriseThreshold = 0.50     ; 驚き判定閾値
SurpriseDelta = 0.10

[nli_emotion]
LabelSet = auto              ; basic(7) | extended(14) | auto（GoEmotionsパスではextendedと同等）
EnableFiltering = true       ; 極性フィルタ（9章参照）
LabelWeight.sadness = 1.20   ; ラベル別スコア重み（canonical 英語名で指定）
GoEmoLabelMode = yakumo14    ; yakumo14 | raw28（8.4節参照・実験的機能）

[nli_translation]
Model = opus                 ; opus | nllb | mt5
Quality = high
TimeoutSeconds = 180
BasePort = 5000

[embedding]
Rescoring.Enabled = false    ; kNN 類似事例投票による再スコアリング（要インデックス生成）
```

各キーの詳細な説明は `affect.config.default` 内のコメントを参照してください。

> **PolarityGate プラグインについて**: 専用の設定セクションはありません。有効/無効は (a) `Yakumo.Affect.PolarityGate.dll` の配置有無、(b) `[nli_emotion] EnableFiltering` の 2 点で決まります（[9章](#9-拡張ポイント-iemotionfilter)参照）。返却する Top-K 件数は設定キーではなく、分析 API の引数 `k`（既定値 3）で指定します。

### 8.2 翻訳モデルとライセンス

| 設定値 | モデル | 特徴 | ライセンス |
|---|---|---|---|
| `opus`（デフォルト） | Helsinki-NLP/opus-mt-ja-en | 軽量・高速 | Apache-2.0 ✅ 商用可 |
| `nllb` | facebook/nllb-200-distilled-600M | 高精度・文脈理解 | CC-BY-NC-4.0 ⚠️ **非商用のみ** |
| `mt5` | google/mt5-small | ⚠️ **実験的・動作未保証** | Apache 2.0 |

> ⚠️ **`mt5` は現状まともな翻訳を出力しません。**
> `google/mt5-small` は穴埋めの事前学習しか行われておらず、翻訳用に調整されていないため、
> 入力に関わらず内部トークン（`<extra_id_0>`）だけを返します（2026-09-01 実測）。
> **将来の差し替え用に経路のみ残してあります。`opus` を使用してください。**

### 8.3 設定値へのプログラムアクセス

```csharp
using Yakumo.Affect.Config;

string model = AffectConfigManager.Get("nli_model", "Model", "goemo-roberta-base");
int port     = AffectConfigManager.GetInt("nli_translation", "BasePort", 5000);
double th    = AffectConfigManager.GetDouble("nli_model", "SurpriseThreshold", 0.50);
```

初回の `Get*` 呼び出し時に上記のレイヤード読み込みが自動実行されます。

### 8.4 GoEmoLabelMode — raw28 モード（実験的・上級者向け）

`[nli_emotion] GoEmoLabelMode = raw28` を設定すると、28→14 ラベル集約を **スキップ** し、GoEmotions 原本 28 ラベルの sigmoid 生スコアをそのまま返します。`ClassificationResult.Scores` / `TopK` のキーは以下の 28 種の英語ラベル名になります。

```
admiration, amusement, anger, annoyance, approval, caring, confusion,
curiosity, desire, disappointment, disapproval, disgust, embarrassment,
excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
pride, realization, relief, remorse, sadness, surprise, neutral
```

14 ラベルでは粒度が粗すぎる分析や、GoEmotions モデルの素の出力を評価したい研究用途向けです。通常利用ではデフォルトの `yakumo14` を推奨します。

**制約（raw28 では以下がすべて適用されません）**:

| 項目 | raw28 での挙動 |
|---|---|
| 日本語ラベル名 | ❌ 非対応。`language="jp"` でも **ラベル名は英語のまま**（28 ラベル分の日本語訳辞書は存在しません） |
| ラベル重み (`LabelWeight.*`) | ❌ 未適用（純粋な sigmoid 生スコア） |
| ロール別重み (`SpeakerRole`) | ❌ 未適用 |
| ラベル昇格補正（内部後処理） | ❌ 未適用 |
| Embedding 再スコアリング | ❌ 無効（`Rescoring.Enabled=true` でも動作しません） |
| 入力テキストの日本語→英語翻訳 | ✅ **通常通り実行**（ラベルモードと翻訳は独立した処理です） |
| 驚き判定 (`IsSurprised`) | ✅ 動作（常に英語キー `"surprise"` で判定） |

> ⚠️ **Top-K フィルタは raw28 でも通常通り呼ばれます。** ただし極性判定は 14 ラベル前提のため、28 ラベルに対しては言語設定によって部分的にしか機能せず、結果が歪む場合があります（特に `language="en"` では一部ラベルにのみ極性フィルタが部分適用されます）。
> **raw28 の生スコアを歪みなく取得したい場合は `[nli_emotion] EnableFiltering = false` を推奨します**（または `Yakumo.Affect.PolarityGate.dll` を配置しない構成）。

`language="jp"` + raw28 で起動した場合、翻訳とラベルモードが独立していることを知らせる注意文が起動時に一度だけコンソール出力されます。

### 8.5 ラベル重みによるチューニング (LabelWeight)

各感情ラベルの最終スコアには `[nli_emotion]` の `LabelWeight.<canonical名>` で指定する乗数が適用されます。**コード変更なしで「特定の感情が出やすい / 出にくい」を調整できる**、外部利用者向けの主要チューニング手段です。

```ini
[nli_emotion]
; 値は乗数: 1.0 = 変更なし / 0.5 = 半減 / 1.5 = 1.5倍
LabelWeight.neutral  = 0.65   ; 中立への逃げを抑制（感情ラベルを立ちやすくする）
LabelWeight.sadness  = 1.20   ; 悲しみの検出を強化
LabelWeight.surprise = 0.85   ; 驚きの過検出を抑制
```

- キー名は **canonical 英語名**（[11章](#11-ラベルリファレンス)の14ラベル）で指定します。`language="jp"` モードでも英語名で書きます（内部で日本語表示名に対応付けられます）
- 重みは 28→14 集約後・Top-K フィルタ前に適用されるため、`TopEmotion` / `TopK` の順位に直接影響します
- 0 以下の値・数値でない値は無視され、ビルトインのデフォルト値が使われます
- テンプレート（`affect.config.default`）には調整済みの推奨値が設定されています。まずそのまま使い、アプリケーションの傾向に合わせて微調整してください
- **反映タイミング**: 設定ファイルはプロセス起動時に一度だけ読み込まれます。変更後はアプリケーションを再起動してください
- ロール別重み（[4章](#4-speakerrole-enum)）はビルトイン固定で、対応する設定キーはありません
- raw28 モード（[8.4節](#84-goemolabelmode--raw28-モード実験的上級者向け)）ではラベル重みは適用されません

> 💡 集約時に埋もれやすい感情シグナルを引き上げる **昇格閾値**（`DisgustPromotion.Threshold` / `AngerPromotion.Threshold` / `SurprisePromotion.Threshold`）も同セクションで調整できます。低くするほど昇格しやすくなります（誤検出増加のリスクあり）。詳細は `affect.config.default` のコメントを参照してください。

---

## 9. 拡張ポイント: IEmotionFilter

Top-K 確定前のスコアに対するポストフィルタは、プラグイン方式で差し替え可能です。

```csharp
public interface IEmotionFilter
{
    List<KeyValuePair<string, double>> Filter(
        IDictionary<string, double> scores,
        int k,
        string language,
        Action<string>? debugLog = null);
}
```

エンジンは起動時に以下の順でフィルタを選択します:

1. **`Yakumo.Affect.PolarityGate.dll`**（オプションのバイナリ配布プラグイン、Apache 2.0）が実行ファイルと同じディレクトリに存在すれば、それをロード
   — ポジティブ / ネガティブ感情が Top-K 内で衝突した場合の解決処理を提供します
2. 存在しない・ロードに失敗した場合は **`DefaultTopKFilter`**（単純スコア降順 Top-K）にフォールバック

フィルタの制御はすべて `[nli_emotion]` セクションで行います（PolarityGate 専用の設定セクションはありません）:

- `EnableFiltering = false` — フィルタリング自体を無効化（DLL があっても単純 Top-K になります）
- `[nli_emotion]` にある PolarityGate 調整キー（`NeutralizeConflicts` 等）は上級者向けです。**プラグインの内部仕様は非公開のため**、特別な理由がない限りデフォルト値のまま使用してください
- 返却件数は分析 API の引数 `k` で指定します（設定キーではありません）

独自のフィルタリング戦略が必要な場合は `IEmotionFilter` を実装してください（`DefaultTopKFilter` が参照実装です）。

---

## 9.5 拡張ポイント: Classified イベント

分析が1件完了するたびに発火します。**ハンドラを登録しなければ何も起きません。**

```csharp
public event Action<ClassificationResult>? Classified;
```

主な用途は、**翻訳で感情が壊れた入力を見つけること**です。
日本語入力の場合、`ClassificationResult` には原文と英訳の両方が入っています。

| プロパティ | 日本語入力時の中身 |
|---|---|
| `OriginalText` | 入力された日本語の原文 |
| `Text` | **翻訳後の英文**（分類器が実際に見たもの) |
| `Scores` / `TopK` | 14ラベルのスコア |

日本語は翻訳を経由して分類されるため、**慣用句が直訳されると感情が丸ごと失われます**。
実際に観測された例:

```
腹の虫が治まらない → "the insect in my stomach"
鳥肌が立つ         → "a bird's skin"
肩を落とす         → "I lost my shoulders"
```

しかし通常の運用では英訳を見ないため、**「なぜか neutral になった」としか分かりません**。
このイベントで英訳を拾っておけば、あとから辞書（`affect.dict.json`）に反映できます。

```csharp
using var core = AffectCore.Create();

core.Classified += result =>
{
    if (result.Language != "jp")
        return;

    // 英訳に日本語が残っている＝翻訳の失敗
    bool translationFailed = result.Text.Any(c => c >= 0x3040 && c <= 0x9FFF);

    if (translationFailed)
    {
        File.AppendAllText("candidates.jsonl",
            JsonSerializer.Serialize(new
            {
                ja = result.OriginalText,
                en = result.Text,
                top1 = result.TopEmotion,
            }) + Environment.NewLine);
    }
};
```

> **⚠️ プライバシー**
> これは利用者の入力文を扱う機能です。**ライブラリ自身は何も記録しません。**
> 記録するかどうか、どこに書くか、いつ消すかは、**すべて利用側の責任**です。
> 入力文が個人情報を含みうることに留意してください。

**動作上の注意**:

- 発火は**同期的**です。重い処理を書くと分析全体が遅くなります
- ハンドラ内で例外が出ても**分析は継続します**（例外は握り潰されます）
- バッチ分析（`AnalyzeTextsWithAutoModelAsync`）では**1件ごとに発火**します
- ハンドラが登録されていなければ、分析への負荷はほぼありません

---

## 10. レガシー / 非推奨 API

以下は後方互換のために公開されていますが、**新規コードでは使用しないでください**。

| API | 状態 | 代替 |
|---|---|---|
| `NLI_Core` クラス | `[Obsolete]`（旧クラス名エイリアス。将来削除予定） | `AffectCore` |
| `AnalyzeText` / `AnalyzeTextAsync` | レガシー（旧 BERT パス専用。GoEmotions 非接続） | `AnalyzeTextWithAutoModelAsync` |
| `AnalyzeTexts` / `AnalyzeTextsAsync` | レガシー（同上） | `AnalyzeTextsWithAutoModelAsync` |
| `ClassifyByNlpAsync` / `ShouldUseRobertaModel` | レガシー（RoBERTa-MNLI パス。研究・フォールバック用） | GoEmotions パス |
| `SetEvalTotalSamples` | 評価ツール用（外部利用非推奨） | — |
| `PythonTranslator` | `Deprecated`（エディタ非表示） | `TranslationService` |

---

## 11. ラベルリファレンス

Yakumo Affect Engine が出力する 14 ラベルの対応表です。
`ClassificationResult.Scores` / `TopK` のキーは言語設定（`language`）に応じた表示名になります。

| canonical (EN) | 日本語表示名 | 極性 |
|---|---|---|
| joy | 喜び | Positive |
| affection | 親しみ・愛情 | Positive |
| trust | 信頼 | Positive |
| anticipation | 期待 | Positive |
| desire | 欲望 | Positive |
| sadness | 悲しみ | Negative |
| anger | 怒り | Negative |
| fear | 恐れ | Negative |
| disgust | 嫌悪 | Negative |
| resentment | 恨み | Negative |
| disbelief | 落胆 | Negative |
| shame | 恥・罪悪感 | Negative |
| surprise | 驚き | Special |
| neutral | 中立 | Special |

**LabelSet モード別の有効ラベル** (`[nli_emotion] LabelSet`):

| モード | 有効ラベル |
|---|---|
| `basic` | 基本 7（surprise, anger, joy, sadness, fear, disgust, neutral）+ 設定によりポジ拡張 |
| `extended` | 全 14 ラベル |
| `auto` | GoEmotions パスでは `extended` と同等（全 14 ラベル。desire の有無は `EnableDesire.Auto` で制御） |

---

## 12. よくある使用パターン

### 会話システムでの感情トラッキング

```csharp
using var engine = new AffectCore("jp");

// ユーザー発話とAI応答でロールを使い分ける
var userEmotion = await engine.AnalyzeTextWithAutoModelAsync(userInput, SpeakerRole.User);
var aiEmotion   = await engine.AnalyzeTextWithAutoModelAsync(aiResponse, SpeakerRole.AI);
```

### RAG 連携での感情スコア取得

```csharp
var result = await engine.AnalyzeTextWithAutoModelAsync(userInput, SpeakerRole.User);

// 全スコアを感情デルタとして利用
foreach (var kv in result.Scores)
    Console.WriteLine($"{kv.Key}: {kv.Value:F4}");

// Top 感情をプロンプトに反映
string dominantEmotion = result.TopEmotion;
```

### 固有名詞を含むテキストの分析

```csharp
// 作品名・人名などが感情語として誤訳されるのを防ぐ
var result = await engine.AnalyzeTextWithAutoModelAsync(
    "モモちゃんの新しいグッズが猫カフェで販売開始！",
    SpeakerRole.User,
    properNouns: new[] { "モモちゃん", "猫カフェ" });
```

### バッチ分析と集計

```csharp
string[] logs = LoadChatLogs();
var batch = await engine.AnalyzeTextsWithAutoModelAsync(logs, SpeakerRole.User);

Console.WriteLine(batch.GetSummary());
Console.WriteLine($"最頻出感情: {batch.MostCommonEmotion}");
Console.WriteLine($"驚き件数  : {batch.SurpriseCount}/{batch.TotalCount}");
```

---