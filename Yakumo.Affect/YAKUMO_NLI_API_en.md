# Yakumo Affect Engine — Public API Reference

> 🌐 **Language**: [日本語](YAKUMO_NLI_API_jp.md) | English

> **Target version**: v1.2
> **Last updated**: 2026-10-05
> **Namespace**: `Yakumo.Affect`
> **Audience**: Library consumers (external developers)

This document is the **public API reference** for library consumers.
Internal implementation and legacy inference paths (e.g. classes deprecated via `[Obsolete]`) are out of scope (listed briefly in [10. Legacy / Deprecated APIs](#10-legacy--deprecated-apis)).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [AffectCore — Emotion Analysis Engine](#3-affectcore--emotion-analysis-engine)
4. [SpeakerRole (enum)](#4-speakerrole-enum)
5. [Result Classes](#5-result-classes)
6. [Converting to AffectAnalysisResult](#6-converting-to-affectanalysisresult)
7. [TranslationService — Translation Dictionary Customization](#7-translationservice--translation-dictionary-customization)
8. [Configuration (affect.config)](#8-configuration-affectconfig)
9. [Extension Point: IEmotionFilter](#9-extension-point-iemotionfilter)
    - [9.5 Extension Point: the Classified event](#95-extension-point-the-classified-event)
10. [Legacy / Deprecated APIs](#10-legacy--deprecated-apis)
11. [Label Reference](#11-label-reference)
12. [Common Usage Patterns](#12-common-usage-patterns)

---

## 1. Overview

Yakumo Affect Engine is a **14-label multi-label emotion analysis** library for Japanese and English text (.NET 8.0 / C# 12.0).

- Inference engine: GoEmotions RoBERTa (ONNX Runtime, DirectML GPU support)
- Japanese input is translated to English by a built-in translation service (local Python process) before inference
- The 28 GoEmotions output labels are aggregated into 14 Yakumo-specific labels

### Getting started (download & setup)

Download the ZIP from the [**Releases page**](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases),
extract it, and run `install.bat`. It downloads the ONNX models, sets up the Python
environment, and places `affect.config` for you.

- Prerequisites: [.NET 8.0 Runtime](https://dotnet.microsoft.com/download/dotnet/8.0) and Python 3.9+
- See the bundled [README](https://github.com/dr-yakumo/Yakumo.AffectEngine#readme) and `install.ps1` for details

---

## 2. Quick Start

```csharp
using Yakumo.Affect;

// Initialize in Japanese mode (affect.config is loaded automatically)
using var engine = new AffectCore(language: "jp", debugMode: false);

// Analyze — the engine is selected automatically based on the Model config key
var result = await engine.AnalyzeTextWithAutoModelAsync(
    "欲しかったグラボが激安で買えて本当に嬉しい！",
    SpeakerRole.User);

Console.WriteLine(result.TopEmotion);   // "喜び" (joy)
Console.WriteLine(result.TopScore);     // 0.809
Console.WriteLine(result.IsSurprised);  // false

foreach (var kv in result.TopK)
    Console.WriteLine($"{kv.Key} = {kv.Value:F3}");
    // 喜び = 0.809   (joy)
    // 期待 = 0.041   (anticipation)
    // 欲望 = 0.034   (desire)
```

> 📊 **These scores are measured with the default setup and are not fixed values.**
> They shift as you adjust label weights in `affect.config` or entries in
> `affect.dict.json`. Because label weights are multipliers, **a score can exceed 1.0** —
> it is not a probability. Treat the **ranking** as the primary output and the numbers
> as relative confidence.

> 💡 A working interactive sample is available at `Yakumo.Affect.Sample/Program.cs`.

> ⚠️ **Runtime assets are required.** The code above needs the assets set up by `install.bat` / `install.ps1`, which ship in the ZIP on the [Releases](https://github.com/dr-yakumo/Yakumo.AffectEngine/releases) page.
> Referencing the library alone is not enough. If you are not using the installer, prepare these manually:
>
> | Asset | Purpose |
> |---|---|
> | `affect.config` | Engine configuration. Copy from `affect.config.default` |
> | `libs/models/goemo-roberta-base/` | ONNX classification model (`model.onnx`, tokenizer files) |
> | Python translation server | **Required for Japanese input.** Started automatically by the engine; needs Python 3.9+ and the packages listed in `requirements.txt` |
> | `Yakumo.Affect.PolarityGate.dll` | Optional. Place it next to the executable to enable the polarity filter |
>
> Paths are resolved relative to the executable's directory (`AppContext.BaseDirectory`).

> 📊 **Scores are not fixed values.** They shift as you tune the engine for your own domain —
> label weights in `affect.config` and entries in `affect.dict.json` are both meant to be adjusted.
> Treat the **ranking** as the primary output and the numbers as relative confidence.

---

## 3. AffectCore — Emotion Analysis Engine

**Implements**: `IDisposable`
The entry point for emotion analysis. Call `Dispose()` (or use a `using` declaration) when finished to release the model sessions.

> ⚠️ **Do NOT use `AnalyzeText` / `AnalyzeTextAsync` / `AnalyzeTexts` / `AnalyzeTextsAsync`.**
> These are legacy APIs wired exclusively to the old BERT sentiment model — **they never reach the GoEmotions model** (calling them runs inference on the older, less accurate path).
> Always use **`AnalyzeTextWithAutoModelAsync` / `AnalyzeTextsWithAutoModelAsync`** instead.

### 3.1 Constructor

```csharp
public AffectCore(
    string language = "jp",
    bool debugMode = false,
    double temperature = 0.3,
    string scoreMode = "ent_minus_neu",
    string aggregate = "max")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `string` | `"jp"` | `"jp"` or `"en"`; anything else throws `ArgumentException` |
| `debugMode` | `bool` | `false` | `true` prints detailed logs to the console |
| `temperature` | `double` | `0.3` | Softmax temperature (NLI paths only) |
| `scoreMode` | `string` | `"ent_minus_neu"` | NLI score computation mode (NLI paths only) |
| `aggregate` | `string` | `"max"` | Prompt aggregation method (NLI paths only) |

> If `temperature` / `scoreMode` / `aggregate` are present under `[nli_model]` in `affect.config`, the **config values override** the constructor arguments. None of these three are used by the GoEmotions path.

> 📌 **`language` is an operating mode fixed at instance creation.** It does not auto-detect the input language, and it cannot be switched per analysis call.
>
> | | `"jp"` mode | `"en"` mode |
> |---|---|---|
> | Translation | Every input goes through the translation step, regardless of content | No translation (input is inferred as-is) |
> | Result label names | Japanese display names (「喜び」 etc.) | Canonical English names (`joy` etc.) |
>
> For workloads that mix Japanese and English text, use a separate instance per language (`CreateJapanese()` / `CreateEnglish()`).

### 3.2 Factory Methods

```csharp
public static AffectCore CreateDefault()   // Japanese + default settings
public static AffectCore CreateJapanese()  // equivalent to CreateDefault
public static AffectCore CreateEnglish()   // English settings
```

### 3.3 Properties

| Property | Type | Description |
|---|---|---|
| `Language` | `string` | `"jp"` or `"en"` |
| `DebugMode` | `bool` | Debug mode state |
| `ModelName` | `string` | The `[nli_model] Model` value from `affect.config` (e.g. `"goemo-roberta-base"`) |

### 3.4 Analysis Methods (Recommended Entry Points)

```csharp
// Single text
public Task<ClassificationResult> AnalyzeTextWithAutoModelAsync(
    string text,
    SpeakerRole role = SpeakerRole.User,
    int k = 3,
    double threshold = 0.25,
    string[]? properNouns = null)

// Multiple texts
public Task<NliAnalysisResults> AnalyzeTextsWithAutoModelAsync(
    string[] texts,
    SpeakerRole role = SpeakerRole.User,
    int k = 3,
    double threshold = 0.25)
```

| Parameter | Description |
|---|---|
| `text` / `texts` | Text to analyze. Goes through translation in `"jp"` mode; inferred as-is in `"en"` mode (see [3.1](#31-constructor)) |
| `role` | Speaker role ([Section 4](#4-speakerrole-enum)). Role-specific weights are applied |
| `k` | Number of top emotions to return (Top-K) |
| `threshold` | Surprise detection threshold. Overridden by `SurpriseThreshold` in `affect.config` when present |
| `properNouns` | Words listed here are replaced with `"entity"` before translation, reducing mistranslation of proper nouns and their impact on scores |

The internal engine is selected automatically based on the `[nli_model] Model` value in `affect.config`:

| Model value | Inference engine | Status |
|---|---|---|
| `goemo-roberta-base` | GoEmotions direct classification | **Recommended (current)** |
| `roberta-large-mnli` | RoBERTa-MNLI NLI | Legacy (research / fallback) |
| anything else | BERT sentiment | Legacy |

### 3.5 Model Check / Low-Level Methods

```csharp
// Returns whether the current configuration selects the GoEmotions model
public bool ShouldUseGoEmotionsModel()

// Calls the GoEmotions path directly
// (normally use AnalyzeTextWithAutoModelAsync instead)
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

Releases all held ONNX sessions. Using a `using` declaration is recommended.

---

## 4. SpeakerRole (enum)

```csharp
public enum SpeakerRole
{
    User,    // Human (user) utterance — default
    AI,      // AI assistant response
    System   // System message
}
```

Role-specific weights are applied to some label scores (for AI/System utterances, "surprise" is suppressed and "neutral" is favored).

| Label | User | AI | System |
|---|---|---|---|
| 驚き / surprise | 1.0 | **0.3** | **0.1** |
| 中立 / neutral | 1.0 | **1.5** | **2.0** |
| 喜び / joy | 1.0 | **0.8** | 1.0 |

---

## 5. Result Classes

### 5.1 ClassificationResult

The return value of `AnalyzeTextWithAutoModelAsync`.

| Property | Type | Description |
|---|---|---|
| `TopEmotion` | `string` | Highest-scoring emotion label (language-specific display name). Neutral when TopK is empty |
| `TopScore` | `double` | Highest score value |
| `Scores` | `Dictionary<string, double>` | Score dictionary for all labels (sigmoid-based × weights; may exceed 1.0 due to internal promotion adjustments) |
| `TopK` | `List<KeyValuePair<string, double>>` | Top K entries (post-filtering, descending by score) |
| `IsSurprised` | `bool` | Surprise detection flag |
| `SurpriseScore` | `double` | Score of the surprise label |
| `Threshold` | `double` | Threshold used for surprise detection |
| `Text` | `string` | Text used for inference (the translated English text for Japanese input) |
| `OriginalText` | `string` | Original input text |
| `Language` | `string` | A copy of the engine's `language` mode (see [3.1](#31-constructor)). Does not change per call, and is not a language-detection result |
| `Role` | `SpeakerRole` | Speaker role |

```csharp
var result = await engine.AnalyzeTextWithAutoModelAsync("最悪だ、電車が遅れた");

Console.WriteLine(result.TopEmotion);              // "嫌悪" (disgust)
if (result.Text != result.OriginalText)
    Console.WriteLine($"Translated: {result.Text}");  // "This is awful. ..."
```

### 5.2 NliAnalysisResults

The return value of `AnalyzeTextsWithAutoModelAsync` (multiple texts).

| Member | Type | Description |
|---|---|---|
| `Results` | `List<ClassificationResult>` | Analysis result per text |
| `TotalCount` | `int` | Number of results |
| `SurpriseCount` | `int` | Number of texts flagged as surprised |
| `MostCommonEmotion` | `string` | Most frequent top emotion |
| `ProcessingTime` | `TimeSpan` | Processing time |
| `ModelUsed` | `string` | Name of the model used |
| `AnalyzedAt` | `DateTime` | Analysis timestamp |
| `GetSummary()` | `string` | Builds a summary string |

---

## 6. Converting to AffectAnalysisResult

`ClassificationResult` carries engine-internal state (such as `Role`). When persisting, serializing, or passing results across layers, converting to the decoupled DTO `AffectAnalysisResult` is recommended.

```csharp
using Yakumo.Affect;

ClassificationResult raw = await engine.AnalyzeTextWithAutoModelAsync(text);

// Convert a single result
AffectAnalysisResult dto = raw.ToAnalysisResult();

// Convert a batch at once
NliAnalysisResults batch = await engine.AnalyzeTextsWithAutoModelAsync(texts);
List<AffectAnalysisResult> dtos = batch.ToAnalysisResults();

// Get the score of a specific label (0.0 when the label is absent)
double joy = dto.GetEmotionScore("喜び");
```

The properties of `AffectAnalysisResult` are nearly identical to `ClassificationResult` (differences: it has no `Role`, and it adds `GetEmotionScore(string)`).

---

## 7. TranslationService — Translation Dictionary Customization

**Singleton**: `TranslationService.Instance`

Japanese-to-English translation runs automatically inside the engine, so you normally never call it directly.
The main external use case is **customizing the translation dictionaries**: registering set phrases, proper nouns, and mistranslation fixes improves translation quality (and therefore emotion analysis accuracy).

### 7.1 Dictionary Registration API (fluent)

```csharp
var ts = TranslationService.Instance;

ts.AddPhrase("眠いなぁ", "I'm sleepy.")            // Phrase dictionary (exact match; skips the translation API)
  .AddProperNoun("株式会社サンプル", "Sample Corporation")  // Proper noun map (substring replacement before translation)
  .AddCorrection("bird skin", "goosebumps");       // Post-translation correction (fixes mistranslation patterns)

// Bulk-load from a JSON file
ts.LoadDictionaryFromJson("affect.dict.json");
```

**Application precedence**: phrase dictionary (exact match) > translation cache > HTTP translation API

### 7.2 Dictionary JSON Format

Copy `affect.dict.json.example` and edit it:

```json
{
  "version": 1,
  "phraseDict":  { "眠いなぁ": "I'm sleepy." },
  "properNouns": { "株式会社サンプル": "Sample Corporation" },
  "corrections": { "bird skin": "goosebumps" }
}
```

> ⚠️ **The bundled dictionary is specific to the default `opus` model.**
> Its entries target the mistranslations that `Helsinki-NLP/opus-mt-ja-en` actually produces.
>
> ```
> 鳥肌が立つ  →  opus renders it literally as "bird skin"  →  corrections maps it to goosebumps
> ```
>
> **Changing `[nli_translation] Model` largely disables the bundled dictionary.**
>
> These entries are tuned to the mistranslations produced by **a specific revision** of
> `opus-mt-ja-en`. The installer pins that revision, so upstream updates will not silently
> make the corrections miss. **If you swap the model yourself, expect to rebuild the
> corrections dictionary as well.**
> A different model fails differently, so strings like `bird skin` never appear in the first
> place. If you switch models, **rebuild the dictionary against that model's output.**

**Replacement values in `properNouns` do not have to be English.**
You can also use it to rewrite a phrase that breaks translation into plainer Japanese with the
same meaning:

```json
"properNouns": {
  "株式会社サンプル": "Sample Corporation",
  "肩を落とす": "がっかりする"
}
```

Staying in Japanese avoids the re-translation that can corrupt injected English. Matching is
substring-based (`text.Replace`), so **each inflected form needs its own entry** —
`肩を落とす` does not match `肩を落とした`.

### 7.3 Translation API (for direct use)

```csharp
public Task<string> TranslateAsync(string japaneseText, string[]? properNouns = null)
public string Translate(string japaneseText, string[]? properNouns = null)  // sync wrapper (backward compatibility)
```

On first access, a local Python translation server is started automatically (using two ports, `BasePort` and `BasePort+1`; default 5000/5001). The first startup may take tens of seconds to a few minutes for model loading.

---

## 8. Configuration (affect.config)

Configuration uses INI format. Copy `affect.config.default` to `affect.config` and edit it (`install.ps1` performs this copy automatically).

**Load order (low → high priority)**:

1. `AIClient.Config` (shared settings)
2. `affect.config` (the legacy name `NLI.Config` also works for backward compatibility)
3. The file pointed to by the `AFFECT_CONFIG` environment variable (legacy name `NLI_CONFIG` also works)

### 8.1 Key Quick Reference

```ini
[nli_model]
Model = goemo-roberta-base   ; recommended; roberta-large-mnli / bert_sentiment are legacy
ModelPrecision = fp32        ; fp32 | fp16 (fp16 recommended only with GPU)
UseDirectML = false          ; true enables DirectML GPU inference (falls back to CPU when unavailable)
DirectMLDeviceId = 0
SurpriseThreshold = 0.50     ; surprise detection threshold
SurpriseDelta = 0.10

[nli_emotion]
LabelSet = auto              ; basic(7) | extended(14) | auto (same as extended on the GoEmotions path)
EnableFiltering = true       ; polarity filter (see Section 9)
LabelWeight.sadness = 1.20   ; per-label score weight (canonical English names)
GoEmoLabelMode = yakumo14    ; yakumo14 | raw28 (see 8.4; experimental)

[nli_translation]
Model = opus                 ; opus | nllb | mt5
Quality = high
TimeoutSeconds = 180
BasePort = 5000

[embedding]
Rescoring.Enabled = false    ; kNN similar-case-vote rescoring (requires a prebuilt index)
```

For detailed explanations of every key, see the comments inside `affect.config.default`.

> **About the PolarityGate plugin**: there is no dedicated config section. It is enabled/disabled by (a) the presence of `Yakumo.Affect.PolarityGate.dll` and (b) `[nli_emotion] EnableFiltering` (see [Section 9](#9-extension-point-iemotionfilter)). The number of Top-K emotions returned is controlled not by a config key but by the analysis API argument `k` (default 3).

### 8.2 Translation Models and Licenses

| Value | Model | Characteristics | License |
|---|---|---|---|
| `opus` (default) | Helsinki-NLP/opus-mt-ja-en | Light & fast | Apache-2.0 ✅ commercial OK |
| `nllb` | facebook/nllb-200-distilled-600M | High accuracy, contextual | CC-BY-NC-4.0 ⚠️ **NON-COMMERCIAL ONLY** |
| `mt5` | google/mt5-small | ⚠️ **Experimental / unsupported** | Apache 2.0 |

> ⚠️ **`mt5` currently produces no usable translation.**
> `google/mt5-small` is pretrained on span corruption only and was never fine-tuned for
> translation, so it returns just its internal sentinel token (`<extra_id_0>`) regardless of
> the input (measured 2026-09-01).
> **The path is kept only for a future replacement model. Use `opus`.**

### 8.3 Programmatic Access to Config Values

```csharp
using Yakumo.Affect.Config;

string model = AffectConfigManager.Get("nli_model", "Model", "goemo-roberta-base");
int port     = AffectConfigManager.GetInt("nli_translation", "BasePort", 5000);
double th    = AffectConfigManager.GetDouble("nli_model", "SurpriseThreshold", 0.50);
```

The layered load described above runs automatically on the first `Get*` call.

### 8.4 GoEmoLabelMode — raw28 Mode (experimental / advanced)

Setting `[nli_emotion] GoEmoLabelMode = raw28` **skips** the 28→14 label aggregation and returns the raw sigmoid scores of the original 28 GoEmotions labels. The keys of `ClassificationResult.Scores` / `TopK` become the following 28 English label names:

```
admiration, amusement, anger, annoyance, approval, caring, confusion,
curiosity, desire, disappointment, disapproval, disgust, embarrassment,
excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
pride, realization, relief, remorse, sadness, surprise, neutral
```

This mode targets analyses where 14 labels are too coarse, or research that needs the model's raw output. For normal use, the default `yakumo14` is recommended.

**Restrictions (none of the following apply in raw28)**:

| Item | Behavior in raw28 |
|---|---|
| Japanese label names | ❌ Not supported. **Label names stay in English** even with `language="jp"` (no Japanese dictionary exists for the 28 labels) |
| Label weights (`LabelWeight.*`) | ❌ Not applied (pure sigmoid raw scores) |
| Role-based weights (`SpeakerRole`) | ❌ Not applied |
| Label promotion adjustments (internal post-processing) | ❌ Not applied |
| Embedding rescoring | ❌ Disabled (does not run even with `Rescoring.Enabled=true`) |
| Japanese→English input translation | ✅ **Runs as usual** (label mode and translation are independent) |
| Surprise detection (`IsSurprised`) | ✅ Works (always keyed on the English `"surprise"` label) |

> ⚠️ **The Top-K filter still runs in raw28.** However, polarity detection assumes the 14-label set, so against 28 labels it functions only partially depending on the language setting, and results may be distorted (in particular, with `language="en"` the polarity filter applies to only a subset of labels).
> **To get undistorted raw28 scores, set `[nli_emotion] EnableFiltering = false`** (or run without `Yakumo.Affect.PolarityGate.dll`).

When started with `language="jp"` + raw28, a one-time console notice explains that translation and label mode are independent.

### 8.5 Tuning with Label Weights (LabelWeight)

A multiplier specified via `LabelWeight.<canonical name>` under `[nli_emotion]` is applied to each emotion label's final score. This is the primary tuning mechanism for external users — **you can make specific emotions more or less likely to surface without any code changes**.

```ini
[nli_emotion]
; Values are multipliers: 1.0 = unchanged / 0.5 = halved / 1.5 = boosted
LabelWeight.neutral  = 0.65   ; suppress the "escape to neutral" tendency (lets emotion labels surface)
LabelWeight.sadness  = 1.20   ; strengthen sadness detection
LabelWeight.surprise = 0.85   ; curb surprise over-detection
```

- Key names use the **canonical English names** (the 14 labels in [Section 11](#11-label-reference)). Use English names even in `"jp"` mode (they are mapped to the Japanese display names internally)
- Weights are applied after the 28→14 aggregation and before Top-K filtering, so they directly affect the `TopEmotion` / `TopK` ranking
- Values ≤ 0 or non-numeric values are ignored and the built-in defaults are used
- The template (`affect.config.default`) ships with tuned recommended values. Start with those and fine-tune to your application's tendencies
- **When changes take effect**: the config file is loaded once per process at startup. Restart your application after editing
- Role-based weights ([Section 4](#4-speakerrole-enum)) are fixed built-ins with no corresponding config keys
- Label weights are not applied in raw28 mode ([Section 8.4](#84-goemolabelmode--raw28-mode-experimental--advanced))

> 💡 The **promotion thresholds** (`DisgustPromotion.Threshold` / `AngerPromotion.Threshold` / `SurprisePromotion.Threshold`), which lift emotion signals that tend to get buried during aggregation, can also be tuned in the same section. Lower values promote more easily (at the risk of more false positives). See the comments in `affect.config.default` for details.

---

## 9. Extension Point: IEmotionFilter

The post-filter applied to scores before the Top-K is finalized is pluggable.

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

At startup, the engine selects a filter in this order:

1. If **`Yakumo.Affect.PolarityGate.dll`** (an optional binary-only plugin, Apache 2.0) exists in the same directory as the executable, it is loaded
   — it resolves conflicts when positive and negative emotions collide within the Top-K
2. If absent, or if loading fails, the engine falls back to **`DefaultTopKFilter`** (simple descending Top-K)

All filter control lives in the `[nli_emotion]` section (there is no dedicated PolarityGate section):

- `EnableFiltering = false` — disables filtering entirely (simple Top-K even when the DLL is present)
- The PolarityGate tuning keys under `[nli_emotion]` (`NeutralizeConflicts` etc.) are for advanced use. **The plugin's internals are intentionally undocumented** — leave them at their defaults unless you have a specific reason to change them
- The number of returned entries is the analysis API argument `k` (not a config key)

To implement your own filtering strategy, implement `IEmotionFilter` (`DefaultTopKFilter` is the reference implementation).

---

## 9.5 Extension Point: the Classified event

Raised once per completed analysis. **Nothing happens unless you subscribe.**

```csharp
public event Action<ClassificationResult>? Classified;
```

Its main use is finding inputs whose emotion was destroyed by translation.
For Japanese input the `ClassificationResult` carries both the source and the translation.

| Property | Contents for Japanese input |
|---|---|
| `OriginalText` | The original Japanese text |
| `Text` | **The translated English** (what the classifier actually saw) |
| `Scores` / `TopK` | Scores across the 14 labels |

Japanese is classified via translation, so **a literally translated idiom loses its emotion
entirely**. Observed examples:

```
腹の虫が治まらない → "the insect in my stomach"
鳥肌が立つ         → "a bird's skin"
肩を落とす         → "I lost my shoulders"
```

In normal operation the English is never seen, so all you notice is that the result came back
neutral for no apparent reason. Capturing the translation here lets you feed the findings back
into `affect.dict.json`.

```csharp
using var core = AffectCore.Create();

core.Classified += result =>
{
    if (result.Language != "jp")
        return;

    // Japanese left in the English output means the translation failed
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

> **⚠️ Privacy**
> This event exposes user input. **The library itself records nothing.**
> Whether to record it, where to write it, and when to delete it are entirely the
> caller's responsibility. Bear in mind that input text may contain personal data.

**Behaviour notes**:

- The event is raised **synchronously**; heavy work in the handler slows down analysis
- An exception thrown by a handler **does not stop the analysis** (it is swallowed)
- Batch analysis (`AnalyzeTextsWithAutoModelAsync`) raises it **once per item**
- With no handlers attached, the overhead is negligible

---

## 10. Legacy / Deprecated APIs

The following remain public for backward compatibility but **must not be used in new code**.

| API | Status | Replacement |
|---|---|---|
| `NLI_Core` class | `[Obsolete]` (old class-name alias; will be removed) | `AffectCore` |
| `AnalyzeText` / `AnalyzeTextAsync` | Legacy (old BERT path only; never reaches GoEmotions) | `AnalyzeTextWithAutoModelAsync` |
| `AnalyzeTexts` / `AnalyzeTextsAsync` | Legacy (same as above) | `AnalyzeTextsWithAutoModelAsync` |
| `ClassifyByNlpAsync` / `ShouldUseRobertaModel` | Legacy (RoBERTa-MNLI path; research / fallback) | GoEmotions path |
| `SetEvalTotalSamples` | Evaluation tooling (not for external use) | — |
| `PythonTranslator` | `Deprecated` (hidden from editors) | `TranslationService` |

---

## 11. Label Reference

The 14 labels emitted by Yakumo Affect Engine.
The keys of `ClassificationResult.Scores` / `TopK` use the display name matching the `language` setting.

| canonical (EN) | Japanese display name | Polarity |
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

**Active labels per LabelSet mode** (`[nli_emotion] LabelSet`):

| Mode | Active labels |
|---|---|
| `basic` | Base 7 (surprise, anger, joy, sadness, fear, disgust, neutral) + optional positive extensions |
| `extended` | All 14 labels |
| `auto` | Same as `extended` on the GoEmotions path (all 14 labels; desire is controlled by `EnableDesire.Auto`) |

---

## 12. Common Usage Patterns

### Emotion tracking in a conversation system

```csharp
using var engine = new AffectCore("jp");

// Use different roles for user utterances vs. AI responses
var userEmotion = await engine.AnalyzeTextWithAutoModelAsync(userInput, SpeakerRole.User);
var aiEmotion   = await engine.AnalyzeTextWithAutoModelAsync(aiResponse, SpeakerRole.AI);
```

### Getting emotion scores for RAG integration

```csharp
var result = await engine.AnalyzeTextWithAutoModelAsync(userInput, SpeakerRole.User);

// Use all scores as an emotion delta
foreach (var kv in result.Scores)
    Console.WriteLine($"{kv.Key}: {kv.Value:F4}");

// Feed the top emotion into a prompt
string dominantEmotion = result.TopEmotion;
```

### Analyzing text that contains proper nouns

```csharp
// Prevents product names / person names from being mistranslated as emotion words
var result = await engine.AnalyzeTextWithAutoModelAsync(
    "モモちゃんの新しいグッズが猫カフェで販売開始！",
    SpeakerRole.User,
    properNouns: new[] { "モモちゃん", "猫カフェ" });
```

### Batch analysis and aggregation

```csharp
string[] logs = LoadChatLogs();
var batch = await engine.AnalyzeTextsWithAutoModelAsync(logs, SpeakerRole.User);

Console.WriteLine(batch.GetSummary());
Console.WriteLine($"Most common emotion: {batch.MostCommonEmotion}");
Console.WriteLine($"Surprised: {batch.SurpriseCount}/{batch.TotalCount}");
```

---
