using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Yakumo.Affect
{
    /// <summary>
    /// GoEmotions (SamLowe/roberta-base-go_emotions) 直接分類パス
    /// Phase B: 28ラベル sigmoid → Yakumo 14ラベルマッピング
    /// </summary>
    public partial class AffectCore
    {
        // ── GoEmotions モデルパス ──
        private static readonly string GoEmoModelDir =
            Path.Combine(AppContext.BaseDirectory, "libs\\models", "goemo-roberta-base");
        private static readonly string GoEmoTokenizerPath =
            Path.Combine(GoEmoModelDir, "tokenizer.json");

        private InferenceSession? _goemoSession;
        private bool _goemoInitialized = false;
        private RobertaBpeTokenizer? _goemoTokenizer;
        private bool _goemoUseDirectML = false;
        private string _goemoModelPrecision = "fp32";

        // ── GoEmotions 28ラベル (config.json id2label 順序) ──
        private static readonly string[] GoEmo28Labels =
        {
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        };

        // ── GoEmotions 28 → Yakumo 14 マッピング (DataPrepPhase0.YakumoMapping 同等) ──
        private static readonly Dictionary<string, string[]> GoEmoToYakumoMap = new()
        {
            ["joy"]          = new[] { "joy", "amusement", "relief" },
            ["sadness"]      = new[] { "sadness", "grief", "disappointment" },
            ["anger"]        = new[] { "anger", "annoyance" },
            ["fear"]         = new[] { "fear", "nervousness" },
            ["disgust"]      = new[] { "disgust" },
            ["surprise"]     = new[] { "surprise" },
            // realization を neutral に置くのは v1→v2 の意図的な修正（巻き戻さないこと）。
            // 「気づき」は日本語では単独で感情として評価するとオーバーリアクションになるため、
            // neutral を sink として抑制する設計。旧 v1 (data/mapping/label_mapping.json、
            // 2025-12 のゼロショットNLI時代の遺物・現在どのコードからも読まれていない) では
            // surprise 側に入っていたが、GoEmotions 直接分類の導入時に現在の配置へ移された。
            // surprise への寄与は下の SurprisePromo が composite 経由で行う（単独昇格はさせない）。
            ["neutral"]      = new[] { "neutral", "realization" },
            ["affection"]    = new[] { "love", "caring" },
            ["trust"]        = new[] { "approval", "pride", "gratitude", "admiration" },
            ["anticipation"] = new[] { "excitement", "optimism", "curiosity" },
            ["resentment"]   = new[] { "disapproval", "remorse" },
            ["disbelief"]    = new[] { "confusion" },
            ["shame"]        = new[] { "embarrassment" },
            ["desire"]       = new[] { "desire" },
        };

        // GoEmotions ラベル名 → index 逆引き
        private static readonly Dictionary<string, int> GoEmoLabelIndex =
            GoEmo28Labels
                .Select((label, i) => (label, i))
                .ToDictionary(x => x.label, x => x.i, StringComparer.OrdinalIgnoreCase);

        // ── 公開エントリポイント ──

        /// <summary>
        /// GoEmotions モデルで感情分類を行います
        /// </summary>
        public async Task<ClassificationResult> ClassifyByGoEmotionsAsync(
            string text,
            SpeakerRole role = SpeakerRole.User,
            int k = 3,
            double threshold = 0.25,
            string[]? properNouns = null)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AffectCore));
            if (!_goemoInitialized)
                await InitializeGoEmotionsModelAsync();

            // 進捗表示
            if (_evalTotalSamples > 0)
            {
                _evalCurrentSample++;
                if (_evalCurrentSample % 10 == 0 || _evalCurrentSample == _evalTotalSamples)
                {
                    Console.WriteLine($"[PROGRESS] {_evalCurrentSample}/{_evalTotalSamples} " +
                                      $"({100.0 * _evalCurrentSample / _evalTotalSamples:F1}%)");
                }
            }

            return await PerformGoEmotionsClassificationAsync(text, role, k, threshold, properNouns);
        }

        /// <summary>
        /// GoEmotions モデル使用判定
        /// </summary>
        public bool ShouldUseGoEmotionsModel()
        {
            try
            {
                string modelType = Config.AffectConfigManager.Get("nli_model", "Model", "bert_sentiment");
                return modelType.Equals("goemo-roberta-base", StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// nli_emotion.GoEmoLabelMode が raw28 かどうか
        /// </summary>
        internal static bool IsGoEmoRaw28Mode()
        {
            return Config.AffectConfigManager
                .Get("nli_emotion", "GoEmoLabelMode", "yakumo14")
                .Equals("raw28", StringComparison.OrdinalIgnoreCase);
        }

        // ── GoEmotions モデル初期化 ──

        private async Task InitializeGoEmotionsModelAsync()
        {
            if (_goemoInitialized) return;
            await Task.Run(() =>
            {
                try
                {
                    DebugWriteLine("GoEmotions (roberta-base-go_emotions) 初期化中...");

                    _goemoModelPrecision = Config.AffectConfigManager.Get("nli_model", "ModelPrecision", "fp32");
                    string onnxPath = GetGoEmoOnnxPath();

                    RequireFile(onnxPath);
                    RequireFile(GoEmoTokenizerPath);

                    _goemoTokenizer = RobertaBpeTokenizer.Load(GoEmoTokenizerPath);

                    var opt = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                    };

                    // DirectML 設定（[nli_model] セクション共用）
                    _goemoUseDirectML = Config.AffectConfigManager
                        .Get("nli_model", "UseDirectML", "false")
                        .Equals("true", StringComparison.OrdinalIgnoreCase);
                    int deviceId = Config.AffectConfigManager.GetInt("nli_model", "DirectMLDeviceId", 0);

                    if (_goemoUseDirectML)
                    {
                        try
                        {
                            opt.AppendExecutionProvider_DML(deviceId);
                            DebugWriteLine($"GoEmo: DirectML (GPU deviceId={deviceId}) プロバイダーを有効化");
                        }
                        catch (Exception ex)
                        {
                            DebugWriteLine($"GoEmo: DirectML利用不可、CPUにフォールバック: {ex.Message}");
                            _goemoUseDirectML = false;
                            ConfigureCpuOptions(opt);
                        }
                    }
                    else
                    {
                        ConfigureCpuOptions(opt);
                    }

                    _goemoSession = new InferenceSession(onnxPath, opt);
                    _goemoInitialized = true;

                    DebugWriteLine($"GoEmotions 初期化完了 (Precision={_goemoModelPrecision}, DirectML={_goemoUseDirectML})");
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"GoEmotions初期化失敗: {ex.Message}", ex);
                }
            });
        }

        private string GetGoEmoOnnxPath()
        {
            string fileName = _goemoModelPrecision.ToLowerInvariant() switch
            {
                "fp16" => "model_fp16.onnx",
                _ => "model.onnx"
            };
            return Path.Combine(GoEmoModelDir, fileName);
        }

        // ── GoEmotions 分類本体 ──

        private async Task<ClassificationResult> PerformGoEmotionsClassificationAsync(
            string text,
            SpeakerRole role,
            int k,
            double threshold,
            string[]? properNouns)
        {
            string original = text;

            // 翻訳 (JP → EN)
            AffectTuning.Language = _language;
            if (_language == "jp")
            {
                text = await TranslationService.Instance.TranslateAsync(text, properNouns);
                DebugWriteLine($"[TRANS] {original} -> {text}");

                // 翻訳が失敗した場合、その出力を分類しても意味のある感情は出ない。
                // 中立を返して打ち切る（詳細は IsTranslationFailure を参照）。
                if (IsTranslationFailure(text))
                {
                    DebugWriteLine($"[TRANS][FAIL] 翻訳失敗と判定 -> 中立を返します: {original}");
                    return BuildNeutralResult(original, text, role, threshold);
                }
            }

            // ── トークン化（単一テキスト: <s> tokens </s>）──
            var inputIds = _goemoTokenizer!.Encode(text, maxLength: 512);
            int seqLen = inputIds.Length;

            var idsT = new DenseTensor<long>(new[] { 1, seqLen });
            var mskT = new DenseTensor<long>(new[] { 1, seqLen });
            for (int i = 0; i < seqLen; i++)
            {
                idsT[0, i] = inputIds[i];
                mskT[0, i] = 1L;   // 全トークン有効（パディングなし）
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idsT),
                NamedOnnxValue.CreateFromTensor("attention_mask", mskT)
            };

            // ── 推論実行 ──
            float[] rawLogits;
            var sw = Stopwatch.StartNew();

            if (_goemoUseDirectML)
            {
                // DirectML: 呼び出しスレッドで同期実行（GPU スレッド安全）
                using var results = _goemoSession!.Run(inputs);
                rawLogits = results.First().AsEnumerable<float>().ToArray();
            }
            else
            {
                rawLogits = await Task.Run(() =>
                {
                    using var results = _goemoSession!.Run(inputs);
                    return results.First().AsEnumerable<float>().ToArray();
                });
            }

            sw.Stop();
            if (_debugMode)
                DebugWriteLine($"[GoEmo][PERF] inference={sw.ElapsedMilliseconds}ms tokens={seqLen}");

            // ── Sigmoid（multi-label classification）──
            var probs28 = new double[rawLogits.Length];
            for (int i = 0; i < rawLogits.Length; i++)
                probs28[i] = Sigmoid(rawLogits[i]);

            if (_debugMode)
            {
                var sorted = probs28
                    .Select((p, i) => (label: GoEmo28Labels[i], prob: p))
                    .OrderByDescending(x => x.prob)
                    .Take(5);
                Debug.WriteLine($"[GoEmo] top5(28): {string.Join(", ", sorted.Select(x => $"{x.label}={x.prob:F3}"))}");
            }

            // ── ラベルモード分岐 (raw28: GoEmotions 28ラベルを生スコアのまま返す) ──
            bool isRaw28 = IsGoEmoRaw28Mode();

            if (isRaw28)
            {
                Console.WriteLine("[GoEmo][WARN] raw28: 14-label compression skipped");
            }

            Dictionary<string, double> scores;

            if (isRaw28)
            {
                // raw28: Promo/ラベル重み/ロール重み/JP訳は未適用（研究・実験用途の生スコア出力）
                scores = new Dictionary<string, double>(GoEmo28Labels.Length, StringComparer.OrdinalIgnoreCase);
                for (int i = 0; i < GoEmo28Labels.Length; i++)
                    scores[GoEmo28Labels[i]] = probs28[i];
            }
            else
            {
                scores = PerformYakumo14Mapping(probs28, role);
            }

            // ── Embedding リスコアリング（Phase D で調整予定・raw28では無効）──
            var rescorer = Embedding.EmbeddingRescorer.Instance;
            if (!isRaw28 && rescorer.IsEnabled)
            {
                try
                {
                    await rescorer.InitializeAsync();
                    scores = await rescorer.RescoreAsync(text, scores, _language, _debugMode);
                }
                catch (Exception ex)
                {
                    if (_debugMode) Debug.WriteLine($"[RESCORE][ERROR] {ex.Message}");
                }
            }

            // ── TopK フィルタリング ──
            var topK = FilterEmotions(scores, k);

            // ── 驚き判定 (raw28では常に英語キー "surprise" を使用) ──
            var surpriseLabel = isRaw28 ? "surprise" : (_language == "en" ? "surprise" : "驚き");
            scores.TryGetValue(surpriseLabel, out var surScore);

            double cfgThr = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseThreshold", threshold);
            double cfgDelta = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseDelta", 0.10);
            bool inTop = topK.Any(x => x.Key == surpriseLabel);
            double topScore = topK.Count > 0 ? topK[0].Value : 0;
            double secondScore = topK.Count > 1 ? topK[1].Value : double.NegativeInfinity;
            double relative = topScore > 0 ? surScore / topScore : 0;

            bool isSurprised =
                surScore >= cfgThr &&
                (inTop || relative >= 0.60 || (surScore - secondScore) >= cfgDelta);

            return new ClassificationResult
            {
                Text = text,
                OriginalText = original,
                Scores = scores,
                TopK = topK,
                SurpriseScore = surScore,
                IsSurprised = isSurprised,
                Threshold = cfgThr,
                Language = _language,
                Role = role
            };
        }

        // ── 翻訳失敗の検出 ──────────────────────────────────────────────
        //
        // OPUS-MT は字幕コーパス（OpenSubtitles 等）で学習しているため、
        // 訳しにくい入力に遭遇すると字幕の効果音表記だけを返すことがある。
        // 実測（配信チャット 224件・2026-09-07）:
        //
        //   88時間プレイして40ちょいになりました    -> "(Laughter)"
        //   〇〇さんの装備が今一番いいですよね    -> "(Laughter)"
        //   一通り漂流はしたから…こたえられるで      -> "(Laughter)"
        //
        // 6件すべてが「喜び 0.78」と判定された。訳せなかったことが
        // 高スコアの「喜び」として出力される、最も避けたい形の誤検出。
        //
        // 原文に共通する言い回しが無いので辞書では直せない（置換すべきキーが無い）。
        // 翻訳結果の側で検出して、感情判定を放棄するのが正しい。
        //
        // !!! 保守的に判定すること。効果音表記を取り除いた結果が空になる場合のみ
        //    失敗とみなす。"I laughed. (Laughter)" のように本文が残っていれば通常処理。

        /// <summary>字幕由来の効果音表記。これだけが残ったら翻訳は失敗している。</summary>
        private static readonly System.Text.RegularExpressions.Regex SubtitleArtifacts =
            new(@"\(\s*(Laughter|Applause|Music|Cheering|Sighs?|Laughs?)\s*\)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase
                | System.Text.RegularExpressions.RegexOptions.Compiled);

        /// <summary>
        /// 翻訳が失敗しているかを判定します。
        /// 効果音表記を除いて意味のある語が残らない場合のみ true。
        /// </summary>
        private static bool IsTranslationFailure(string translated)
        {
            if (string.IsNullOrWhiteSpace(translated))
                return true;

            string stripped = SubtitleArtifacts.Replace(translated, " ").Trim();

            // 記号だけが残った場合も内容が無い
            bool hasContent = false;
            foreach (char c in stripped)
            {
                if (char.IsLetterOrDigit(c))
                {
                    hasContent = true;
                    break;
                }
            }
            return !hasContent;
        }

        /// <summary>
        /// 翻訳失敗時に返す中立の結果を作ります。
        /// スコアは 0 にして「判定できなかった」ことが分かるようにします。
        /// </summary>
        private ClassificationResult BuildNeutralResult(
            string original, string translated, SpeakerRole role, double threshold)
        {
            string neutralLabel = IsGoEmoRaw28Mode()
                ? "neutral"
                : (_language == "en" ? "neutral" : "中立");

            var scores = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase)
            {
                [neutralLabel] = 0.0
            };

            return new ClassificationResult
            {
                Text = translated,
                OriginalText = original,
                Scores = scores,
                TopK = new List<KeyValuePair<string, double>>
                {
                    new(neutralLabel, 0.0)
                },
                SurpriseScore = 0.0,
                IsSurprised = false,
                Threshold = threshold,
                Language = _language,
                Role = role
            };
        }

        // ── 28 → 14 マッピング + Promo + ラベル重み・ロール重み・ローカライズ (yakumo14 モード) ──
        private Dictionary<string, double> PerformYakumo14Mapping(double[] probs28, SpeakerRole role)
        {
            // ── 28 → 14 マッピング（各グループ内 max 採用）──
            var scores14en = new Dictionary<string, double>(GoEmoToYakumoMap.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var (yakumoLabel, goemoLabels) in GoEmoToYakumoMap)
            {
                double maxProb = 0;
                foreach (var gl in goemoLabels)
                {
                    if (GoEmoLabelIndex.TryGetValue(gl, out int idx))
                        maxProb = Math.Max(maxProb, probs28[idx]);
                }
                scores14en[yakumoLabel] = maxProb;
            }

            if (_debugMode)
            {
                var sorted14 = scores14en.OrderByDescending(x => x.Value).Take(5);
                Debug.WriteLine($"[GoEmo] top5(14): {string.Join(", ", sorted14.Select(x => $"{x.Key}={x.Value:F3}"))}");
            }

            // ── Promo 共通の top1 スナップショット ──
            // 各 Promo が個別に top1 を再評価すると、先に走った Promo の昇格結果が
            // 後続 Promo のゲート条件を書き換えてしまう（例: DisgustPromo が disgust を
            // neutral より上に押し上げると、SurprisePromo の "top1 == neutral" が
            // 成立しなくなり surprise が永久に昇格できない）。
            // Promo は「マッピング直後の素の top1」に対する独立した補正であるべきなので、
            // 昇格前の top1 を1回だけ確定して全 Promo で共有する。
            string promoBaseTop14 = scores14en.OrderByDescending(x => x.Value).First().Key;

            // ── disgust 昇格ロジック (annoyance + disapproval 集約) ──
            {
                double disgust28     = probs28[GoEmoLabelIndex["disgust"]];
                double annoyance28   = probs28[GoEmoLabelIndex["annoyance"]];
                double disapproval28 = probs28[GoEmoLabelIndex["disapproval"]];

                // disgust 最低シグナル条件:
                // 物理的嫌悪は disgust28 が低くても annoyance+disapproval で補完できるため
                // 0.03 まで緩和 (0.05 → 0.03)
                if (disgust28 >= 0.03)
                {
                    double compositScore = disgust28
                                         + annoyance28   * 0.40
                                         + disapproval28 * 0.35;

                    string top14Label = promoBaseTop14;
                    // sadness / fear も候補: 物理的嫌悪は GoEmotions が sadness/fear に流しやすい
                    bool topIsCandidate = top14Label is "anger" or "resentment" or "neutral"
                                                      or "sadness" or "fear";

                    double promotionThreshold = Config.AffectConfigManager
                        .GetDouble("nli_emotion", "DisgustPromotion.Threshold", 0.38);

                    if (topIsCandidate && compositScore >= promotionThreshold
                        && compositScore > scores14en["disgust"])
                    {
                        scores14en["disgust"] = compositScore;
                        if (_debugMode)
                            DebugWriteLine($"[DisgustPromo] 昇格: disgust28={disgust28:F3} " +
                                           $"annoyance={annoyance28:F3} disapproval={disapproval28:F3} " +
                                           $"→ composite={compositScore:F3} (top1was={top14Label})");
                    }
                }
            }

            // ── anger 昇格ロジック ──
            {
                double anger28       = probs28[GoEmoLabelIndex["anger"]];
                double annoyance28   = probs28[GoEmoLabelIndex["annoyance"]];
                double disapproval28 = probs28[GoEmoLabelIndex["disapproval"]];

                if (anger28 >= 0.05)
                {
                    double compositScore = anger28
                                         + annoyance28    * 0.45
                                         + disapproval28  * 0.30;

                    string top14Label = promoBaseTop14;
                    bool topIsCandidate = top14Label is "resentment";

                    double promotionThreshold = Config.AffectConfigManager
                        .GetDouble("nli_emotion", "AngerPromotion.Threshold", 0.38);

                    if (topIsCandidate && compositScore >= promotionThreshold
                        && compositScore > scores14en["anger"])
                    {
                        scores14en["anger"] = compositScore;
                        if (_debugMode)
                        {
                            DebugWriteLine($"[AngerPromo] 昇格: anger28={anger28:F3} " +   // ← DebugWriteLine に修正
                                           $"annoyance={annoyance28:F3} disapproval={disapproval28:F3} " +
                                           $"→ composite={compositScore:F3} (top1was={top14Label})");
                        }
                    }
                }
            }

            // ── surprise 昇格ロジック ──
            {
                double surprise28    = probs28[GoEmoLabelIndex["surprise"]];
                double realization28 = probs28[GoEmoLabelIndex["realization"]];

                if (surprise28 >= 0.05)
                {
                    double compositScore = surprise28 + realization28 * 0.50;

                    string top14Label = promoBaseTop14;
                    bool topIsCandidate = top14Label is "neutral";

                    double promotionThreshold = Config.AffectConfigManager
                        .GetDouble("nli_emotion", "SurprisePromotion.Threshold", 0.35);

                    if (topIsCandidate && compositScore >= promotionThreshold
                        && compositScore > scores14en["surprise"])
                    {
                        scores14en["surprise"] = compositScore;
                        if (_debugMode)
                        {
                            DebugWriteLine($"[SurprisePromo] 昇格: surprise28={surprise28:F3} " + // ← DebugWriteLine に修正
                                           $"realization={realization28:F3} " +
                                           $"→ composite={compositScore:F3} (top1was={top14Label})");
                        }
                    }
                }
            }

            // ── ローカライズ (en → localized) ──
            var locMap = AffectTuning.EmotionLabels.TryGetValue(_language, out var mapLoc)
                ? mapLoc
                : AffectTuning.EmotionLabels["jp"];

            var scores = new Dictionary<string, double>(scores14en.Count);
            foreach (var (canonEn, score) in scores14en)
            {
                string localized = _language == "jp"
                    ? (locMap.TryGetValue(canonEn, out var jp) ? jp : canonEn)
                    : canonEn;
                scores[localized] = score;
            }

            // ── ラベル重み・ロール重み適用 ──
            var labelWeights = GetLabelWeights();
            foreach (var key in scores.Keys.ToArray())
            {
                scores[key] = ApplyLabelWeight(scores[key], key, labelWeights);
                scores[key] = ApplyRoleBasedWeight(scores[key], key, role);
            }

            return scores;
        }

        // ── Sigmoid 関数（数値安定版）──
        private static double Sigmoid(float x)
        {
            if (x >= 0)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
            else
            {
                double ex = Math.Exp(x);
                return ex / (1.0 + ex);
            }
        }
    }
}