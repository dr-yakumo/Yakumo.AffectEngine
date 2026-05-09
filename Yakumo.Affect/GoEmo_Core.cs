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
                DebugWriteLine($"[GoEmo] top5(28): {string.Join(", ", sorted.Select(x => $"{x.label}={x.prob:F3}"))}");
            }

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
                DebugWriteLine($"[GoEmo] top5(14): {string.Join(", ", sorted14.Select(x => $"{x.Key}={x.Value:F3}"))}");
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

            // ── Embedding リスコアリング（Phase D で調整予定）──
            var rescorer = Embedding.EmbeddingRescorer.Instance;
            if (rescorer.IsEnabled)
            {
                try
                {
                    await rescorer.InitializeAsync();
                    scores = await rescorer.RescoreAsync(text, scores, _language, _debugMode);
                }
                catch (Exception ex)
                {
                    if (_debugMode) DebugWriteLine($"[RESCORE][ERROR] {ex.Message}");
                }
            }

            // ── TopK フィルタリング ──
            var topK = FilterEmotions(scores, k);

            // ── 驚き判定 ──
            var surpriseLabel = _language == "en" ? "surprise" : "驚き";
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