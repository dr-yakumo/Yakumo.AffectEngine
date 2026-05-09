using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace Yakumo.Affect
{
    [CodeStatus(CodeStatus.Legacy,
        Note = "RoBERTa-large-MNLI NLI inference path. Retained as research/fallback after migration to GoEmotions. " +
               "Primary path is GoEmo_Core.cs. / " +
               "RoBERTa-large-MNLI NLI推論パス。GoEmotions移行後の研究・フォールバック用として保持。" +
               "主系統はGoEmo_Core.cs。")]
    public partial class AffectCore
    {
        // RoBERTaパス
        private static readonly string RobertaModelDir = System.IO.Path.Combine(AppContext.BaseDirectory, "libs\\models", "roberta-large-mnli");
        private static readonly string RobertaTokenizerPath = System.IO.Path.Combine(RobertaModelDir, "tokenizer.json");

        private InferenceSession? _robertaSession;
        private bool _robertaInitialized = false;
        private RobertaBpeTokenizer? _robertaTokenizer;
        private bool _useDirectML = false;
        private string _modelPrecision = "fp32";  // 追加: 精度設定

        private static readonly HashSet<string> BasicCanonForRoberta = new(
            new[] { "surprise", "anger", "joy", "sadness", "fear", "disgust", "neutral" },
            StringComparer.OrdinalIgnoreCase);

        // 追加: 評価進捗カウンター
        private int _evalCurrentSample = 0;
        private int _evalTotalSamples = 0;

        /// <summary>
        /// 評価用の総サンプル数を設定します
        /// </summary>
        public void SetEvalTotalSamples(int total)
        {
            _evalTotalSamples = total;
            _evalCurrentSample = 0;
        }

        // 公開: RoBERTa経路（自動バッチ判定付き）
        public async Task<ClassificationResult> ClassifyByNlpAsync(
            string text,
            SpeakerRole role = SpeakerRole.User,
            int k = 3,
            double threshold = 0.25,
            string[]? properNouns = null)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AffectCore));
            if (!_robertaInitialized)
                await InitializeRobertaModelAsync();

            // 追加: 進捗表示
            if (_evalTotalSamples > 0)
            {
                _evalCurrentSample++;
                if (_evalCurrentSample % 10 == 0 || _evalCurrentSample == _evalTotalSamples)
                {
                    Console.WriteLine($"[PROGRESS] {_evalCurrentSample}/{_evalTotalSamples} ({100.0 * _evalCurrentSample / _evalTotalSamples:F1}%)");
                }
            }

            return await PerformRobertaClassificationAsync(text, _robertaSession!, role, k, threshold, properNouns);
        }

        // 追加: 精度設定に基づくモデルパス取得
        private string GetRobertaOnnxPath()
        {
            string fileName = _modelPrecision.ToLowerInvariant() switch
            {
                "fp16" => "model_fp16.onnx",
                _ => "model.onnx"
            };
            return System.IO.Path.Combine(RobertaModelDir, fileName);
        }

        // 初期化
        private async Task InitializeRobertaModelAsync()
        {
            if (_robertaInitialized) return;
            await Task.Run(() =>
            {
                try
                {
                    DebugWriteLine("RoBERTa-large-MNLI 初期化...");

                    // 追加: 精度設定を読み込み
                    _modelPrecision = Config.AffectConfigManager.Get("nli_model", "ModelPrecision", "fp32");
                    string onnxPath = GetRobertaOnnxPath();

                    RequireFile(onnxPath);
                    RequireFile(RobertaTokenizerPath);

                    _robertaTokenizer = RobertaBpeTokenizer.Load(RobertaTokenizerPath);

                    var opt = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                    };

                    // DirectML (AMD GPU) 設定を確認
                    _useDirectML = Config.AffectConfigManager.Get("nli_model", "UseDirectML", "false")
                        .Equals("true", StringComparison.OrdinalIgnoreCase);
                    int deviceId = Config.AffectConfigManager.GetInt("nli_model", "DirectMLDeviceId", 0);

                    if (_useDirectML)
                    {
                        try
                        {
                            opt.AppendExecutionProvider_DML(deviceId);
                            DebugWriteLine($"DirectML (GPU deviceId={deviceId}) プロバイダーを有効化");
                        }
                        catch (Exception ex)
                        {
                            DebugWriteLine($"DirectML利用不可、CPUにフォールバック: {ex.Message}");
                            _useDirectML = false;
                            ConfigureCpuOptions(opt);
                        }
                    }
                    else
                    {
                        ConfigureCpuOptions(opt);
                    }

                    _robertaSession = new InferenceSession(onnxPath, opt);
                    _robertaInitialized = true;

                    // 修正: 初期化完了時の情報表示
                    int batchSize = GetConfiguredBatchSize();
                    DebugWriteLine($"RoBERTa 初期化完了 (Precision={_modelPrecision}, DirectML={_useDirectML}, BatchSize={batchSize})");
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"RoBERTa初期化失敗: {ex.Message}", ex);
                }
            });
        }

        private static void ConfigureCpuOptions(SessionOptions opt)
        {
            opt.IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount);
            opt.InterOpNumThreads = 1;
        }

        // 追加: GPU/CPU別バッチサイズ取得
        private int GetConfiguredBatchSize()
        {
            if (_useDirectML)
            {
                return Config.AffectConfigManager.GetInt("nli_model", "RobertaBatchSize.GPU", 64);
            }
            else
            {
                return Config.AffectConfigManager.GetInt("nli_model", "RobertaBatchSize.CPU", 16);
            }
        }

        // メイン分類（A+B: config + 閾値）
        private async Task<ClassificationResult> PerformRobertaClassificationAsync(
            string text,
            InferenceSession session,
            SpeakerRole role,
            int k,
            double threshold,
            string[]? properNouns)
        {
            // ここを追加: RoBERTa用プロンプトを上書き
            var robertaPrompts = GetActivePromptsForRoberta();
            Console.WriteLine("[TRACE] RoBERTa ActivePrompts count=" + robertaPrompts.Count);
            foreach (var kv in robertaPrompts)
            {
                Console.WriteLine($"[TRACE]   {kv.Key}: {kv.Value.Length} prompts");
            }

            string original = text;

            // 翻訳
            AffectTuning.Language = _language;
            if (_language == "jp")
            {
                text = await TranslationService.Instance.TranslateAsync(text, properNouns);
                DebugWriteLine($"[TRANS] {original} -> {text}");
            }

            // ラベル辞書
            var enMap = AffectTuning.EmotionLabels.TryGetValue("en", out var mapEn) ? mapEn : new Dictionary<string, string>();
            var locMap = AffectTuning.EmotionLabels.TryGetValue(_language, out var mapLoc) ? mapLoc : AffectTuning.EmotionLabels["jp"];
            var activeLocalized = new HashSet<string>(AffectTuning.GetActiveLabelList(), StringComparer.Ordinal);

            var canonicalKeys = enMap.Keys
                .Where(c =>
                {
                    var localized = locMap.TryGetValue(c, out var l) ? l : c;
                    return activeLocalized.Contains(localized);
                })
                .ToArray();

            string scoreMode = Config.AffectConfigManager.Get("nli_model", "ScoreMode", _scoreMode);
            double temp = Config.AffectConfigManager.GetDouble("nli_model", "Temperature", _temperature);

            // プロンプト収集を robertaPrompts に差し替え（オーバーライド反映）
            var labelToPrompts = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            int totalSamples = 0;
            foreach (var labelEn in canonicalKeys)
            {
                // robertaPrompts（canonical英語キー）優先。なければ英語デフォルト1行をフォールバック
                if (!robertaPrompts.TryGetValue(labelEn, out var arr) || arr.Length == 0)
                    arr = new[] { $"This text expresses {labelEn}." };

                labelToPrompts[labelEn] = arr;
                totalSamples += arr.Length;
            }

            bool useBatch = ShouldUseRobertaBatch(totalSamples);
            if (_debugMode)
            {
                var modeCfg = Config.AffectConfigManager.Get("nli_model", "RobertaBatching", "auto");
                DebugWriteLine($"[MODE][RoBERTa] batching={modeCfg} samples={totalSamples} -> {(useBatch ? "batch" : "seq")}");
            }

            var rawScoresEn = useBatch
                ? await RunRobertaBatched(session, text, labelToPrompts, scoreMode, temp, role)
                : await RunRobertaSequential(session, text, labelToPrompts, scoreMode, temp, role);

            // キャリブレーションは削除（後で適用）

            // スコア後処理
            var labelWeights = GetLabelWeights();
            var scores = new Dictionary<string, double>(rawScoresEn.Count);

            foreach (var labelEn in canonicalKeys)
            {
                if (!rawScoresEn.TryGetValue(labelEn, out var baseScore)) continue;

                string localized = _language == "jp"
                    ? (locMap.TryGetValue(labelEn, out var jp) ? jp : labelEn)
                    : labelEn;

                baseScore = ApplyLabelWeight(baseScore, localized, labelWeights);
                baseScore = ApplyRoleBasedWeight(baseScore, localized, role);
                baseScore = ApplyEmotionThresholdEmphasis(baseScore, localized);

                if (activeLocalized.Contains(localized))
                    scores[localized] = baseScore;
            }

            // キャリブレーション適用（正規化付き）
            var calibStore = Calibration.EmotionCalibrationStore.Instance;
            calibStore.Load(verbose: _debugMode);
            if (calibStore.IsEnabled)
            {
                scores = calibStore.CalibrateAllCanonical(scores);
                if (_debugMode) DebugWriteLine("[CALIB] スコアキャリブレーション適用済み（正規化）");
            }

            // 追加: Embedding ベース再スコアリング (Phase 5-4 Step 7)
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

            var topK = FilterEmotions(scores, k);

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

        // バッチ利用判定 (Config + 閾値)
        private bool ShouldUseRobertaBatch(int totalSamples)
        {
            string mode = Config.AffectConfigManager.Get("nli_model", "RobertaBatching", "auto");
            int threshold = Config.AffectConfigManager.GetInt("nli_model", "RobertaBatchThreshold", 12);
            return mode.ToLowerInvariant() switch
            {
                "always" => true,
                "off" => false,
                "auto" => totalSamples >= threshold,
                _ => totalSamples >= threshold
            };
        }

        // 逐次
        private async Task<Dictionary<string, double>> RunRobertaSequential(
            InferenceSession session,
            string premise,
            Dictionary<string, string[]> labelToPrompts,
            string scoreMode,
            double temp,
            SpeakerRole role)
        {
            var raw = new Dictionary<string, double>(labelToPrompts.Count);
            var sw = Stopwatch.StartNew();
            foreach (var kv in labelToPrompts)
            {
                var per = new List<double>(kv.Value.Length);
                foreach (var hyp in kv.Value)
                {
                    var s = await ExecuteRobertaInferenceAsync(session, premise, hyp, role, temp, scoreMode);
                    per.Add(s);
                }
                double agg = GetAggregatedScore(per);
                raw[kv.Key] = agg;
                if (_debugMode)
                    DebugWriteLine($"[SEQ]{kv.Key}: {string.Join("/", per.Select(v => v.ToString("F3")))} => {agg:F3}");
            }
            sw.Stop();
            DebugWriteLine($"[PERF][RoBERTa] sequential samples={labelToPrompts.Sum(x => x.Value.Length)} time={sw.ElapsedMilliseconds}ms");
            return raw;
        }

        // チャンク分割対応
        private async Task<Dictionary<string, double>> RunRobertaBatched(
            InferenceSession session,
            string premise,
            Dictionary<string, string[]> labelToPrompts,
            string scoreMode,
            double temp,
            SpeakerRole role)
        {
            var map = new List<(string labelEn, int idx)>();
            var hyps = new List<string>();
            foreach (var kv in labelToPrompts)
            {
                for (int i = 0; i < kv.Value.Length; i++)
                {
                    map.Add((kv.Key, i));
                    hyps.Add(kv.Value[i]);
                }
            }
            if (hyps.Count == 0) return new Dictionary<string, double>();

            // バッチサイズをConfigから取得してチャンク分割
            int maxBatchSize = GetConfiguredBatchSize();
            var allLogits = new List<double[]>();

            var sw = Stopwatch.StartNew();

            // チャンク単位で処理
            for (int chunkStart = 0; chunkStart < hyps.Count; chunkStart += maxBatchSize)
            {
                int chunkEnd = Math.Min(chunkStart + maxBatchSize, hyps.Count);
                var chunkHyps = hyps.Skip(chunkStart).Take(chunkEnd - chunkStart).ToArray();

                (long[][] idsBatch, long[][] maskBatch) = TokenizeBatchForRoberta(premise, chunkHyps, Math.Min(AffectTuning.MaxTotal, 512));
                var inputs = CreateRobertaBatchInputTensors(idsBatch, maskBatch);

                // DirectML 使用時はスレッドプールを使わない（GPU スレッド安全性）
                double[][] chunkLogits;
                if (_useDirectML)
                {
                    using var results = session.Run(inputs);
                    var flat = results.First().AsEnumerable<float>().ToArray();
                    int batch = chunkHyps.Length;
                    const int dim = 3;
                    chunkLogits = new double[batch][];
                    for (int i = 0; i < batch; i++)
                    {
                        chunkLogits[i] = new double[dim];
                        for (int j = 0; j < dim; j++)
                            chunkLogits[i][j] = flat[i * dim + j];
                    }
                }
                else
                {
                    chunkLogits = await Task.Run(() =>
                    {
                        using var results = session.Run(inputs);
                        var flat = results.First().AsEnumerable<float>().ToArray();
                        int batch = chunkHyps.Length;
                        const int dim = 3;
                        var arr = new double[batch][];
                        for (int i = 0; i < batch; i++)
                        {
                            arr[i] = new double[dim];
                            for (int j = 0; j < dim; j++)
                                arr[i][j] = flat[i * dim + j];
                        }
                        return arr;
                    });
                }

                allLogits.AddRange(chunkLogits);
            }

            sw.Stop();
            Debug.WriteLine($"[PERF][RoBERTa] batched total={hyps.Count} batchSize={maxBatchSize} time={sw.ElapsedMilliseconds}ms");

            var perLabelPromptScores = labelToPrompts.ToDictionary(k => k.Key, v => new double[v.Value.Length]);
            for (int i = 0; i < allLogits.Count; i++)
            {
                var (lab, pIdx) = map[i];
                var probs = SoftmaxTemp(allLogits[i], temp);
                double s = ComputeMnliScore(probs, scoreMode);
                s = ApplyRoleSpecificAdjustment(s, premise, role);
                perLabelPromptScores[lab][pIdx] = s;
            }

            var raw = new Dictionary<string, double>(perLabelPromptScores.Count);
            foreach (var kv in perLabelPromptScores)
            {
                var lst = kv.Value.ToList();
                double agg = GetAggregatedScore(lst);
                raw[kv.Key] = agg;
                if (_debugMode)
                    DebugWriteLine($"[BAT]{kv.Key}: {string.Join("/", lst.Select(v => v.ToString("F3")))} => {agg:F3}");
            }
            return raw;
        }

        // バッチトークナイズ
        private (long[][] ids, long[][] mask) TokenizeBatchForRoberta(string premise, string[] hypotheses, int maxTotal)
        {
            var idsList = new long[hypotheses.Length][];
            var maskList = new long[hypotheses.Length][];
            for (int i = 0; i < hypotheses.Length; i++)
            {
                if (_robertaTokenizer != null)
                {
                    var (ids, mask) = _robertaTokenizer.EncodePair(premise, hypotheses[i], maxTotal);
                    idsList[i] = ids;
                    maskList[i] = mask;
                }
                else
                {
                    var (ids, mask) = FallbackSimple(premise, hypotheses[i]);
                    idsList[i] = ids;
                    maskList[i] = mask;
                }
            }
            return (idsList, maskList);
        }

        // バッチ入力テンソル生成
        private List<NamedOnnxValue> CreateRobertaBatchInputTensors(long[][] inputIdsBatch, long[][] attentionMaskBatch)
        {
            int batch = inputIdsBatch.Length;
            int seq = inputIdsBatch[0].Length;
            var idsT = new DenseTensor<long>(new[] { batch, seq });
            var mskT = new DenseTensor<long>(new[] { batch, seq });
            for (int b = 0; b < batch; b++)
            {
                var ids = inputIdsBatch[b];
                var msk = attentionMaskBatch[b];
                for (int t = 0; t < seq; t++)
                {
                    idsT[b, t] = ids[t];
                    mskT[b, t] = msk[t];
                }
            }
            return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idsT),
                NamedOnnxValue.CreateFromTensor("attention_mask", mskT)
            };
        }

        // 単発推論
        private async Task<double> ExecuteRobertaInferenceAsync(
            InferenceSession session,
            string premise,
            string hypothesis,
            SpeakerRole role,
            double temperature,
            string scoreMode)
        {
            try
            {
                var (ids, mask) = await TokenizeForRobertaAsync(premise, hypothesis);
                var inputs = CreateRobertaInputTensors(ids, mask);

                double[] logits;
                // ★ 修正: DirectML 使用時はスレッドプールを使わない
                if (_useDirectML)
                {
                    using var results = session.Run(inputs);
                    logits = results.First().AsEnumerable<float>().Select(v => (double)v).ToArray();
                }
                else
                {
                    logits = await Task.Run(() =>
                    {
                        using var results = session.Run(inputs);
                        return results.First().AsEnumerable<float>().Select(v => (double)v).ToArray();
                    });
                }

                var probs = SoftmaxTemp(logits, temperature);
                double score = ComputeMnliScore(probs, scoreMode);
                return ApplyRoleSpecificAdjustment(score, premise, role);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ERR][SEQ] {ex.Message}");
                return 0.0;
            }
        }

        // MNLIスコア計算
        private double ComputeMnliScore(double[] probs, string mode)
        {
            if (probs.Length < 3) return 0;
            double c = probs[0], n = probs[1], e = probs[2];
            return mode switch
            {
                "ent_only" => e,
                "margin" => e - Math.Max(c, n),
                "ent_minus_neu" => e - n,
                "log_margin" => Math.Log(e + 1e-8) - Math.Log(Math.Max(c, n) + 1e-8),
                _ => e - n
            };
        }

        // 個別トークナイズ
        private async Task<(long[] ids, long[] mask)> TokenizeForRobertaAsync(string premise, string hypothesis)
        {
            return await Task.Run(() =>
            {
                try
                {
                    if (_robertaTokenizer == null) return FallbackSimple(premise, hypothesis);
                    return _robertaTokenizer.EncodePair(premise, hypothesis, Math.Min(AffectTuning.MaxTotal, 512));
                }
                catch
                {
                    return FallbackSimple(premise, hypothesis);
                }
            });
        }

        // フォールバック簡易
        private (long[] ids, long[] mask) FallbackSimple(string premise, string hypothesis)
        {
            string combined = $"{premise}</s></s>{hypothesis}";
            var toks = SimpleTokenize(combined);
            var ids = toks.Select(t => (long)GetTokenId(t)).ToList();
            if (ids.Count > AffectTuning.MaxTotal) ids = ids.Take(AffectTuning.MaxTotal).ToList();

            var mask = Enumerable.Repeat(1L, ids.Count).ToList();
            if (ids.Count < AffectTuning.MaxTotal)
            {
                int pad = AffectTuning.MaxTotal - ids.Count;
                ids.AddRange(Enumerable.Repeat(0L, pad));
                mask.AddRange(Enumerable.Repeat(0L, pad));
            }
            return (ids.ToArray(), mask.ToArray());
        }

        private List<NamedOnnxValue> CreateRobertaInputTensors(long[] inputIds, long[] attentionMask)
        {
            var idsT = new DenseTensor<long>(new[] { 1, inputIds.Length });
            var mskT = new DenseTensor<long>(new[] { 1, attentionMask.Length });
            for (int i = 0; i < inputIds.Length; i++)
            {
                idsT[0, i] = inputIds[i];
                mskT[0, i] = attentionMask[i];
            }
            return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idsT),
                NamedOnnxValue.CreateFromTensor("attention_mask", mskT)
            };
        }

        // ロール調整
        private double ApplyRoleSpecificAdjustment(double baseScore, string premise, SpeakerRole role) =>
            role switch
            {
                SpeakerRole.AI => ContainsPoliteExpressions(premise) || ContainsAdviceExpressions(premise) ? baseScore * 0.7 : baseScore,
                SpeakerRole.System => baseScore * 0.3,
                _ => baseScore
            };

        private bool ContainsPoliteExpressions(string text)
        {
            string[] patterns = { "please", "could you", "would you", "thank you", "I appreciate", "kindly", "respectfully" };
            return patterns.Any(p => text.Contains(p, StringComparison.OrdinalIgnoreCase));
        }
        private bool ContainsAdviceExpressions(string text)
        {
            string[] patterns = { "I recommend", "I suggest", "you should", "you might", "consider", "it would be better", "my advice" };
            return patterns.Any(p => text.Contains(p, StringComparison.OrdinalIgnoreCase));
        }

        private double ApplyLabelWeight(double score, string label, Dictionary<string, double> weights) =>
            weights.TryGetValue(label, out var w) ? score * w : score;

        private double ApplyRoleBasedWeight(double score, string label, SpeakerRole role)
        {
            var rw = AffectTuning.GetRoleBasedWeights(role);
            if (rw.TryGetValue(label, out var mult))
            {
                double after = score * mult;
                if (_debugMode) Debug.WriteLine($"[ROLE]{role} {label}: {score:F3} -> {after:F3}");
                return after;
            }
            return score;
        }

        /// <summary>
        /// Ja /Enの感情ごとの閾値を超えたスコアに対して、重みを乗じて強調します。
        /// </summary>
        /// <param name="score"></param>
        /// <param name="localizedLabel"></param>
        /// <returns></returns>
        private double ApplyEmotionThresholdEmphasis(double score, string localizedLabel)
        {
            string enabled = Config.AffectConfigManager.Get("nli_emotion", "EnableRobertaThresholds", "true");
            if (!enabled.Equals("true", StringComparison.OrdinalIgnoreCase)) return score;
            if (EmotionScoring.EmotionThresholds.TryGetValue(localizedLabel, out var cfg))
            {
                if (score > cfg.PositiveThreshold || score < cfg.NegativeThreshold)
                    score *= cfg.Weight;
            }
            return score;
        }

        // 集約
        private double GetAggregatedScore(List<double> scores)
        {
            if (scores.Count == 0) return 0;
            return _aggregate switch
            {
                "mean" => scores.Average(),
                "min" => scores.Min(),
                _ => scores.Max()
            };
        }

        // 簡易トークナイズ/ID
        private string[] SimpleTokenize(string text) =>
            text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        private int GetTokenId(string token) =>
            Math.Abs(token.GetHashCode()) % 50000;

        // === 追加: RoBERTa利用判定（既存partialから欠落したため再定義） ===
        public bool ShouldUseRobertaModel()
        {
            try
            {
                string modelType = Config.AffectConfigManager.Get("nli_model", "Model", "bert_sentiment");
                return modelType.Equals("roberta-large-mnli", StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }

        // === 追加: RoBERTaリソース解放（Dispose拡張で使用） ===
        private void DisposeRobertaResources()
        {
            if (_robertaSession != null)
            {
                _robertaSession.Dispose();
                _robertaSession = null;
                _robertaInitialized = false;
                DebugWriteLine("RoBERTaリソースを解放しました");
            }
        }

        // 追加ヘルパー: RoBERTa用に canonical 英語ラベルへ正規化した有効プロンプト集合を取得
        static Dictionary<string, string[]> GetActivePromptsForRoberta()
        {
            // 有効プロンプト（言語設定に応じた localized キー）
            var activeLocalized = AffectTuning.GetActivePromptsDict();

            // 英語 canonical ← localized 逆引き
            var langMap = AffectTuning.EmotionLabels.TryGetValue(AffectTuning.Language, out var locMap)
                ? locMap
                : AffectTuning.EmotionLabels["jp"];
            // locMap: canonical -> localized
            var inverse = locMap.ToDictionary(kv => kv.Value, kv => kv.Key, StringComparer.OrdinalIgnoreCase);

            // RoBERTaは英語モデル前提: canonical 英語キーへ統一
            var normalized = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in activeLocalized)
            {
                var localized = kv.Key;
                var canonical = inverse.TryGetValue(localized, out var c) ? c : localized;

                // 日本語行を混在させたくない場合は簡易フィルタ（英字主体行を優先）
                var filtered = kv.Value
                    .Where(s => s.Any(ch => ch <= 127 && char.IsLetter(ch))) // ASCII文字を含む行優先
                    .DefaultIfEmpty(kv.Value.First()) // 全て日本語なら先頭を残す
                    .ToArray();

                normalized[canonical] = filtered;
            }

            return normalized;
        }
    }
}