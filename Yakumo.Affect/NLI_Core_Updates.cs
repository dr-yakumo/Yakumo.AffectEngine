using System.Diagnostics;
using System.Threading.Tasks;

namespace Yakumo.Affect
{
    /// <summary>
    /// NLI_Coreクラスのモデル自動選択メソッド
    /// </summary>
    public partial class AffectCore
    {
        /// <summary>
        /// 設定に基づいて適切なモデル（BERT/RoBERTa/GoEmotions）で分析を行います
        /// </summary>
        /// <param name="text">入力対象のテキスト</param>
        /// <param name="role">話者の役割</param>
        /// <param name="k">返却する上位感情の数</param>
        /// <param name="threshold">感情検出閾値</param>
        /// <param name="properNouns">固有名詞マスキング用ワード群</param>
        /// <returns>分析結果</returns>
        public async Task<ClassificationResult> AnalyzeTextWithAutoModelAsync(
            string text, 
            SpeakerRole role = SpeakerRole.User, 
            int k = 3, 
            double threshold = 0.25,
            string[]? properNouns = null)
        {
            if (_disposed) 
                throw new ObjectDisposedException(nameof(AffectCore));

            // Classified イベントを1箇所で発火するため、各経路は変数へ受けて最後に返す。
            ClassificationResult result;

            // ★ Phase B: GoEmotions 直接分類パス
            if (ShouldUseGoEmotionsModel())
            {
                DebugWriteLine("=== モデル選択 ===");
                DebugWriteLine("選択されたモデル: GoEmotions (roberta-base-go_emotions)");
                DebugWriteLine(IsGoEmoRaw28Mode()
                    ? "実行エンジン: GoEmo_Core (直接分類・raw28出力)"
                    : "実行エンジン: GoEmo_Core (直接分類・28→14マッピング)");
                DebugWriteLine("================");
                result = await ClassifyByGoEmotionsAsync(text, role, k, threshold, properNouns);
            }
            else if (ShouldUseRobertaModel())
            {
                DebugWriteLine("=== モデル選択 ===");
                DebugWriteLine("選択されたモデル: RoBERTa-large-MNLI");
                DebugWriteLine("実行エンジン: NLP_Core (高精度NLI用)");
                DebugWriteLine("================");
                result = await ClassifyByNlpAsync(text, role, k, threshold, properNouns);
            }
            else
            {
                var modelConfig = Config.AffectModelConfiguration.GetModelConfig();
                DebugWriteLine("=== モデル選択 ===");
                DebugWriteLine($"選択されたモデル: {modelConfig.Name}");
                DebugWriteLine("実行エンジン: NLI_Core (BERT系)");
                DebugWriteLine($"モデルタイプ: {modelConfig.ModelType}");
                DebugWriteLine("================");
                result = await ClassifyByNliAsync(text, _tokenizer!, _session!, role, k, threshold);
            }

            RaiseClassified(result);
            return result;
        }

        // AnalyzeTextsAsync メソッドのRoBERTa/GoEmotions対応版
        public async Task<NliAnalysisResults> AnalyzeTextsWithAutoModelAsync(
            string[] texts, 
            SpeakerRole role = SpeakerRole.User, 
            int k = 3, 
            double threshold = 0.25)
        {
            if (_disposed) 
                throw new ObjectDisposedException(nameof(AffectCore));

            var stopwatch = Stopwatch.StartNew();
            var results = new NliAnalysisResults();

            // ★ Phase B: GoEmotions
            if (ShouldUseGoEmotionsModel())
            {
                DebugWriteLine("=== 複数テキスト分析：GoEmotionsモデル使用 ===");

                if (!_goemoInitialized)
                    await InitializeGoEmotionsModelAsync();

                foreach (var text in texts)
                {
                    var result = await ClassifyByGoEmotionsAsync(text, role, k, threshold);
                    results.Results.Add(result);
                }
                results.ModelUsed = "goemo-roberta-base";
            }
            else if (ShouldUseRobertaModel())
            {
                DebugWriteLine("=== 複数テキスト分析：RoBERTaモデル使用 ===");
                DebugWriteLine("選択されたモデル: RoBERTa-large-MNLI");
                DebugWriteLine("実行エンジン: NLP_Core (高精度NLI用)");

                if (!_robertaInitialized)
                    await InitializeRobertaModelAsync();

                var tasks = texts.Select(text => ClassifyByNlpAsync(text, role, k, threshold)).ToArray();
                var analysisResults = await Task.WhenAll(tasks);
                results.Results.AddRange(analysisResults);
                results.ModelUsed = "roberta-large-mnli";
            }
            else
            {
                var modelConfig = Config.AffectModelConfiguration.GetModelConfig();
                DebugWriteLine("=== 複数テキスト分析：BERTモデル使用 ===");
                DebugWriteLine($"選択されたモデル: {modelConfig.Name}");

                var tasks = texts.Select(text => AnalyzeTextAsync(text, role, k, threshold)).ToArray();
                var analysisResults = await Task.WhenAll(tasks);
                results.Results.AddRange(analysisResults);
                results.ModelUsed = modelConfig.Name;
            }

            // 全分岐が results.Results に集約されるので、発火はここ1箇所でよい。
            // ※ BERT系の経路は AnalyzeTextAsync 経由だが、そちらはルーターを通らないため
            //    ここで発火しても二重にはならない。
            foreach (var result in results.Results)
                RaiseClassified(result);

            stopwatch.Stop();
            results.ProcessingTime = stopwatch.Elapsed;
            return results;
        }
    }
}