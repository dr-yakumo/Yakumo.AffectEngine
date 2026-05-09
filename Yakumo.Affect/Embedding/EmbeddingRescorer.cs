using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace Yakumo.Affect.Embedding
{
    /// <summary>
    /// NLI スコアと Embedding 類似文投票を組み合わせて再スコアリングする
    /// Phase 5-4 Step 6: Embedding ベース再スコアリング
    /// </summary>
    public sealed class EmbeddingRescorer
    {
        private bool _initialized = false;

        /// <summary>
        /// シングルトンインスタンス
        /// </summary>
        public static EmbeddingRescorer Instance { get; } = new();

        /// <summary>
        /// 再スコアリングが有効かどうか
        /// </summary>
        public bool IsEnabled =>
            Config.AffectConfigManager.Get("embedding", "Rescoring.Enabled", "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// 初期化（EmbeddingService + VectorIndex のロード）
        /// </summary>
        public async Task InitializeAsync()
        {
            if (_initialized) return;

            try
            {
                // EmbeddingService の初期化
                await EmbeddingService.Instance.InitializeAsync();

                // VectorIndex のロード
                VectorIndex.Instance.Load();

                _initialized = EmbeddingService.Instance.IsInitialized && VectorIndex.Instance.IsLoaded;

                if (_initialized)
                    Console.WriteLine($"[RESCORE] 初期化完了 (index={VectorIndex.Instance.Count} 件)");
                else
                    Console.WriteLine("[RESCORE][WARN] 初期化不完全 - 再スコアリング無効");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[RESCORE][ERROR] 初期化失敗: {ex.Message}");
                _initialized = false;
            }
        }

        /// <summary>
        /// NLI スコアに対して Embedding ベースの再スコアリングを適用
        /// </summary>
        public async Task<Dictionary<string, double>> RescoreAsync(
            string text,
            Dictionary<string, double> nliScores,
            string language = "jp",
            bool debugMode = false)
        {
            if (!IsEnabled || !_initialized)
                return nliScores;

            // トリガー条件チェック
            if (!ShouldTriggerRescore(nliScores, language))
                return nliScores;

            var sw = Stopwatch.StartNew();

            try
            {
                // 1. テキストを Embedding 化
                var queryVector = await EmbeddingService.Instance.EmbedAsync(text);

                // 2. 類似文検索
                int topK = Config.AffectConfigManager.GetInt("embedding", "Rescoring.TopK", 30);
                float minSim = (float)Config.AffectConfigManager.GetDouble("embedding", "Rescoring.MinSimilarity", 0.60);
                var neighbors = VectorIndex.Instance.Search(queryVector, topK, minSim);

                if (neighbors.Count == 0)
                {
                    if (debugMode) Console.WriteLine("[RESCORE] 類似文が見つかりません - スキップ");
                    return nliScores;
                }

                // 3. ラベル投票スコアを取得（IDF 重み付け + ダンピング）
                bool useIdf = Config.AffectConfigManager.Get("embedding", "Rescoring.IdfWeighting", "true")
                    .Equals("true", StringComparison.OrdinalIgnoreCase);
                int idfDampingThreshold = Config.AffectConfigManager.GetInt("embedding", "Rescoring.IdfDampingThreshold", 20);
                var voteScores = VectorIndex.Instance.VoteLabelScores(neighbors, useIdf, idfDampingThreshold);

                // 4. NLI スコアと投票スコアを融合
                var rescored = FuseScores(nliScores, voteScores, language);

                sw.Stop();

                if (debugMode)
                {
                    Console.WriteLine($"[RESCORE] 類似文 {neighbors.Count} 件 (top sim={neighbors[0].Similarity:F3})");
                    if (useIdf)
                    {
                        double idfStrength = (idfDampingThreshold > 0 && neighbors.Count < idfDampingThreshold)
                            ? (double)neighbors.Count / idfDampingThreshold : 1.0;
                        Console.WriteLine($"[RESCORE] IDF: 有効 (damping={idfStrength:P0}, threshold={idfDampingThreshold}, neighbors={neighbors.Count})");
                    }
                    Console.WriteLine($"[RESCORE] 投票: {string.Join(", ", voteScores.OrderByDescending(x => x.Value).Take(5).Select(x => $"{x.Key}={x.Value:F3}"))}");

                    // 変更があったラベルを表示
                    foreach (var kv in rescored)
                    {
                        if (nliScores.TryGetValue(kv.Key, out var orig) && Math.Abs(orig - kv.Value) > 0.001)
                        {
                            Console.WriteLine($"[RESCORE]   {kv.Key}: {orig:F3} → {kv.Value:F3} (Δ={kv.Value - orig:+0.000;-0.000})");
                        }
                    }

                    Console.WriteLine($"[RESCORE] 処理時間: {sw.ElapsedMilliseconds}ms");
                }

                return rescored;
            }
            catch (Exception ex)
            {
                if (debugMode) Console.WriteLine($"[RESCORE][ERROR] {ex.Message}");
                return nliScores;
            }
        }

        /// <summary>
        /// 再スコアリングのトリガー条件をチェック
        /// </summary>
        private bool ShouldTriggerRescore(Dictionary<string, double> nliScores, string language)
        {
            bool triggerOnNeutral = Config.AffectConfigManager.Get("embedding", "Rescoring.TriggerOnNeutral", "true")
                .Equals("true", StringComparison.OrdinalIgnoreCase);
            double neutralMargin = Config.AffectConfigManager.GetDouble("embedding", "Rescoring.NeutralMargin", 0.08);

            if (!triggerOnNeutral)
                return true; // 常にトリガー

            // Top1 が neutral の場合にトリガー
            string neutralLabel = language == "en" ? "neutral" : "中立";
            var sorted = nliScores.OrderByDescending(x => x.Value).ToList();

            if (sorted.Count == 0) return false;

            string topLabel = sorted[0].Key;
            double topScore = sorted[0].Value;
            double secondScore = sorted.Count > 1 ? sorted[1].Value : 0;

            // Top1 が neutral かつ マージンが小さい
            if (topLabel.Equals(neutralLabel, StringComparison.OrdinalIgnoreCase) &&
                (topScore - secondScore) < neutralMargin)
            {
                return true;
            }

            // Top1 と Top2 の差が小さい（接戦）
            if ((topScore - secondScore) < neutralMargin)
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// NLI スコアと投票スコアを融合
        /// </summary>
        private static Dictionary<string, double> FuseScores(
            Dictionary<string, double> nliScores,
            Dictionary<string, double> voteScores,
            string language)
        {
            // ラベルマッピング（canonical英語 → ローカライズ済み）
            var enMap = AffectTuning.EmotionLabels.TryGetValue("en", out var mapEn) ? mapEn : new Dictionary<string, string>();
            var locMap = AffectTuning.EmotionLabels.TryGetValue(language, out var mapLoc) ? mapLoc : AffectTuning.EmotionLabels["jp"];

            // 融合比率（Config から読込、デフォルト: NLI 70% + Embedding 30%）
            double nliWeight = Config.AffectConfigManager.GetDouble("embedding", "Rescoring.NliWeight", 0.70);
            double embWeight = 1.0 - nliWeight;

            // ★ NLI raw スコアを softmax で正規化 (0〜1, 合計=1)
            var nliNormalized = SoftmaxNormalize(nliScores);

            var result = new Dictionary<string, double>(nliScores.Count, StringComparer.OrdinalIgnoreCase);

            foreach (var kv in nliScores)
            {
                string localizedLabel = kv.Key;

                // 正規化済み NLI スコアを取得
                double nliNorm = nliNormalized.TryGetValue(localizedLabel, out var n) ? n : 0;

                // ローカライズ済みラベルに対応する canonical 英語ラベルを探す
                string? canonicalEn = null;
                foreach (var en in enMap)
                {
                    var loc = locMap.TryGetValue(en.Key, out var l) ? l : en.Key;
                    if (loc.Equals(localizedLabel, StringComparison.OrdinalIgnoreCase) ||
                        en.Key.Equals(localizedLabel, StringComparison.OrdinalIgnoreCase))
                    {
                        canonicalEn = en.Key;
                        break;
                    }
                }

                // 英語ラベルで投票スコアを検索
                double voteScore = 0;
                if (canonicalEn != null && voteScores.TryGetValue(canonicalEn, out var vs))
                {
                    voteScore = vs;
                }
                else if (voteScores.TryGetValue(localizedLabel, out var vs2))
                {
                    voteScore = vs2;
                }

                // ★ 正規化済みスコア同士で融合（両方 0〜1 スケール）
                result[localizedLabel] = nliNorm * nliWeight + voteScore * embWeight;
            }

            return result;
        }

        /// <summary>
        /// スコア辞書を softmax で正規化 (全値を 0〜1 に変換、合計=1)
        /// </summary>
        private static Dictionary<string, double> SoftmaxNormalize(Dictionary<string, double> scores)
        {
            if (scores.Count == 0) return scores;

            // オーバーフロー防止: max を引く
            double maxVal = scores.Values.Max();
            double sumExp = 0;
            var exps = new Dictionary<string, double>(scores.Count, StringComparer.OrdinalIgnoreCase);

            foreach (var kv in scores)
            {
                double e = Math.Exp(kv.Value - maxVal);
                exps[kv.Key] = e;
                sumExp += e;
            }

            var result = new Dictionary<string, double>(scores.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var kv in exps)
            {
                result[kv.Key] = kv.Value / sumExp;
            }

            return result;
        }
    }
}