using System;
using System.Collections.Generic;
using System.Linq;

namespace Yakumo.Affect
{
    /// <summary>
    /// BERT Sentiment（5値）出力をYakumoの感情スキーマへ写像し、しきい値・文脈・ロールでスコア調整するクラス
    /// - EmotionThresholds: 感情ごとのしきい値と重み
    /// - WordAdjustments: 単語/句のヒューリスティック調整
    /// - CalculateSentimentScore: スコア計算のエントリポイント
    /// </summary>
    [CodeStatus(CodeStatus.Legacy,
        Note = "Used only by the BERT-sentiment path (ClassifyByNli/ClassifyByNliAsync). " +
               "Not used in GoEmotions or RoBERTa-MNLI paths. Retained as fallback/research. / " +
               "BERT-sentimentパス（ClassifyByNli/ClassifyByNliAsync）専用。" +
               "GoEmotions・RoBERTa-MNLIパスでは不使用。研究・フォールバック用として保持。")]
    public static class EmotionScoring
    {
        /// <summary>
        /// 感情判定のしきい値・重み（JP/ENを網羅）
        /// PositiveThreshold を超える/NegativeThreshold を下回る場合に Weight を乗算します。
        /// </summary>
        public static readonly Dictionary<string, ThresholdConfig> EmotionThresholds = new()
        {
            // 既存7種（JP）
            ["驚き"] = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["怒り"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.3, Weight = 0.9 },
            ["喜び"] = new ThresholdConfig { PositiveThreshold = 0.5, NegativeThreshold = -0.1, Weight = 1.0 },
            ["悲しみ"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 1.0 },
            ["恐れ"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 0.9 },
            ["嫌悪"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.3, Weight = 0.8 },
            ["中立"] = new ThresholdConfig { PositiveThreshold = 0.2, NegativeThreshold = -0.2, Weight = 0.8 },

            // 既存7種（EN）
            ["surprise"] = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["anger"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.3, Weight = 0.9 },
            ["joy"] = new ThresholdConfig { PositiveThreshold = 0.5, NegativeThreshold = -0.1, Weight = 1.0 },
            ["sadness"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 1.0 },
            ["fear"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 0.9 },
            ["disgust"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.3, Weight = 0.8 },
            ["neutral"] = new ThresholdConfig { PositiveThreshold = 0.2, NegativeThreshold = -0.2, Weight = 0.8 },

            // RAG拡張（JP）
            ["親しみ・愛情"] = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["信頼"]       = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["恨み"]       = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 1.0 },
            ["呆れ・失望"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 0.9 },
            ["恥・罪悪感"] = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 1.0 },
            ["期待"]       = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["欲望"]       = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },

            // RAG拡張（EN）
            ["affection"]    = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["trust"]        = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["resentment"]   = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 1.0 },
            ["disbelief"]    = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 0.9 },
            ["shame"]        = new ThresholdConfig { PositiveThreshold = 0.3, NegativeThreshold = -0.2, Weight = 1.0 },
            ["anticipation"] = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
            ["desire"]       = new ThresholdConfig { PositiveThreshold = 0.4, NegativeThreshold = -0.1, Weight = 1.0 },
        };

        /// <summary>
        /// 単語/句のヒューリスティック調整（感情別の加算・減算）
        /// 例: "wow" が含まれる場合は「驚き」を上げる、など。
        /// </summary>
        private static readonly Dictionary<string, Dictionary<string, double>> WordAdjustments = new()
        {
            // 眠気 → 驚きを強く下げ、中立を上げる
            ["I'm sleepy"] = new Dictionary<string, double>
            {
                ["驚き"] = -1.0,
                ["中立"] = 0.5
            },

            // 助言/提案語 → 中立寄せ（AI応答っぽさ）
            ["recommend"] = new Dictionary<string, double> { ["中立"] = 0.3 },
            ["suggest"]   = new Dictionary<string, double> { ["中立"] = 0.3 },
            ["advice"]    = new Dictionary<string, double> { ["中立"] = 0.3 },

            // 強い感嘆表現 → 驚きを上げる
            ["wow"]     = new Dictionary<string, double> { ["驚き"] = 0.5 },
            ["amazing"] = new Dictionary<string, double> { ["驚き"] = 0.6 }
        };

        /// <summary>
        /// NliTuning側の重み（互換用途）。必要に応じて参照。
        /// </summary>
        public static readonly Dictionary<string, double> LabelWeights = new()
        {
            ["驚き"] = 1.00,
            ["中立"] = 1.00
        };

        /// <summary>
        /// スコア計算（ロール対応版）
        /// 1) BERT出力の5値（VN/N/Neu/P/VP）から基礎スコアを算出
        /// 2) 単語ベースのヒューリスティック加算/減算
        /// 3) ロール（User/AI/System）で文脈調整
        /// 4) 感情別しきい値で重み（Weight）を乗算
        /// </summary>
        public static double CalculateSentimentScore(
            double[] probabilities,
            string emotion,
            string text,
            SpeakerRole role = SpeakerRole.User)
        {
            // 1) モデル出力から基礎スコアを計算
            double baseScore = CalculateBaseScore(probabilities);

            // 2) 単語ベースの調整
            foreach (var entry in WordAdjustments)
            {
                bool containsCue = text.Contains(entry.Key, StringComparison.OrdinalIgnoreCase);
                if (containsCue)
                {
                    bool hasAdjustment = entry.Value.TryGetValue(emotion, out double adjustment);
                    if (hasAdjustment)
                    {
                        baseScore += adjustment;
                    }
                }
            }

            // 3) ロール別に文脈調整
            AdjustScoreByRole(ref baseScore, emotion, text, role);

            // 4) 感情ごとのしきい値に基づく重みの適用
            bool hasThreshold = EmotionThresholds.TryGetValue(emotion, out var config);
            if (hasThreshold)
            {
                // 押し上げ/押し下げのどちらも「重み乗算」で出力の確度を強調
                if (baseScore > config.PositiveThreshold)
                {
                    return baseScore * config.Weight;
                }

                if (baseScore < config.NegativeThreshold)
                {
                    return baseScore * config.Weight;
                }
            }

            // しきい値に触れない場合はそのまま返す
            return baseScore;
        }

        /// <summary>
        /// スコア計算（互換オーバーロード: role = User）
        /// </summary>
        public static double CalculateSentimentScore(
            double[] probabilities,
            string emotion,
            string text)
        {
            return CalculateSentimentScore(probabilities, emotion, text, SpeakerRole.User);
        }

        /// <summary>
        /// ロール別の文脈調整ディスパッチ
        /// </summary>
        private static void AdjustScoreByRole(
            ref double score,
            string emotion,
            string text,
            SpeakerRole role)
        {
            switch (role)
            {
                case SpeakerRole.AI:
                {
                    AdjustScoreForAI(ref score, emotion, text);
                    break;
                }
                case SpeakerRole.System:
                {
                    AdjustScoreForSystem(ref score, emotion, text);
                    break;
                }
                case SpeakerRole.User:
                default:
                {
                    AdjustScoreByContext(ref score, emotion, text);
                    break;
                }
            }
        }

        /// <summary>
        /// AI応答っぽい文体（丁寧語/助言/説明）では「驚き」を強く抑制し「中立」を上げる
        /// </summary>
        private static void AdjustScoreForAI(
            ref double score,
            string emotion,
            string text)
        {
            // 丁寧語・敬語パターン
            bool isPolitePattern =
                //text.Contains("です") ||
                //text.Contains("ます") ||
                //text.Contains("でしょう") ||
                text.Contains("かもしれません") ||
                text.Contains("いかがでしょうか");

            // 助言・提案パターン
            bool isAdvicePattern =
                text.Contains("おすすめ") ||
                text.Contains("提案") ||
                text.Contains("suggest") ||
                text.Contains("recommend") ||
                text.Contains("いかがでしょう") ||
                text.Contains("と良い");

            // 説明パターン
            bool isExplanationPattern =
                text.Contains("なぜなら") ||
                text.Contains("そのため") ||
                text.Contains("つまり") ||
                text.Contains("具体的には") ||
                text.Contains("because") ||
                text.Contains("therefore");

            // AI特有の文型の場合は驚きを強く下げ、中立を強く上げる
            if (isPolitePattern || isAdvicePattern || isExplanationPattern)
            {
                if (emotion == "驚き" || emotion == "surprise")
                {
                    score -= 1.5;
                }

                if (emotion == "中立" || emotion == "neutral")
                {
                    score += 1.0;
                }
            }

            // 共通の文脈調整も適用
            AdjustScoreByContext(ref score, emotion, text);
        }

        /// <summary>
        /// システムメッセージは原則中立：
        /// - 驚きはほぼゼロへ
        /// - 中立は大きく押し上げ
        /// </summary>
        private static void AdjustScoreForSystem(
            ref double score,
            string emotion,
            string text)
        {
            if (emotion == "驚き" || emotion == "surprise")
            {
                score -= 2.0;
            }

            if (emotion == "中立" || emotion == "neutral")
            {
                score += 1.5;
            }
        }

        /// <summary>
        /// 一般文脈に基づく微調整：
        /// - 思考表現（I think/ believe）→ 驚きを下げ、中立を上げる
        /// - 感嘆符あり → 驚きを上げる
        /// </summary>
        private static void AdjustScoreByContext(
            ref double score,
            string emotion,
            string text)
        {
            bool isThinkingPattern =
                text.StartsWith("I think", StringComparison.OrdinalIgnoreCase) ||
                text.StartsWith("I believe", StringComparison.OrdinalIgnoreCase);

            bool isQuestionPattern =
                text.EndsWith("?") ||
                text.StartsWith("What", StringComparison.OrdinalIgnoreCase);
            // isQuestionPattern は現在は参照のみ（必要なら将来拡張）

            bool isExclamationPattern = text.Contains("!");

            if (isThinkingPattern)
            {
                if (emotion == "驚き" || emotion == "surprise")
                {
                    score -= 0.7;
                }

                if (emotion == "中立" || emotion == "neutral")
                {
                    score += 0.4;
                }
            }

            if (isExclamationPattern)
            {
                if (emotion == "驚き" || emotion == "surprise")
                {
                    score += 0.3;
                }
            }
        }

        /// <summary>
        /// BERT Sentimentの5カテゴリ出力から基礎スコアを算出
        /// p[0]=Very Negative, p[1]=Negative, p[2]=Neutral, p[3]=Positive, p[4]=Very Positive
        /// - 正/負を優勢側の符号で[-1,1]に正規化
        /// - 中立は弱く寄与（0.5係数）、最大カテゴリ位置で傾きを微調整
        /// </summary>
        private static double CalculateBaseScore(double[] p)
        {
            // 想定外の配列長は0扱い（安全策）
            if (p.Length != 5)
            {
                return 0.0;
            }

            // 正負/中立の寄与を合成（重みは経験的に調整）
            double positiveScore = (p[3] * 0.7) + (p[4] * 1.0);  // Positive=0.7, Very Positive=1.0
            double negativeScore = (p[0] * 1.0) + (p[1] * 0.7);  // Very Negative=1.0, Negative=0.7
            double neutralScore  = (p[2] * 0.5);                 // Neutral=0.5

            // 最大カテゴリ（最も支配的なクラス）
            int maxIndex = Array.IndexOf(p, p.Max());

            // 正規化（-1.0 〜 1.0）
            if (positiveScore > negativeScore && positiveScore > neutralScore)
            {
                return Math.Min(positiveScore, 1.0);
            }
            else if (negativeScore > positiveScore && negativeScore > neutralScore)
            {
                return -Math.Min(negativeScore, 1.0);
            }
            else
            {
                // 中立優勢：最大カテゴリが正/負側かで傾きを微調整
                return neutralScore * (maxIndex > 2 ? 0.3 : -0.3);
            }
        }
    }

    /// <summary>
    /// 感情しきい値設定
    /// </summary>
    public class ThresholdConfig
    {
        /// <summary>この値を超えたら「当該感情が強い」とみなす</summary>
        public double PositiveThreshold { get; set; }

        /// <summary>この値を下回ったら「当該感情が弱い/負」とみなす</summary>
        public double NegativeThreshold { get; set; }

        /// <summary>しきい値判定時に乗算する重み（強調係数）</summary>
        public double Weight { get; set; }
    }
}