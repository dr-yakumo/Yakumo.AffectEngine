using System.Collections.Generic;

namespace Yakumo.Affect
{
    /// <summary>
    /// NLI（自然言語推論）による感情分析の結果を格納するクラス
    /// </summary>
    public class AffectAnalysisResult
    {
        /// <summary>
        /// 分析対象テキスト
        /// </summary>
        public string Text { get; set; }

        /// <summary>
        /// オリジナルの入力テキスト（翻訳前）
        /// </summary>
        public string OriginalText { get; set; }

        /// <summary>
        /// 各感情のスコア（キー=感情名, 値=スコア）
        /// </summary>
        public Dictionary<string, double> Scores { get; set; }

        /// <summary>
        /// スコアの高い順にソートされた感情とスコアのペアのリスト
        /// </summary>
        public List<KeyValuePair<string, double>> TopK { get; set; }

        /// <summary>
        /// 驚きのスコア値
        /// </summary>
        public double SurpriseScore { get; set; }

        /// <summary>
        /// 驚きが一定のしきい値を超えたかどうか
        /// </summary>
        public bool IsSurprised { get; set; }

        /// <summary>
        /// 驚きの判定に使用したしきい値
        /// </summary>
        public double Threshold { get; set; }

        /// <summary>
        /// 最も高いスコアの感情名
        /// </summary>
        public string TopEmotion
        {
            get
            {
                if (TopK != null && TopK.Count > 0)
                {
                    return TopK[0].Key;
                }
                return "中立"; // デフォルト値
            }
        }

        /// <summary>
        /// 最も高いスコア値
        /// </summary>
        public double TopScore
        {
            get
            {
                if (TopK != null && TopK.Count > 0)
                {
                    return TopK[0].Value;
                }
                return 0.0;
            }
        }

        /// <summary>
        /// 指定した感情のスコアを取得
        /// </summary>
        public double GetEmotionScore(string emotion)
        {
            if (Scores != null && Scores.ContainsKey(emotion))
            {
                return Scores[emotion];
            }
            return 0.0;
        }

        public string Language { get; set; }
    }
}