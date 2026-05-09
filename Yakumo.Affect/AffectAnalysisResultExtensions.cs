using System.Collections.Generic;

namespace Yakumo.Affect
{
    /// <summary>
    /// ClassificationResultからNLI_AnalysisResultへの変換を提供する拡張メソッド
    /// </summary>
    public static class AffectAnalysisResultExtensions
    {
        /// <summary>
        /// ClassificationResultをNLI_AnalysisResultに変換します
        /// </summary>
        /// <param name="result">変換元のClassificationResult</param>
        /// <returns>変換されたNLI_AnalysisResult</returns>
        public static AffectAnalysisResult ToAnalysisResult(this ClassificationResult result)
        {
            if (result == null)
            {
                return null;
            }

            var analysisResult = new AffectAnalysisResult
            {
                Text = result.Text,
                OriginalText = result.OriginalText,
                Scores = new Dictionary<string, double>(result.Scores),
                TopK = new List<KeyValuePair<string, double>>(result.TopK),
                SurpriseScore = result.SurpriseScore,
                IsSurprised = result.IsSurprised,
                Threshold = result.Threshold,
                Language = result.Language
            };
            
            return analysisResult;
        }

        /// <summary>
        /// NliAnalysisResultsをNLI_AnalysisResultのリストに変換します
        /// </summary>
        /// <param name="results">変換元のNliAnalysisResults</param>
        /// <returns>変換されたNLI_AnalysisResultのリスト</returns>
        public static List<AffectAnalysisResult> ToAnalysisResults(this NliAnalysisResults results)
        {
            if (results == null)
                return null;
                
            var analysisResults = new List<AffectAnalysisResult>();
            foreach (var result in results.Results)
            {
                analysisResults.Add(result.ToAnalysisResult());
            }
            
            return analysisResults;
        }
    }
}