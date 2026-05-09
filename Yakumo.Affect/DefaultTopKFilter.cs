namespace Yakumo.Affect
{
    /// <summary>
    /// [EN] Default fallback filter that simply returns Top-K by descending score.
    /// Used when no external filter plugin (e.g., Polarity Gate DLL) is found.
    /// [JA] 外部フィルタプラグイン（極性ゲートDLL等）が見つからない場合に使用される
    /// デフォルトのフォールバックフィルタ。スコア降順でTop-Kを返す。
    /// </summary>
    public sealed class DefaultTopKFilter : IEmotionFilter
    {
        /// <inheritdoc />
        public List<KeyValuePair<string, double>> Filter(
            IDictionary<string, double> scores,
            int k,
            string language,
            Action<string>? debugLog = null)
        {
            debugLog?.Invoke("[FILTER] DefaultTopKFilter: 単純スコア降順Top-K を適用");

            return scores
                .OrderByDescending(kv => kv.Value)
                .Take(k)
                .ToList();
        }
    }
}