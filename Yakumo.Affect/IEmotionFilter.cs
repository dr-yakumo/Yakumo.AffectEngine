namespace Yakumo.Affect
{
    /// <summary>
    /// [EN] Plugin interface for emotion post-filtering (e.g., polarity gate).
    /// Implementations receive raw scores and return a filtered Top-K list.
    /// [JA] 感情ポストフィルタ（極性ゲート等）のプラグインインターフェース。
    /// 生スコアを受け取り、フィルタ済みTop-Kリストを返す。
    /// </summary>
    public interface IEmotionFilter
    {
        /// <summary>
        /// [EN] Filters emotion scores and returns up to <paramref name="k"/> results.
        /// [JA] 感情スコアをフィルタし、最大 <paramref name="k"/> 件を返す。
        /// </summary>
        /// <param name="scores">感情ラベル→スコアの辞書</param>
        /// <param name="k">返却する最大件数（Top-K）</param>
        /// <param name="language">言語コード（"ja" / "en"）</param>
        /// <param name="debugLog">デバッグログ出力コールバック（任意）</param>
        /// <returns>フィルタ適用後のスコア降順リスト（最大K件）</returns>
        List<KeyValuePair<string, double>> Filter(
            IDictionary<string, double> scores,
            int k,
            string language,
            Action<string>? debugLog = null);
    }
}