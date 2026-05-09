// ================================================================
// Yakumo Affect Engine — Backward Compatibility Shim
// ================================================================
// 旧名称 (NLI_Core 等) からの移行を支援するエイリアス定義。
// Phase 2-5b (将来) にて削除予定。
// ================================================================

namespace Yakumo.Affect
{
    /// <summary>
    /// [JA] 後方互換エイリアス: <see cref="AffectCore"/> の旧名称。
    /// 新規コードでは <see cref="AffectCore"/> を使用してください。
    /// </summary>
    [System.Obsolete(
        "NLI_Core is deprecated. Use AffectCore instead. " +
        "This alias will be removed in a future version.",
        error: false)]
    public class NLI_Core : AffectCore
    {
        /// <inheritdoc cref="AffectCore(string, bool, double, string, string)"/>
        public NLI_Core(
            string language = "jp",
            bool debugMode = false,
            double temperature = 0.3,
            string scoreMode = "ent_minus_neu",
            string aggregate = "max")
            : base(language, debugMode, temperature, scoreMode, aggregate) { }
    }
}