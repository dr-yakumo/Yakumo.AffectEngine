using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Yakumo.Affect
{
    /// <summary>
    /// ユーザー辞書 JSON ファイルのスキーマモデル。
    /// <c>affect.dict.json</c> として配置することで自動ロードされます。
    /// </summary>
    internal sealed class UserDictionaryFile
    {
        /// <summary>スキーマバージョン（将来の互換性管理用）</summary>
        [JsonPropertyName("version")]
        public int Version { get; set; } = 1;

        /// <summary>
        /// フレーズ辞書（完全一致 → 翻訳をスキップして直接返す）。
        /// キー: 日本語原文、値: 英語訳
        /// </summary>
        [JsonPropertyName("phraseDict")]
        public Dictionary<string, string>? PhraseDict { get; set; }

        /// <summary>
        /// 固有名詞置換マップ（翻訳前に部分一致で置換）。
        /// キー: 日本語固有名詞、値: 英語表記
        /// </summary>
        [JsonPropertyName("properNouns")]
        public Dictionary<string, string>? ProperNouns { get; set; }

        /// <summary>
        /// 英語後処理補正（翻訳後に部分一致で置換）。
        /// キー: 誤訳パターン、値: 正しい英語表現
        /// </summary>
        [JsonPropertyName("corrections")]
        public Dictionary<string, string>? Corrections { get; set; }
    }
}