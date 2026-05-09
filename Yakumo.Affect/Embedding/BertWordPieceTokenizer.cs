using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Yakumo.Affect.Embedding
{
    /// <summary>
    /// BERT系モデル用 WordPiece トークナイザー
    /// all-MiniLM-L6-v2 などの BERT 系 Embedding モデルで使用
    /// </summary>
    public sealed class BertWordPieceTokenizer
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly int _clsId;
        private readonly int _sepId;
        private readonly int _padId;
        private readonly int _unkId;
        private readonly int _maxInputCharsPerWord;
        private readonly string _continuingSubwordPrefix;
        private readonly bool _lowercase;

        private BertWordPieceTokenizer(
            Dictionary<string, int> vocab,
            int clsId, int sepId, int padId, int unkId,
            int maxInputCharsPerWord,
            string continuingSubwordPrefix,
            bool lowercase)
        {
            _vocab = vocab;
            _clsId = clsId;
            _sepId = sepId;
            _padId = padId;
            _unkId = unkId;
            _maxInputCharsPerWord = maxInputCharsPerWord;
            _continuingSubwordPrefix = continuingSubwordPrefix;
            _lowercase = lowercase;
        }

        /// <summary>
        /// tokenizer.json から読み込み
        /// </summary>
        public static BertWordPieceTokenizer Load(string tokenizerJsonPath)
        {
            if (!File.Exists(tokenizerJsonPath))
                throw new FileNotFoundException("tokenizer.json が見つかりません", tokenizerJsonPath);

            var json = File.ReadAllText(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // vocab の読み込み
            if (!root.TryGetProperty("model", out var modelElem))
                throw new InvalidDataException("tokenizer.json: 'model' が存在しません");

            if (!modelElem.TryGetProperty("vocab", out var vocabElem))
                throw new InvalidDataException("tokenizer.json: model.vocab が存在しません");

            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var kv in vocabElem.EnumerateObject())
            {
                vocab[kv.Name] = kv.Value.GetInt32();
            }

            // 特殊トークン ID
            int clsId = vocab.GetValueOrDefault("[CLS]", 101);
            int sepId = vocab.GetValueOrDefault("[SEP]", 102);
            int padId = vocab.GetValueOrDefault("[PAD]", 0);
            int unkId = vocab.GetValueOrDefault("[UNK]", 100);

            // モデル設定
            int maxInputCharsPerWord = 100;
            string continuingSubwordPrefix = "##";

            if (modelElem.TryGetProperty("max_input_chars_per_word", out var maxCharsElem))
                maxInputCharsPerWord = maxCharsElem.GetInt32();

            if (modelElem.TryGetProperty("continuing_subword_prefix", out var prefixElem))
                continuingSubwordPrefix = prefixElem.GetString() ?? "##";

            // lowercase 設定
            bool lowercase = true;
            if (root.TryGetProperty("normalizer", out var normElem) &&
                normElem.TryGetProperty("lowercase", out var lcElem))
            {
                lowercase = lcElem.GetBoolean();
            }

            Console.WriteLine($"[Tokenizer] BERT WordPiece: vocab={vocab.Count}, lowercase={lowercase}");

            return new BertWordPieceTokenizer(
                vocab, clsId, sepId, padId, unkId,
                maxInputCharsPerWord, continuingSubwordPrefix, lowercase);
        }

        /// <summary>
        /// テキストをトークン ID 配列に変換
        /// 形式: [CLS] tokens [SEP]
        /// </summary>
        public long[] Encode(string text, int maxLength = 512)
        {
            if (_lowercase)
                text = text.ToLowerInvariant();

            var tokens = Tokenize(text);
            var ids = new List<int>(Math.Min(tokens.Count + 2, maxLength)) { _clsId };

            int takeCount = Math.Min(tokens.Count, maxLength - 2);
            for (int i = 0; i < takeCount; i++)
            {
                ids.Add(_vocab.TryGetValue(tokens[i], out var id) ? id : _unkId);
            }
            ids.Add(_sepId);

            return ids.Select(i => (long)i).ToArray();
        }

        /// <summary>
        /// WordPiece トークン化
        /// </summary>
        private List<string> Tokenize(string text)
        {
            var result = new List<string>();

            // 基本的な単語分割（空白 + 句読点）
            var words = BasicTokenize(text);

            foreach (var word in words)
            {
                if (word.Length > _maxInputCharsPerWord)
                {
                    result.Add("[UNK]");
                    continue;
                }

                // WordPiece アルゴリズム
                var subTokens = WordPieceTokenize(word);
                result.AddRange(subTokens);
            }

            return result;
        }

        /// <summary>
        /// 基本トークン化（空白・句読点で分割）
        /// </summary>
        private static List<string> BasicTokenize(string text)
        {
            var tokens = new List<string>();
            var sb = new StringBuilder();

            foreach (char c in text)
            {
                if (char.IsWhiteSpace(c))
                {
                    if (sb.Length > 0)
                    {
                        tokens.Add(sb.ToString());
                        sb.Clear();
                    }
                }
                else if (IsPunctuation(c))
                {
                    if (sb.Length > 0)
                    {
                        tokens.Add(sb.ToString());
                        sb.Clear();
                    }
                    tokens.Add(c.ToString());
                }
                else
                {
                    sb.Append(c);
                }
            }

            if (sb.Length > 0)
                tokens.Add(sb.ToString());

            return tokens;
        }

        /// <summary>
        /// WordPiece サブワード分割
        /// </summary>
        private List<string> WordPieceTokenize(string word)
        {
            var result = new List<string>();
            int start = 0;

            while (start < word.Length)
            {
                int end = word.Length;
                string? foundToken = null;

                while (start < end)
                {
                    string substr = word.Substring(start, end - start);
                    if (start > 0)
                        substr = _continuingSubwordPrefix + substr;

                    if (_vocab.ContainsKey(substr))
                    {
                        foundToken = substr;
                        break;
                    }
                    end--;
                }

                if (foundToken == null)
                {
                    result.Add("[UNK]");
                    break;
                }

                result.Add(foundToken);
                start = end;
            }

            return result;
        }

        private static bool IsPunctuation(char c)
        {
            // ASCII 句読点 + Unicode 句読点カテゴリ
            if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
                (c >= 91 && c <= 96) || (c >= 123 && c <= 126))
                return true;

            return char.IsPunctuation(c);
        }
    }
}