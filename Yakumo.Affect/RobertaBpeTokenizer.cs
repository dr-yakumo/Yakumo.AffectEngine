using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Yakumo.Affect
{
    public sealed class RobertaBpeTokenizer
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<(string, string), int> _mergeRanks;
        private readonly Dictionary<string, string> _cache = new();
        private readonly Dictionary<byte, string> _byte2unicode;
        private readonly Dictionary<string, byte> _unicode2byte;

        private readonly int _clsId;
        private readonly int _sepId;
        private readonly int _padId;
        private readonly int _unkId;
        private readonly bool _addPrefixSpace;

        private const int CACHE_MAX = 100_000;
        private readonly object _cacheLock = new();

        private RobertaBpeTokenizer(
            Dictionary<string, int> vocab,
            List<(string, string)> merges,
            int clsId,
            int sepId,
            int padId,
            int unkId,
            bool addPrefixSpace)
        {
            _vocab = vocab;
            _mergeRanks = new Dictionary<(string, string), int>(merges.Count);
            for (int i = 0; i < merges.Count; i++)
                _mergeRanks[merges[i]] = i;

            (_byte2unicode, _unicode2byte) = BuildByteLevelMaps();

            _clsId = clsId;
            _sepId = sepId;
            _padId = padId;
            _unkId = unkId;
            _addPrefixSpace = addPrefixSpace;
        }

        public static RobertaBpeTokenizer Load(string tokenizerJsonPath)
        {
            // merges.txt パス推定
            string? dir = Path.GetDirectoryName(tokenizerJsonPath);
            string mergesTxt = Path.Combine(dir ?? ".", "merges.txt");

            if (!File.Exists(tokenizerJsonPath))
                throw new FileNotFoundException("tokenizer.json が見つかりません", tokenizerJsonPath);

            // 1) tokenizer.json 読込
            var json = File.ReadAllText(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (!root.TryGetProperty("model", out var modelElem))
                throw new InvalidDataException("tokenizer.json: 'model' が存在しません");

            if (!modelElem.TryGetProperty("vocab", out var vocabElem) || vocabElem.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("tokenizer.json: model.vocab が不正");

            var vocab = new Dictionary<string, int>(vocabElem.GetRawText().Count(c => c == ':') + 8);
            foreach (var kv in vocabElem.EnumerateObject())
                vocab[kv.Name] = kv.Value.GetInt32();

            // add_prefix_space 検出（pre_tokenizer 深掘り → roberta フォールバック → model）
            bool addPrefixSpace = false;
            bool detectedFromPreTokenizer = false;
            bool hasByteLevel = false;

            if (root.TryGetProperty("pre_tokenizer", out var preTok))
            {
                hasByteLevel = ContainsByteLevel(preTok);
                if (TryExtractAddPrefixSpaceFromPreTokenizer(preTok, out var aps))
                {
                    addPrefixSpace = aps;
                    detectedFromPreTokenizer = true;
                }
            }

            if (!detectedFromPreTokenizer)
            {
                // ByteLevel があるが add_prefix_space が書かれていない/読めない場合の最終フォールバック
                if (hasByteLevel && tokenizerJsonPath.IndexOf("roberta", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    addPrefixSpace = true;
                }
                else
                {
                    addPrefixSpace = TryGetBool(modelElem, "add_prefix_space") ?? false;
                }
            }

            var merges = new List<(string, string)>();
            bool foundInJson = false;
            if (modelElem.TryGetProperty("merges", out var mergesElem) && mergesElem.ValueKind == JsonValueKind.Array)
            {
                foundInJson = true;
                foreach (var m in mergesElem.EnumerateArray())
                {
                    if (m.ValueKind == JsonValueKind.String)
                    {
                        var line = m.GetString();
                        if (string.IsNullOrWhiteSpace(line) || line!.StartsWith("#"))
                            continue;
                        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length == 2)
                            merges.Add((parts[0], parts[1]));
                    }
                    else if (m.ValueKind == JsonValueKind.Array)
                    {
                        var arr = m.EnumerateArray().Take(2).Select(x => x.GetString() ?? "").ToArray();
                        if (arr.Length == 2 && arr[0] != "" && arr[1] != "")
                            merges.Add((arr[0], arr[1]));
                    }
                }
            }

            // 3) tokenizer.json に無い/少なすぎる場合 merges.txt を読む
            if ((!foundInJson || merges.Count < 100) && File.Exists(mergesTxt))
            {
                merges.Clear();
                foreach (var line in File.ReadLines(mergesTxt))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#"))
                        continue;
                    // 典型: "Ġthere to"
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 2)
                        merges.Add((parts[0], parts[1]));
                }
            }

            // 4) special token ID
            int clsId = ResolveTokenId(root, vocab, "<s>", fallback: 0);
            int sepId = ResolveTokenId(root, vocab, "</s>", fallback: 2);
            int padId = ResolveTokenId(root, vocab, "<pad>", fallback: 1);
            int unkId = ResolveTokenId(root, vocab, "<unk>", fallback: 3);

            // 軽いバリデーション
            if (merges.Count == 0)
                Console.WriteLine("[Tokenizer] 警告: merges が 0 件（単文字分割モード）");
            else
                Console.WriteLine($"[Tokenizer] merges 読込: {merges.Count} 件 (例: {string.Join(", ", merges.Take(5).Select(m => m.Item1 + '+' + m.Item2))})");

            Console.WriteLine($"[Tokenizer] vocab={vocab.Count}, add_prefix_space={addPrefixSpace}, cls={clsId}, sep={sepId}, pad={padId}, unk={unkId}");

            return new RobertaBpeTokenizer(vocab, merges, clsId, sepId, padId, unkId, addPrefixSpace);
        }

        private static int ResolveTokenId(JsonElement root, Dictionary<string, int> vocab, string token, int fallback)
        {
            int? added = TryFindAddedTokenId(root, token);
            if (added.HasValue) return added.Value;
            if (vocab.TryGetValue(token, out var id)) return id;
            return fallback;
        }

        private static int? TryFindAddedTokenId(JsonElement root, string content)
        {
            if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
            {
                foreach (var t in added.EnumerateArray())
                {
                    if (t.ValueKind == JsonValueKind.Object &&
                        t.TryGetProperty("content", out var cEl) &&
                        cEl.ValueKind == JsonValueKind.String &&
                        cEl.GetString() == content &&
                        t.TryGetProperty("id", out var idEl) &&
                        idEl.ValueKind == JsonValueKind.Number)
                        return idEl.GetInt32();
                }
            }
            return null;
        }

        private static (Dictionary<byte, string> b2u, Dictionary<string, byte> u2b) BuildByteLevelMaps()
        {
            List<int> bs = new();
            for (int i = '!'; i <= '~'; i++) bs.Add(i);
            for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
            for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

            List<int> cs = new(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add(256 + n);
                    n++;
                }
            }

            var b2u = new Dictionary<byte, string>(256);
            var u2b = new Dictionary<string, byte>(256);
            for (int i = 0; i < bs.Count; i++)
            {
                byte b = (byte)bs[i];
                int c = cs[i];
                string ch = char.ConvertFromUtf32(c);
                b2u[b] = ch;
                u2b[ch] = b;
            }
            return (b2u, u2b);
        }

        private IEnumerable<string> ByteLevelPreTokenize(string text)
        {
            // 改行正規化
            text = text.Replace("\r\n", "\n");

            if (_addPrefixSpace && !string.IsNullOrEmpty(text) && !char.IsWhiteSpace(text[0]))
                text = " " + text;

            var sb = new StringBuilder(text.Length);
            sb.Append(text);

            var tokens = new List<string>();
            int idx = 0;
            while (idx < sb.Length)
            {
                int start = idx;
                while (idx < sb.Length && char.IsWhiteSpace(sb[idx])) idx++;
                int wsLen = idx - start;
                string ws = wsLen > 0 ? sb.ToString(start, wsLen) : "";

                start = idx;
                while (idx < sb.Length && !char.IsWhiteSpace(sb[idx])) idx++;
                int wordLen = idx - start;
                string word = wordLen > 0 ? sb.ToString(start, wordLen) : "";

                if (wordLen > 0)
                    tokens.Add(ws + word);
            }
            return tokens;
        }

        private string BytesToUnicode(string token)
        {
            var bytes = Encoding.UTF8.GetBytes(token);
            var mapped = bytes.Select(b => _byte2unicode[b]);
            return string.Concat(mapped);
        }

        private List<int> EncodeInternal(string text)
        {
            var outIds = new List<int>();
            foreach (var token in ByteLevelPreTokenize(text))
            {
                var unicode = BytesToUnicode(token);
                var bpe = Bpe(unicode);
                foreach (var piece in bpe.Split(' '))
                    outIds.Add(_vocab.TryGetValue(piece, out var id) ? id : _unkId);
            }
            return outIds;
        }

        private string Bpe(string token)
        {
            lock (_cacheLock)
            {
                if (_cache.TryGetValue(token, out var cached))
                    return cached;
            }

            var chars = token.Select(c => c.ToString()).ToList();
            if (chars.Count == 1)
            {
                lock (_cacheLock) AddCache(token, token);
                return token;
            }

            var word = new List<string>(chars);
            var pairs = GetPairs(word);

            while (pairs.Count > 0)
            {
                (string, string)? minPair = null;
                int minRank = int.MaxValue;
                foreach (var p in pairs)
                {
                    if (_mergeRanks.TryGetValue(p, out int rank) && rank < minRank)
                    {
                        minRank = rank;
                        minPair = p;
                    }
                }
                if (minPair == null) break;

                var (a, b) = minPair.Value;
                var newWord = new List<string>();
                int i = 0;
                while (i < word.Count)
                {
                    int j = word.IndexOf(a, i);
                    if (j == -1)
                    {
                        newWord.AddRange(word.Skip(i));
                        break;
                    }
                    newWord.AddRange(word.Skip(i).Take(j - i));
                    i = j;
                    if (i < word.Count - 1 && word[i] == a && word[i + 1] == b)
                    {
                        newWord.Add(a + b);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }
                word = newWord;
                if (word.Count == 1) break;
                pairs = GetPairs(word);
            }

            string result = string.Join(' ', word);
            lock (_cacheLock) AddCache(token, result);
            return result;
        }

        private void AddCache(string key, string value)
        {
            if (_cache.Count >= CACHE_MAX)
                _cache.Clear();
            _cache[key] = value;
        }

        private static HashSet<(string, string)> GetPairs(List<string> word)
        {
            var set = new HashSet<(string, string)>();
            for (int i = 0; i < word.Count - 1; i++)
                set.Add((word[i], word[i + 1]));
            return set;
        }

        // RoBERTa ペア入力 (<s> A </s></s> B </s>)
        public (long[] inputIds, long[] attentionMask) EncodePair(string premise, string hypothesis, int maxTotal)
        {
            var a = EncodeInternal(premise);
            var b = EncodeInternal(hypothesis);

            // 特殊: <s> A </s></s> B </s>
            const int special = 4;
            int room = Math.Max(0, maxTotal - special);

            // 最低確保
            const int MIN_A = 8;
            const int MIN_B = 8;

            int aTake = a.Count;
            int bTake = b.Count;

            if (aTake + bTake > room)
            {
                // 初期配分
                double ratio = aTake / (double)(aTake + bTake + 1e-9);
                int aAlloc = (int)Math.Round(room * ratio);
                int bAlloc = room - aAlloc;

                // 最低長保証
                if (aAlloc < MIN_A) { bAlloc = Math.Max(0, bAlloc - (MIN_A - aAlloc)); aAlloc = MIN_A; }
                if (bAlloc < MIN_B) { aAlloc = Math.Max(0, aAlloc - (MIN_B - bAlloc)); bAlloc = MIN_B; }

                aTake = Math.Min(aTake, aAlloc);
                bTake = Math.Min(bTake, bAlloc);
            }

            var ids = new List<int>(1 + aTake + 1 + 1 + bTake + 1)
            {
                _clsId
            };
            ids.AddRange(a.Take(aTake));
            ids.Add(_sepId);
            ids.Add(_sepId);
            ids.AddRange(b.Take(bTake));
            ids.Add(_sepId);

            if (ids.Count > maxTotal)
                ids = ids.Take(maxTotal).ToList();

            var attn = Enumerable.Repeat(1L, ids.Count).ToList();
            if (ids.Count < maxTotal)
            {
                int pad = maxTotal - ids.Count;
                ids.AddRange(Enumerable.Repeat(_padId, pad));
                attn.AddRange(Enumerable.Repeat(0L, pad));
            }

            return (ids.Select(i => (long)i).ToArray(), attn.ToArray());
        }

        // === 追加: 単一テキストのエンコード（Embeddingモデル用） ===
        /// <summary>
        /// 単一テキストをトークン化して ID 配列を返す。
        /// 形式: [CLS] tokens [SEP]
        /// </summary>
        /// <param name="text">入力テキスト</param>
        /// <param name="maxLength">最大トークン数（特殊トークン含む）</param>
        /// <returns>トークンID配列</returns>
        public long[] Encode(string text, int maxLength = 512)
        {
            var ids = EncodeInternal(text);

            // <s> tokens </s> の形式で構築
            var result = new List<int>(Math.Min(ids.Count + 2, maxLength))
            {
                _clsId
            };

            // 最大長から特殊トークン分（前後2つ）を引いた数だけ追加
            int takeCount = Math.Min(ids.Count, maxLength - 2);
            result.AddRange(ids.Take(takeCount));
            result.Add(_sepId);

            return result.Select(i => (long)i).ToArray();
        }

        // pre_tokenizer に ByteLevel があるかだけを判定
        private static bool ContainsByteLevel(JsonElement elem)
        {
            bool found = false;

            void Walk(JsonElement e)
            {
                if (found) return;

                if (e.ValueKind == JsonValueKind.Object)
                {
                    if (e.TryGetProperty("type", out var typeEl) && typeEl.ValueKind == JsonValueKind.String)
                    {
                        var t = typeEl.GetString() ?? "";
                        if (t.IndexOf("ByteLevel", StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            found = true;
                            return;
                        }
                    }
                    foreach (var p in e.EnumerateObject())
                        Walk(p.Value);
                }
                else if (e.ValueKind == JsonValueKind.Array)
                {
                    foreach (var c in e.EnumerateArray())
                        Walk(c);
                }
            }

            Walk(elem);
            return found;
        }

        // add_prefix_space を抽出（Sequence 配下も探索）
        private static bool TryExtractAddPrefixSpaceFromPreTokenizer(JsonElement elem, out bool addPrefixSpace)
        {
            addPrefixSpace = false;

            bool MatchByteLevel(JsonElement obj, out bool aps)
            {
                aps = false;
                if (obj.ValueKind != JsonValueKind.Object) return false;

                if (obj.TryGetProperty("type", out var typeEl) && typeEl.ValueKind == JsonValueKind.String)
                {
                    var t = typeEl.GetString() ?? "";
                    if (t.IndexOf("ByteLevel", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        var v = TryGetBool(obj, "add_prefix_space");
                        if (v.HasValue)
                        {
                            aps = v.Value;
                            return true;
                        }
                    }
                    if (t.IndexOf("Sequence", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        if (obj.TryGetProperty("pretokenizers", out var arr) || obj.TryGetProperty("pre_tokenizers", out arr))
                        {
                            if (arr.ValueKind == JsonValueKind.Array)
                            {
                                foreach (var child in arr.EnumerateArray())
                                {
                                    if (MatchByteLevel(child, out var foundAps))
                                    {
                                        aps = foundAps;
                                        return true;
                                    }
                                }
                            }
                        }
                        // 念のため、配下の配列/オブジェクトも総当たりで探索
                        foreach (var prop in obj.EnumerateObject())
                        {
                            if (prop.Value.ValueKind == JsonValueKind.Array)
                            {
                                foreach (var child in prop.Value.EnumerateArray())
                                {
                                    if (MatchByteLevel(child, out var foundAps))
                                    {
                                        aps = foundAps;
                                        return true;
                                    }
                                }
                            }
                            else if (prop.Value.ValueKind == JsonValueKind.Object)
                            {
                                if (MatchByteLevel(prop.Value, out var foundAps))
                                {
                                    aps = foundAps;
                                    return true;
                                }
                            }
                        }
                    }
                }
                return false;
            }

            if (elem.ValueKind == JsonValueKind.Object)
            {
                if (MatchByteLevel(elem, out var aps))
                {
                    addPrefixSpace = aps;
                    return true;
                }
                // 念のため、オブジェクト配下の配列/オブジェクトも探索
                foreach (var prop in elem.EnumerateObject())
                {
                    if (prop.Value.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var child in prop.Value.EnumerateArray())
                        {
                            if (MatchByteLevel(child, out var aps2))
                            {
                                addPrefixSpace = aps2;
                                return true;
                            }
                        }
                    }
                    else if (prop.Value.ValueKind == JsonValueKind.Object)
                    {
                        if (MatchByteLevel(prop.Value, out var aps3))
                        {
                            addPrefixSpace = aps3;
                            return true;
                        }
                    }
                }
            }
            else if (elem.ValueKind == JsonValueKind.Array)
            {
                foreach (var child in elem.EnumerateArray())
                {
                    if (MatchByteLevel(child, out var aps))
                    {
                        addPrefixSpace = aps;
                        return true;
                    }
                }
            }
            return false;
        }

        private static bool? TryGetBool(JsonElement obj, string prop)
        {
            if (obj.ValueKind == JsonValueKind.Object &&
                obj.TryGetProperty(prop, out var el) &&
                (el.ValueKind == JsonValueKind.True || el.ValueKind == JsonValueKind.False))
            {
                return el.GetBoolean();
            }
            return null;
        }
    }
}