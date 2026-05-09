using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Yakumo.Affect.DataPrep
{
    /// <summary>
    /// Phase0: 感情コーパス取得＆前処理（GoEmotions / EmoBank / NRC手動配置オプション）
    /// </summary>
    public static class DataPrepPhase0
    {
        // 保存ルート
        private static readonly string DataRoot = Path.Combine(AppContext.BaseDirectory, "data");
        private static readonly string RawRoot = Path.Combine(DataRoot, "raw");
        private static readonly string WorkingRoot = Path.Combine(DataRoot, "working");
        private static readonly string SplitsRoot = Path.Combine(DataRoot, "splits");
        private static readonly string MappingRoot = Path.Combine(DataRoot, "mapping");
        private static readonly string StatsRoot = Path.Combine(DataRoot, "stats");

        // 出力ファイル
        private static readonly string GoEmoCleanPath = Path.Combine(WorkingRoot, "go_emotions.cleaned.jsonl");
        private static readonly string EmoBankCleanPath = Path.Combine(WorkingRoot, "emobank.cleaned.jsonl");
        private static readonly string NrcEmotionPath = Path.Combine(WorkingRoot, "nrc_emotion_lexicon.tsv");
        private static readonly string NrcVadPath = Path.Combine(WorkingRoot, "nrc_vad_lexicon.tsv");
        private static readonly string LabelMappingPath = Path.Combine(MappingRoot, "label_mapping.yaml");

        // 手動配置期待ファイル名
        private static readonly string NrcEmotionManualFile = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt";
        private static readonly string NrcVadManualFile = "NRC-VAD-Lexicon-Aug2018Release.txt";

        // GoEmotions元ラベル（28ラベル: インデックス 0-27）
        private static readonly string[] GoEmoOriginalLabels = new[]
        {
            "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire",
            "disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love",
            "nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
        };

        // Yakumo拡張ラベル写像（修正版v2: インデックスバグ修正後・admiration除外）
        // GoEmotions 28ラベル → Yakumo 14ラベル（1対1マッピング、重複なし）
        private static readonly Dictionary<string, string[]> YakumoMapping = new()
        {
            // === 基本7感情 ===
            ["joy"]         = new[] { "joy", "amusement", "relief" },
            ["sadness"]     = new[] { "sadness", "grief", "disappointment" },
            ["anger"]       = new[] { "anger", "annoyance" },
            ["fear"]        = new[] { "fear", "nervousness" },
            ["disgust"]     = new[] { "disgust" },
            ["surprise"]    = new[] { "surprise" },
            ["neutral"]     = new[] { "neutral", "realization" },

            // === 拡張7感情 ===
            ["affection"]    = new[] { "love", "caring" },
            ["trust"]        = new[] { "approval", "pride", "gratitude", "admiration" },  // admiration 復帰
            ["anticipation"] = new[] { "excitement", "optimism", "curiosity" },
            ["resentment"]   = new[] { "disapproval", "remorse" },              // disapproval が機能するようになる
            ["disbelief"]    = new[] { "confusion" },
            ["shame"]        = new[] { "embarrassment" },
            ["desire"]       = new[] { "desire" },
        };

        // ソースURL
        private static readonly Dictionary<string, List<string>> SourceUrlCandidates = new()
        {
            ["goemotions_train"] = new() { "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv" },
            ["goemotions_dev"] = new() { "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv" },
            ["goemotions_test"] = new() { "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv" },
            ["emobank"] = new() { "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv" }
        };

        // 正規化Regex
        private static readonly Regex UrlRegex = new(@"https?://\S+", RegexOptions.Compiled);
        private static readonly Regex MentionRegex = new(@"@\w+", RegexOptions.Compiled);
        private static readonly Regex MultiSpaceRegex = new(@"\s{2,}", RegexOptions.Compiled);

        public static async Task RunAsync(bool verbose = true)
        {
            EnsureDirs();
            if (verbose) Console.WriteLine("[Phase0] 開始: データ取得と前処理");

            // ラベルモードを表示
            string labelMode = Config.AffectConfigManager.Get("nli_emotion", "GoEmoLabelMode", "yakumo14");
            if (verbose) Console.WriteLine($"[Phase0] GoEmoLabelMode = {labelMode}");

            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
            var rawFiles = new Dictionary<string, string>();

            // 1) ダウンロード
            foreach (var kv in SourceUrlCandidates)
            {
                var localPath = Path.Combine(RawRoot, kv.Key + ".raw");
                if (File.Exists(localPath))
                {
                    if (verbose) Console.WriteLine($"[SKIP] 既存: {localPath}");
                    rawFiles[kv.Key] = localPath;
                    continue;
                }

                bool success = false;
                foreach (var url in kv.Value)
                {
                    try
                    {
                        var bytes = await http.GetByteArrayAsync(url);
                        if (bytes.Length < 50) throw new Exception("内容が小さ過ぎるため失敗扱い");
                        await File.WriteAllBytesAsync(localPath, bytes);
                        if (verbose) Console.WriteLine($"[DL] {kv.Key} -> {localPath} ({bytes.Length} bytes)");
                        success = true;
                        break;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[WARN] {kv.Key} 取得失敗 URL={url} ({ex.Message})");
                    }
                }
                if (success) rawFiles[kv.Key] = localPath;
                else Console.WriteLine($"[ERROR] {kv.Key} の全候補URL取得失敗");
            }

            // 2) GoEmotions 統合＋前処理
            var goLines = new List<GoEmoItem>();
            foreach (var part in new[] { "goemotions_train", "goemotions_dev", "goemotions_test" })
                if (rawFiles.TryGetValue(part, out var path))
                    goLines.AddRange(ParseGoEmotionsTsv(path));

            if (verbose) Console.WriteLine($"[INFO] GoEmotions 総件数(統合): {goLines.Count}");

            var goClean = goLines
                .Select(CleanGoEmo)
                .Where(i => i != null)
                .Cast<GoEmoItem>()
                .ToList();

            var dedupList = Deduplicate(goClean, i => i.Text).ToList();
            if (verbose) Console.WriteLine($"[INFO] GoEmotions 前処理後: {goClean.Count} → 重複後: {dedupList.Count}");
            if (dedupList.Count == 0)
                Console.WriteLine("[WARN] GoEmotions 有効件数が 0 件。");

            // ★ ラベル分布をログ出力
            if (verbose && dedupList.Count > 0)
            {
                var labelCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                foreach (var item in dedupList)
                    foreach (var lbl in item.Labels)
                    {
                        if (!labelCounts.ContainsKey(lbl)) labelCounts[lbl] = 0;
                        labelCounts[lbl]++;
                    }
                Console.WriteLine($"[Phase0] ラベル分布 (mode={labelMode}):");
                foreach (var kv in labelCounts.OrderByDescending(x => x.Value))
                    Console.WriteLine($"  {kv.Key}: {kv.Value} 件");
            }

            // 3) EmoBank
            var emoBankClean = new List<EmoBankItem>();
            if (rawFiles.TryGetValue("emobank", out var emoPath))
            {
                var ebParsed = ParseEmoBankCsv(emoPath);
                emoBankClean = ebParsed
                    .Select(CleanEmoBank)
                    .Where(i => i != null)
                    .Cast<EmoBankItem>()
                    .ToList();
                if (verbose) Console.WriteLine($"[INFO] EmoBank 件数: {ebParsed.Count} → 前処理後: {emoBankClean.Count}");
                if (emoBankClean.Count == 0)
                    Console.WriteLine("[WARN] EmoBank 有効件数が 0 件。");
            }
            else
            {
                Console.WriteLine("[WARN] EmoBank 未取得");
            }

            // 4) NRC 手動配置確認
            var nrcEmotionManualPath = Path.Combine(RawRoot, NrcEmotionManualFile);
            var nrcVadManualPath = Path.Combine(RawRoot, NrcVadManualFile);

            if (File.Exists(nrcEmotionManualPath))
            {
                File.Copy(nrcEmotionManualPath, NrcEmotionPath, overwrite: true);
                Console.WriteLine($"[OK] NRC Emotion Lexicon 手動取り込み: {NrcEmotionPath}");
            }
            else Console.WriteLine($"[INFO] NRC Emotion Lexicon 未配置: raw/{NrcEmotionManualFile}");

            if (File.Exists(nrcVadManualPath))
            {
                File.Copy(nrcVadManualPath, NrcVadPath, overwrite: true);
                Console.WriteLine($"[OK] NRC VAD Lexicon 手動取り込み: {NrcVadPath}");
            }
            else Console.WriteLine($"[INFO] NRC VAD Lexicon 未配置: raw/{NrcVadManualFile}");

            // 5) ラベル写像 YAML
            WriteMappingYaml(LabelMappingPath);
            if (verbose) Console.WriteLine($"[OK] 写像ファイル出力: {LabelMappingPath}");

            // 6) JSONL 出力（前処理済み）
            WriteJsonl(GoEmoCleanPath, dedupList.Select(MapGoEmoToJson));
            WriteJsonl(EmoBankCleanPath, emoBankClean.Select(MapEmoBankToJson));
            if (verbose) Console.WriteLine($"[OK] JSONL 出力: {GoEmoCleanPath}, {EmoBankCleanPath}");

            // 7) 分割
            if (dedupList.Count > 0)
                MakeGoEmoSplits(dedupList, verbose);
            else
                Console.WriteLine("[SKIP] GoEmotions 分割（有効データ0件）");

            // 8) スタブ統計
            var baselineStub = new
            {
                created_at = DateTime.UtcNow,
                goemotions_total = dedupList.Count,
                emobank_total = emoBankClean.Count,
                label_mode = labelMode,
                neutral_ratio_stub = dedupList.Count == 0
                    ? 0.0
                    : (double)dedupList.Count(x => x.Labels.Contains("neutral")) / dedupList.Count,
                note = "Phase0完了（NRCは任意）。次段: モデルスコア抽出→キャリブレーション"
            };
            Directory.CreateDirectory(StatsRoot);
            var statsPath = Path.Combine(StatsRoot, "baseline_stub.json");
            File.WriteAllText(statsPath, JsonSerializer.Serialize(baselineStub, new JsonSerializerOptions { WriteIndented = true }));
            if (verbose) Console.WriteLine($"[OK] スタブ統計出力: {statsPath}");

            Console.WriteLine("[Phase0] 完了");
        }

        #region GoEmotions処理


        private static List<string> MapLabelTokensToNames(string token)
        {
            var names = new List<string>();
            foreach (var part in token.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (int.TryParse(part, out var idx))
                {
                    if (idx >= 0 && idx < GoEmoOriginalLabels.Length)
                        names.Add(GoEmoOriginalLabels[idx]);
                }
                else
                {
                    var name = part.ToLowerInvariant();
                    if (GoEmoOriginalLabels.Contains(name))
                        names.Add(name);
                }
            }
            return names;
        }

        private static GoEmoItem? CleanGoEmo(GoEmoItem item)
        {
            var norm = Normalize(item.Text);
            if (!IsLikelyEnglish(norm)) return null;
            var tokens = Tokenize(norm);
            if (tokens.Length < 2 || tokens.Length > 60) return null;

            // ★ Config スイッチでラベルモードを切り替え
            string labelMode = Config.AffectConfigManager.Get("nli_emotion", "GoEmoLabelMode", "yakumo14");

            var mapped = new HashSet<string>();

            if (labelMode.Equals("raw28", StringComparison.OrdinalIgnoreCase))
            {
                // raw28: GoEmotions 原本ラベルをそのまま使用（マッピングなし）
                foreach (var rawLabel in item.RawLabels)
                {
                    if (GoEmoOriginalLabels.Contains(rawLabel))
                        mapped.Add(rawLabel);
                }
            }
            else
            {
                // yakumo14: Yakumo 14ラベルにマッピング（デフォルト）
                foreach (var (yakumoLabel, srcList) in YakumoMapping)
                    if (srcList.Any(src => item.RawLabels.Contains(src)))
                        mapped.Add(yakumoLabel);

                // ★ 基本7ラベル直接追加は削除
                // YakumoMapping で既に基本7ラベルはカバー済み（HashSet で重複排除）
            }

            if (mapped.Count == 0) return null;

            item.Text = norm;
            item.Labels = mapped.ToList();
            return item;
        }

        private static string MapGoEmoToJson(GoEmoItem item)
        {
            var obj = new
            {
                id = Guid.NewGuid().ToString(),
                text = item.Text,
                labels = item.Labels,
                raw = item.RawLabels
            };
            return JsonSerializer.Serialize(obj);
        }
        #endregion

        #region EmoBank処理
        private static List<EmoBankItem> ParseEmoBankCsv(string path)
        {
            var list = new List<EmoBankItem>();
            bool header = true;
            foreach (var line in File.ReadLines(path))
            {
                if (header) { header = false; continue; }
                if (string.IsNullOrWhiteSpace(line)) continue;
                var cols = SplitCsv(line);
                // id,split,V,A,D,text の6列が必要
                if (cols.Length < 6) continue;

                // V,A,Dは3,4,5列目 (インデックス 2,3,4)
                if (!double.TryParse(cols[2], NumberStyles.Any, CultureInfo.InvariantCulture, out var v)) continue;
                if (!double.TryParse(cols[3], NumberStyles.Any, CultureInfo.InvariantCulture, out var a)) continue;
                if (!double.TryParse(cols[4], NumberStyles.Any, CultureInfo.InvariantCulture, out var d)) continue;

                list.Add(new EmoBankItem
                {
                    // textは6列目 (インデックス 5)
                    Text = cols[5],
                    Valence = v,
                    Arousal = a,
                    Dominance = d
                });
            }
            return list;
        }

        private static EmoBankItem? CleanEmoBank(EmoBankItem item)
        {
            var norm = Normalize(item.Text);
            if (!IsLikelyEnglish(norm)) return null;
            var tokens = Tokenize(norm);
            if (tokens.Length < 2 || tokens.Length > 60) return null;
            item.Text = norm;
            return item;
        }


        private static string MapEmoBankToJson(EmoBankItem item)
        {
            var obj = new
            {
                id = Guid.NewGuid().ToString(),
                text = item.Text,
                vad = new { valence = item.Valence, arousal = item.Arousal, dominance = item.Dominance }
            };
            return JsonSerializer.Serialize(obj);
        }
        #endregion

        private static void EnsureDirs()
        {
            Directory.CreateDirectory(DataRoot);
            Directory.CreateDirectory(RawRoot);
            Directory.CreateDirectory(WorkingRoot);
            Directory.CreateDirectory(SplitsRoot);
            Directory.CreateDirectory(MappingRoot);
            Directory.CreateDirectory(StatsRoot);
        }

        private static string Normalize(string text)
        {
            var t = text.Replace("\r", " ").Replace("\n", " ");
            t = UrlRegex.Replace(t, " ");
            t = MentionRegex.Replace(t, " ");
            t = MultiSpaceRegex.Replace(t, " ").Trim();
            return t;
        }

        private static bool IsLikelyEnglish(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return false;
            int letters = text.Count(c => char.IsAsciiLetter(c));
            double ratio = (double)letters / Math.Max(1, text.Length);
            return ratio >= 0.4;
        }

        private static string[] Tokenize(string text) =>
            text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        private static IEnumerable<T> Deduplicate<T>(IEnumerable<T> items, Func<T, string> keySelector)
        {
            var seen = new HashSet<ulong>();
            foreach (var it in items)
            {
                var text = keySelector(it);
                var hash = SimHash64(text);
                if (seen.Any(h => HammingDistance(h, hash) <= 3)) continue;
                seen.Add(hash);
                yield return it;
            }
        }

        private static ulong SimHash64(string text)
        {
            var grams = new List<string>();
            var norm = text.ToLowerInvariant();
            for (int i = 0; i < norm.Length - 1; i++)
                grams.Add(norm.Substring(i, 2));
            var freq = grams.GroupBy(g => g).ToDictionary(g => g.Key, g => g.Count());
            int[] bits = new int[64];
            foreach (var kv in freq)
            {
                ulong h = (ulong)kv.Key.GetHashCode();
                int w = kv.Value;
                for (int b = 0; b < 64; b++)
                {
                    if (((h >> b) & 1UL) == 1UL) bits[b] += w;
                    else bits[b] -= w;
                }
            }
            ulong result = 0;
            for (int b = 0; b < 64; b++)
                if (bits[b] > 0) result |= (1UL << b);
            return result;
        }

        private static int HammingDistance(ulong a, ulong b)
        {
            ulong x = a ^ b;
            int count = 0;
            while (x != 0)
            {
                x &= (x - 1);
                count++;
            }
            return count;
        }

        private static void WriteJsonl(string path, IEnumerable<string> lines)
        {
            using var sw = new StreamWriter(path, false, new UTF8Encoding(false));
            foreach (var l in lines) sw.WriteLine(l);
        }

        private static void WriteMappingYaml(string path)
        {
            var sb = new StringBuilder();
            sb.AppendLine("# Yakumo 拡張ラベル写像 (Phase0)");
            sb.AppendLine("yakumo:");
            foreach (var kv in YakumoMapping)
            {
                sb.Append("  ").Append(kv.Key).Append(": [");
                sb.Append(string.Join(",", kv.Value));
                sb.AppendLine("]");
            }
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
        }

        private class GoEmoItem
        {
            public string Text { get; set; } = "";
            public List<string> RawLabels { get; set; } = new();
            public List<string> Labels { get; set; } = new();
        }

        private class EmoBankItem
        {
            public string Text { get; set; } = "";
            public double Valence { get; set; }
            public double Arousal { get; set; }
            public double Dominance { get; set; }
        }

        #region GoEmotions処理
        private static List<GoEmoItem> ParseGoEmotionsTsv(string path)
        {
            var list = new List<GoEmoItem>();
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var cols = line.Split('\t');
                // GoEmotionsのTSVは text, labels, id の3列
                if (cols.Length < 2) continue;

                string text = cols[0].Trim();
                // labelsは常に2列目(index 1)
                string labelsToken = cols[1].Trim();

                var mappedLabels = MapLabelTokensToNames(labelsToken);
                list.Add(new GoEmoItem { Text = text, RawLabels = mappedLabels });
            }
            return list;
        }

        #endregion

        #region 共通ユーティリティ
        private static void MakeGoEmoSplits(List<GoEmoItem> items, bool verbose)
        {
            var rnd = new Random(42);
            var shuffled = items.OrderBy(_ => rnd.Next()).ToList();

            int total = shuffled.Count;
            int trainCount = (int)(total * 0.70);
            int calibCount = (int)(total * 0.15);

            var train = shuffled.Take(trainCount).ToList();
            var calib = shuffled.Skip(trainCount).Take(calibCount).ToList();
            var eval = shuffled.Skip(trainCount + calibCount).ToList();

            WriteJsonl(Path.Combine(SplitsRoot, "goemo_train.jsonl"), train.Select(MapGoEmoToJson));
            WriteJsonl(Path.Combine(SplitsRoot, "goemo_calib.jsonl"), calib.Select(MapGoEmoToJson));
            WriteJsonl(Path.Combine(SplitsRoot, "goemo_eval.jsonl"), eval.Select(MapGoEmoToJson));

            if (verbose)
                Console.WriteLine($"[SPLIT] train={train.Count}, calib={calib.Count}, eval={eval.Count}");
        }

        /// <summary>
        /// 引用符対応 CSV 分割（EmoBank用）
        /// </summary>
        private static string[] SplitCsv(string line)
        {
            var fields = new List<string>();
            var sb = new StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];

                if (inQuotes)
                {
                    if (c == '"')
                    {
                        // 次の文字が " ならエスケープされた引用符
                        if (i + 1 < line.Length && line[i + 1] == '"')
                        {
                            sb.Append('"');
                            i++; // 2文字分進める
                        }
                        else
                        {
                            // フィールドの終わり
                            inQuotes = false;
                        }
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }
                else // 引用符の外
                {
                    if (c == '"')
                    {
                        inQuotes = true;
                    }
                    else if (c == ',')
                    {
                        fields.Add(sb.ToString());
                        sb.Clear();
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }
            }
            fields.Add(sb.ToString());
            return fields.ToArray();
        }
        #endregion
    }
}
