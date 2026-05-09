using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Yakumo.Affect.DataPrep
{
    internal static class DataPrepPhase0Helpers
    {
        private static readonly string[] GoEmotionLabels = new[]
        {
            "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire",
            "disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief",
            "joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
        };

        public static IEnumerable<(string text, string[] labels)> ParseGoEmotionsTsv(string path)
        {
            if (!File.Exists(path)) yield break;
            foreach (var line in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var cols = line.Split('\t');
                if (cols.Length < 2) continue;

                string text = cols[0].Trim();
                string labelToken = cols[^1].Trim();

                var labelIndices = labelToken.Split(',', StringSplitOptions.RemoveEmptyEntries);
                var labels = MapLabelTokensToNames(labelIndices).ToArray();
                if (labels.Length == 0) continue;
                yield return (text, labels);
            }
        }

        public static IEnumerable<string> MapLabelTokensToNames(IEnumerable<string> numericTokens)
        {
            foreach (var t in numericTokens)
            {
                if (int.TryParse(t, out int idx) && idx >= 0 && idx < GoEmotionLabels.Length)
                    yield return GoEmotionLabels[idx];
            }
        }

        public static IEnumerable<(string text, string[] labels)> ParseEmoBankCsv(string path)
        {
            if (!File.Exists(path)) yield break;
            bool header = true;
            foreach (var line in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var cols = SplitCsvLine(line);
                if (header) { header = false; continue; }
                if (cols.Length < 2) continue;

                string text = cols[1];
                if (string.IsNullOrWhiteSpace(text)) continue;

                yield return (text, new[] { "neutral" });
            }
        }

        private static string[] SplitCsvLine(string line)
        {
            var result = new List<string>();
            var sb = new StringBuilder();
            bool inQuotes = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }
                }
                else if (c == ',' && !inQuotes)
                {
                    result.Add(sb.ToString());
                    sb.Clear();
                }
                else
                {
                    sb.Append(c);
                }
            }
            result.Add(sb.ToString());
            return result.Select(s => s.Trim()).ToArray();
        }
    }
}