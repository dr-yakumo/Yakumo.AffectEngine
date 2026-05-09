using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Yakumo.Affect.Embedding
{
    /// <summary>
    /// GoEmotions Embedding のインメモリベクトルインデックス
    /// Phase 5-4 Step 5: 類似文検索
    /// </summary>
    public sealed class VectorIndex
    {
        private readonly List<IndexEntry> _entries = new();
        private bool _loaded = false;

        /// <summary>
        /// ラベルごとのドキュメント出現数（IDF 計算用）
        /// </summary>
        private readonly Dictionary<string, int> _labelDocCounts = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// シングルトンインスタンス
        /// </summary>
        public static VectorIndex Instance { get; } = new();

        /// <summary>
        /// 登録件数
        /// </summary>
        public int Count => _entries.Count;

        /// <summary>
        /// ロード済みかどうか
        /// </summary>
        public bool IsLoaded => _loaded;

        /// <summary>
        /// goemo_layer_a.jsonl からインデックスをロード
        /// </summary>
        public void Load(string? path = null)
        {
            if (_loaded) return;

            path ??= Config.AffectConfigManager.Get("embedding",
                "Index.GoEmotionsPath", "data/embeddings/goemo_layer_a.jsonl");

            // 相対パスの場合は BaseDirectory を基準に解決
            if (!Path.IsPathRooted(path))
                path = Path.Combine(AppContext.BaseDirectory, path);

            if (!File.Exists(path))
            {
                Console.WriteLine($"[VECIDX][WARN] インデックスファイルが見つかりません: {path}");
                return;
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            _entries.Clear();
            _labelDocCounts.Clear();

            foreach (var line in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    var text = root.GetProperty("text").GetString() ?? "";
                    var labels = new List<string>();

                    if (root.TryGetProperty("labels", out var labelsElem))
                    {
                        foreach (var lbl in labelsElem.EnumerateArray())
                        {
                            var labelStr = lbl.GetString();
                            if (!string.IsNullOrEmpty(labelStr))
                                labels.Add(labelStr);
                        }
                    }

                    // Base64 → float[]
                    var embBase64 = root.GetProperty("embedding_b64").GetString() ?? "";
                    var embBytes = Convert.FromBase64String(embBase64);
                    var embedding = new float[embBytes.Length / sizeof(float)];
                    Buffer.BlockCopy(embBytes, 0, embedding, 0, embBytes.Length);

                    _entries.Add(new IndexEntry
                    {
                        Text = text,
                        Labels = labels,
                        Embedding = embedding
                    });

                    // ラベルごとのドキュメント出現数をカウント
                    foreach (var label in labels.Distinct(StringComparer.OrdinalIgnoreCase))
                    {
                        if (!_labelDocCounts.ContainsKey(label))
                            _labelDocCounts[label] = 0;
                        _labelDocCounts[label]++;
                    }
                }
                catch
                {
                    // パースエラーはスキップ
                }
            }

            sw.Stop();
            _loaded = true;
            Console.WriteLine($"[VECIDX] ロード完了: {_entries.Count} 件 ({sw.ElapsedMilliseconds}ms)");
        }

        /// <summary>
        /// 指定ラベルの IDF 重みを返す: log(N / n_label)
        /// </summary>
        public double GetIdfWeight(string label)
        {
            if (_entries.Count == 0) return 1.0;

            if (_labelDocCounts.TryGetValue(label, out int docCount) && docCount > 0)
                return Math.Log((double)_entries.Count / docCount);

            return 1.0; // 未知ラベルはデフォルト重み
        }

        /// <summary>
        /// クエリベクトルに対して Top-K の類似エントリを返す
        /// </summary>
        public List<SearchResult> Search(float[] queryVector, int topK = 30, float minSimilarity = 0.0f)
        {
            if (!_loaded || _entries.Count == 0)
                return new List<SearchResult>();

            // 全件ブルートフォース（13K件なので十分高速）
            var scored = new List<(int index, float similarity)>(_entries.Count);

            for (int i = 0; i < _entries.Count; i++)
            {
                float sim = EmbeddingService.CosineSimilarity(queryVector, _entries[i].Embedding);
                if (sim >= minSimilarity)
                    scored.Add((i, sim));
            }

            // 類似度降順でソートして Top-K を取得
            scored.Sort((a, b) => b.similarity.CompareTo(a.similarity));

            var results = new List<SearchResult>(Math.Min(topK, scored.Count));
            for (int i = 0; i < Math.Min(topK, scored.Count); i++)
            {
                var entry = _entries[scored[i].index];
                results.Add(new SearchResult
                {
                    Text = entry.Text,
                    Labels = entry.Labels,
                    Similarity = scored[i].similarity
                });
            }

            return results;
        }

        /// <summary>
        /// 類似文のラベル投票でスコアを計算
        /// </summary>
        /// <param name="results">kNN 検索結果</param>
        /// <param name="useIdf">IDF 重み付けを適用するか</param>
        /// <param name="idfDampingThreshold">IDF ダンピング閾値（近傍数がこの値未満だと IDF 効果を減衰）</param>
        public Dictionary<string, double> VoteLabelScores(List<SearchResult> results, bool useIdf = false, int idfDampingThreshold = 0)
        {
            var scores = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);

            // IDF ダンピング強度: 近傍数が少ないほど IDF を 1.0 に近づける
            // idfStrength = 1.0 なら IDF フル適用、0.0 なら IDF 無効（重み=1.0）
            double idfStrength = 1.0;
            if (useIdf && idfDampingThreshold > 0 && results.Count < idfDampingThreshold)
            {
                idfStrength = (double)results.Count / idfDampingThreshold;
            }

            foreach (var result in results)
            {
                // 類似度で重み付け投票
                double simWeight = result.Similarity;

                foreach (var label in result.Labels)
                {
                    double effectiveWeight = 1.0;
                    if (useIdf)
                    {
                        double rawIdf = GetIdfWeight(label);
                        // ダンピング: effectiveIdf = 1.0 + (rawIdf - 1.0) * idfStrength
                        // idfStrength=1.0 → rawIdf そのまま
                        // idfStrength=0.0 → 1.0（IDF 無効）
                        effectiveWeight = 1.0 + (rawIdf - 1.0) * idfStrength;
                    }

                    if (!scores.ContainsKey(label))
                        scores[label] = 0;
                    scores[label] += simWeight * effectiveWeight;
                }
            }

            // 正規化（合計を1.0にする）
            double total = scores.Values.Sum();
            if (total > 0)
            {
                foreach (var key in scores.Keys.ToList())
                    scores[key] /= total;
            }

            return scores;
        }

        // ===== 内部クラス =====

        private class IndexEntry
        {
            public string Text { get; set; } = "";
            public List<string> Labels { get; set; } = new();
            public float[] Embedding { get; set; } = Array.Empty<float>();
        }

        public class SearchResult
        {
            public string Text { get; set; } = "";
            public List<string> Labels { get; set; } = new();
            public float Similarity { get; set; }
        }
    }
}