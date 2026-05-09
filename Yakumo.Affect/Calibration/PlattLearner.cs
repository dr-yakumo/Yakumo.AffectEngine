using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Yakumo.Affect.Calibration
{
    /// <summary>
    /// Platt Calibration (ロジスティック回帰) による確率補正
    /// f(raw_score) = sigmoid(a * raw_score + b)
    /// </summary>
    public static class PlattLearner
    {
        private static readonly string DataRoot = Path.Combine(AppContext.BaseDirectory, "data");
        private static readonly string StatsRoot = Path.Combine(DataRoot, "stats");
        private static readonly string ScoresInputPath = Path.Combine(StatsRoot, "roberta_scores_calib.jsonl");
        private static readonly string PlattOutputPath = Path.Combine(StatsRoot, "calibration_platt.json");

        // 全14ラベル（キャリブレーション対象）
        private static readonly string[] TargetLabels = new[]
        {
            // 基本7ラベル
            "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral",
            // 拡張ラベル
            "affection", "trust", "resentment", "disbelief", "shame", "anticipation", "desire"
        };

        /// <summary>
        /// Platt 係数学習のエントリポイント
        /// </summary>
        public static void Learn(bool verbose = true)
        {
            if (verbose)
            {
                Console.WriteLine("=".PadRight(60, '='));
                Console.WriteLine("  Phase 1: Platt Calibration 学習");
                Console.WriteLine("=".PadRight(60, '='));
                Console.WriteLine();
            }

            // 1) スコアデータ読み込み
            if (!File.Exists(ScoresInputPath))
            {
                Console.WriteLine($"[ERROR] スコアデータが見つかりません: {ScoresInputPath}");
                Console.WriteLine("[INFO] 先にスコア収集を実行してください: dotnet run --phase1-collect");
                return;
            }

            var samples = LoadScoreSamples(ScoresInputPath);
            if (verbose)
                Console.WriteLine($"[OK] {samples.Count} 件のスコアデータを読み込みました");

            // 2) ラベルごとに学習
            var plattParams = new Dictionary<string, PlattCoefficients>();

            foreach (var label in TargetLabels)
            {
                var (a, b, posCount, negCount) = FitPlattForLabel(samples, label);
                plattParams[label] = new PlattCoefficients { A = a, B = b };

                if (verbose)
                {
                    Console.WriteLine($"  [{label,-10}] a={a:F4}, b={b:F4}  (pos={posCount}, neg={negCount})");
                }
            }

            // 3) JSON 出力
            var output = new PlattCalibrationData
            {
                CreatedAt = DateTime.UtcNow,
                Method = "platt",
                Labels = plattParams
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(PlattOutputPath, JsonSerializer.Serialize(output, options), Encoding.UTF8);

            if (verbose)
            {
                Console.WriteLine();
                Console.WriteLine($"[OK] 係数ファイル出力: {PlattOutputPath}");
                Console.WriteLine();
                Console.WriteLine("=".PadRight(60, '='));
                Console.WriteLine("  Platt Calibration 学習完了");
                Console.WriteLine("=".PadRight(60, '='));
                Console.WriteLine();
                Console.WriteLine("[NEXT] キャリブレーション評価:");
                Console.WriteLine("       dotnet run --eval --samples=200");
            }
        }

        /// <summary>
        /// 特定ラベルに対する Platt 係数 (a, b) を学習
        /// 最急降下法によるロジスティック回帰
        /// </summary>
        private static (double a, double b, int posCount, int negCount) FitPlattForLabel(
            List<ScoreSample> samples, string label)
        {
            // 正例: true_labels に label が含まれる
            // 負例: true_labels に label が含まれない
            var data = new List<(double x, double y)>();

            foreach (var sample in samples)
            {
                if (!sample.RawScores.TryGetValue(label, out double rawScore))
                    continue;

                // ラベル一致判定（基本7ラベルへのマッピング）
                bool isPositive = IsLabelMatch(sample.TrueLabels, label);
                data.Add((rawScore, isPositive ? 1.0 : 0.0));
            }

            int posCount = data.Count(d => d.y > 0.5);
            int negCount = data.Count - posCount;

            // サンプル不足時はデフォルト係数
            if (posCount < 3 || negCount < 3)
            {
                return (1.0, 0.0, posCount, negCount);
            }

            // 最急降下法でロジスティック回帰
            double a = 1.0;
            double b = 0.0;
            double lr = 0.1;  // 学習率
            int iterations = 500;

            for (int iter = 0; iter < iterations; iter++)
            {
                double gradA = 0, gradB = 0;

                foreach (var (x, y) in data)
                {
                    double p = Sigmoid(a * x + b);
                    double error = p - y;
                    gradA += error * x;
                    gradB += error;
                }

                gradA /= data.Count;
                gradB /= data.Count;

                a -= lr * gradA;
                b -= lr * gradB;

                // 学習率減衰
                if (iter % 100 == 0 && iter > 0)
                    lr *= 0.8;
            }

            return (a, b, posCount, negCount);
        }

        /// <summary>
        /// true_labels がターゲットラベルにマッチするか判定
        /// </summary>
        private static bool IsLabelMatch(List<string> trueLabels, string targetLabel)
        {
            // 直接一致（全ラベル共通）
            if (trueLabels.Contains(targetLabel, StringComparer.OrdinalIgnoreCase))
                return true;

            // 基本7ラベルのみ：拡張ラベルからのマッピングも考慮
            var mapping = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["joy"] = new[] { "affection", "anticipation", "desire" },
                ["sadness"] = new[] { "shame", "resentment" },
                ["anger"] = new[] { "resentment" },
                ["disgust"] = new[] { "shame" },
                ["surprise"] = new[] { "disbelief" },
                ["neutral"] = new[] { "trust" }
            };

            if (mapping.TryGetValue(targetLabel, out var extendedLabels))
            {
                return trueLabels.Any(tl =>
                    extendedLabels.Contains(tl, StringComparer.OrdinalIgnoreCase));
            }

            return false;
        }

        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        /// <summary>
        /// スコアデータ読み込み
        /// </summary>
        private static List<ScoreSample> LoadScoreSamples(string path)
        {
            var samples = new List<ScoreSample>();

            foreach (var line in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    var sample = new ScoreSample
                    {
                        Text = root.GetProperty("text").GetString() ?? "",
                        TrueLabels = new List<string>(),
                        RawScores = new Dictionary<string, double>()
                    };

                    // true_labels
                    foreach (var lbl in root.GetProperty("true_labels").EnumerateArray())
                    {
                        var s = lbl.GetString();
                        if (!string.IsNullOrEmpty(s))
                            sample.TrueLabels.Add(s);
                    }

                    // raw_scores
                    foreach (var prop in root.GetProperty("raw_scores").EnumerateObject())
                    {
                        sample.RawScores[prop.Name] = prop.Value.GetDouble();
                    }

                    samples.Add(sample);
                }
                catch
                {
                    // パースエラーはスキップ
                }
            }

            return samples;
        }

        private class ScoreSample
        {
            public string Text { get; set; } = "";
            public List<string> TrueLabels { get; set; } = new();
            public Dictionary<string, double> RawScores { get; set; } = new();
        }
    }

    /// <summary>
    /// Platt 係数 (a, b)
    /// </summary>
    public class PlattCoefficients
    {
        public double A { get; set; }
        public double B { get; set; }
    }

    /// <summary>
    /// キャリブレーション出力データ
    /// </summary>
    public class PlattCalibrationData
    {
        public DateTime CreatedAt { get; set; }
        public string Method { get; set; } = "platt";
        public Dictionary<string, PlattCoefficients> Labels { get; set; } = new();
    }
}