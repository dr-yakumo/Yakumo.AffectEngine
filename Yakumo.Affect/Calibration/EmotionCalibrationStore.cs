using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Yakumo.Affect.Calibration
{
    /// <summary>
    /// Platt Calibration 係数を読み込み、スコア補正を適用するストア
    /// 推論時のキャリブレーション適用を担当
    /// </summary>
    [CodeStatus(CodeStatus.Legacy,
        Note = "Platt Scaling calibration store. Currently disabled (Calibration.Enabled=false in NLI.Config). " +
               "Retained for future re-evaluation. / " +
               "Plattスケーリング・キャリブレーションストア。現在は無効化中（NLI.Config: Calibration.Enabled=false）。" +
               "将来の再検討のため保持。")]
    public class EmotionCalibrationStore
    {
        private static readonly string DataRoot = Path.Combine(AppContext.BaseDirectory, "data");
        private static readonly string StatsRoot = Path.Combine(DataRoot, "stats");
        private static readonly string PlattPath = Path.Combine(StatsRoot, "calibration_platt.json");

        private Dictionary<string, PlattCoefficients> _coefficients = new();
        private bool _loaded = false;
        private bool _enabled = false;

        /// <summary>
        /// シングルトンインスタンス
        /// </summary>
        public static EmotionCalibrationStore Instance { get; } = new();

        private EmotionCalibrationStore() { }

        /// <summary>
        /// キャリブレーションが有効か
        /// </summary>
        public bool IsEnabled => _enabled && _coefficients.Count > 0;

        /// <summary>
        /// 係数ファイルを読み込み、キャリブレーションを有効化
        /// </summary>
        public bool Load(bool verbose = true)
        {
            if (_loaded) return _enabled;

            _loaded = true;

            // Config で Calibration.Enabled を確認
            string enabledStr = Config.AffectConfigManager.Get("nli_emotion", "Calibration.Enabled", "false");
            _enabled = enabledStr.Equals("true", StringComparison.OrdinalIgnoreCase);

            if (!_enabled)
            {
                if (verbose) Console.WriteLine("[CALIB] Calibration.Enabled=false のためスキップ");
                return false;
            }

            if (!File.Exists(PlattPath))
            {
                if (verbose) Console.WriteLine($"[CALIB][WARN] 係数ファイルなし: {PlattPath}");
                _enabled = false;
                return false;
            }

            try
            {
                var json = File.ReadAllText(PlattPath);
                var data = JsonSerializer.Deserialize<PlattCalibrationData>(json);

                if (data?.Labels != null)
                {
                    _coefficients = data.Labels;
                    if (verbose) Console.WriteLine($"[CALIB][OK] {_coefficients.Count} ラベルの係数を読み込み (method={data.Method})");
                    return true;
                }
            }
            catch (Exception ex)
            {
                if (verbose) Console.WriteLine($"[CALIB][ERROR] 係数読み込み失敗: {ex.Message}");
            }

            _enabled = false;
            return false;
        }

        /// <summary>
        /// 強制リロード（設定変更後に使用）
        /// </summary>
        public void Reload()
        {
            _loaded = false;
            _coefficients.Clear();
            Load();
        }

        /// <summary>
        /// 生スコアをキャリブレーション済み確率に変換
        /// f(raw) = sigmoid(a * raw + b)
        /// </summary>
        public double Calibrate(string label, double rawScore)
        {
            if (!_enabled)
                return rawScore;

            string canonicalLabel = NormalizeToCanonical(label);

            if (_coefficients.TryGetValue(canonicalLabel, out var coef))
            {
                double calibrated = Sigmoid(coef.A * rawScore + coef.B);
                return calibrated;
            }

            return rawScore;
        }

        /// <summary>
        /// スコア辞書全体をキャリブレーション（正規化付き）
        /// Calibration 後に全スコアの合計が 1.0 になるよう正規化し、
        /// 特定ラベル（joy等）の B 係数バイアスを除去する
        /// </summary>
        public Dictionary<string, double> CalibrateAllCanonical(Dictionary<string, double> rawScores)
        {
            if (!_enabled)
                return rawScores;

            var calibrated = new Dictionary<string, double>(rawScores.Count);
            double sum = 0;

            foreach (var kv in rawScores)
            {
                double val = Calibrate(kv.Key, kv.Value);
                calibrated[kv.Key] = val;
                sum += val;
            }

            // 正規化: 合計で割ることでラベル間バイアスを除去
            if (sum > 0)
            {
                foreach (var key in calibrated.Keys.ToList())
                {
                    calibrated[key] /= sum;
                }
            }

            return calibrated;
        }

        /// <summary>
        /// ローカライズ済みラベル → 英語 canonical への正規化
        /// </summary>
        private static string NormalizeToCanonical(string label)
        {
            var jpToEn = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                // 基本7ラベル
                ["驚き"] = "surprise",
                ["怒り"] = "anger",
                ["喜び"] = "joy",
                ["悲しみ"] = "sadness",
                ["恐れ"] = "fear",
                ["嫌悪"] = "disgust",
                ["中立"] = "neutral",
                // 拡張ラベル
                ["親しみ・愛情"] = "affection",
                ["信頼"] = "trust",
                ["恨み"] = "resentment",
                ["落胆"] = "disbelief",
                ["恥・罪悪感"] = "shame",
                ["期待"] = "anticipation",
                ["欲望"] = "desire"
            };

            if (jpToEn.TryGetValue(label, out var canonical))
                return canonical;

            return label.ToLowerInvariant();
        }

        /// <summary>
        /// デバッグ用: 係数一覧を表示
        /// </summary>
        public void DumpCoefficients()
        {
            Console.WriteLine("[CALIB] Platt 係数一覧:");
            foreach (var kv in _coefficients)
            {
                Console.WriteLine($"  {kv.Key,-10}: a={kv.Value.A:F4}, b={kv.Value.B:F4}");
            }
        }

        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    }
}