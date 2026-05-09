using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Yakumo.Affect.Calibration
{
    /// <summary>
    /// キャリブレーション学習用のユーティリティクラス群
    /// EmotionCalibrationStore とは別に、学習フェーズで使用
    /// </summary>

    /// <summary>
    /// ラベル単位のキャリブレータインターフェース
    /// </summary>
    public interface ILabelCalibrator
    {
        double Calibrate(double raw);
        int SampleCount { get; }
    }

    /// <summary>
    /// Plattスケーリング: sigmoid(Ax + B)
    /// 学習時に使用（PlattLearner.cs の代替・互換用）
    /// </summary>
    public class PlattCalibrator : ILabelCalibrator
    {
        public double A { get; }
        public double B { get; }
        public int SampleCount { get; }

        public PlattCalibrator(double a, double b, int n)
        {
            A = a;
            B = b;
            SampleCount = n;
        }

        public double Calibrate(double raw)
        {
            double z = A * raw + B;
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        /// <summary>
        /// 最急降下法による Platt 係数学習
        /// </summary>
        public static PlattCalibrator Fit(List<(double score, int y)> samples, int maxIter = 100, double lr = 0.01)
        {
            double a = 0, b = 0;
            for (int iter = 0; iter < maxIter; iter++)
            {
                double da = 0, db = 0;
                foreach (var (s, y) in samples)
                {
                    double z = a * s + b;
                    double p = 1.0 / (1.0 + Math.Exp(-z));
                    double diff = p - y;
                    da += diff * s;
                    db += diff;
                }
                da /= samples.Count;
                db /= samples.Count;
                a -= lr * da;
                b -= lr * db;
            }
            return new PlattCalibrator(a, b, samples.Count);
        }
    }

    /// <summary>
    /// 単純Isotonic回帰（スコア昇順で累積平均 → 単調化）
    /// 将来の拡張用
    /// </summary>
    public class IsotonicCalibrator : ILabelCalibrator
    {
        private readonly double[] _x;
        private readonly double[] _y;
        public int SampleCount { get; }

        private IsotonicCalibrator(double[] x, double[] y)
        {
            _x = x;
            _y = y;
            SampleCount = x.Length;
        }

        public double Calibrate(double raw)
        {
            if (_x.Length == 0) return raw;
            int idx = Array.BinarySearch(_x, raw);
            if (idx >= 0) return _y[idx];
            idx = ~idx;
            if (idx <= 0) return _y[0];
            if (idx >= _x.Length) return _y[^1];
            double t = (raw - _x[idx - 1]) / (_x[idx] - _x[idx - 1]);
            return _y[idx - 1] + t * (_y[idx] - _y[idx - 1]);
        }

        public static IsotonicCalibrator Fit(List<(double score, int y)> samples)
        {
            if (samples.Count == 0)
                return new IsotonicCalibrator(Array.Empty<double>(), Array.Empty<double>());

            var sorted = samples.OrderBy(s => s.score).ToList();
            var xs = sorted.Select(s => s.score).ToArray();
            var ys = sorted.Select(s => (double)s.y).ToArray();

            // PAVA (Pool Adjacent Violators Algorithm) 簡易版
            for (int i = 1; i < ys.Length; i++)
            {
                if (ys[i] < ys[i - 1])
                {
                    double avg = (ys[i] + ys[i - 1]) / 2;
                    ys[i] = avg;
                    ys[i - 1] = avg;
                }
            }

            return new IsotonicCalibrator(xs, ys);
        }
    }

    /// <summary>
    /// 旧形式の係数ファイル読み込み用DTO（互換性維持）
    /// </summary>
    internal class LegacyCalibrationFileDto
    {
        public string Mode { get; set; } = "";
        public List<LegacyCalibrationItem> Items { get; set; } = new();
    }

    internal class LegacyCalibrationItem
    {
        public string Label { get; set; } = "";
        public double A { get; set; }
        public double B { get; set; }
        public int SampleCount { get; set; }
    }
}