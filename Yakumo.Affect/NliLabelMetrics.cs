using System;
using System.Collections.Generic;
using System.Linq;

namespace Yakumo.Affect
{
    /// <summary>
    /// ラベル単体の精度指標
    /// </summary>
    public class NliLabelMetrics
    {
        public int TP { get; init; }
        public int FP { get; init; }
        public int FN { get; init; }
        public double Precision => TP + FP == 0 ? 0.0 : (double)TP / (TP + FP);
        public double Recall => TP + FN == 0 ? 0.0 : (double)TP / (TP + FN);
        public double F1 => Precision + Recall == 0 ? 0.0
                                    : 2.0 * Precision * Recall / (Precision + Recall);

        public override string ToString() =>
            $"TP={TP} FP={FP} FN={FN} | P={Precision:F3} R={Recall:F3} F1={F1:F3}";
    }

    /// <summary>
    /// NLI 評価実行の結果をまとめたクラス。
    /// EvalRunner.RunAndReturn() の戻り値として使用します。
    /// </summary>
    public class NliEvalResult
    {
        // ── 基本指標 ──────────────────────────────────────────────
        /// <summary>評価サンプル数</summary>
        public int Count { get; init; }

        /// <summary>Macro-F1 (canonical 7ラベル平均)</summary>
        public double MacroF1 { get; init; }

        /// <summary>TopK ヒット率 (正解が TopK に含まれる割合)</summary>
        public double TopKHitRate { get; init; }

        /// <summary>neutral と予測された割合</summary>
        public double NeutralRate { get; init; }

        /// <summary>neutral 予測のうち正解が neutral でなかった割合</summary>
        public double FalseNeutralRate { get; init; }

        // ── ラベル別詳細 ──────────────────────────────────────────
        /// <summary>
        /// ラベル別の TP / FP / FN と Precision / Recall / F1。
        /// キーは canonical 英語ラベル (surprise / anger / joy / sadness / fear / disgust / neutral)。
        /// </summary>
        public IReadOnlyDictionary<string, NliLabelMetrics> PerLabel { get; init; }
            = new Dictionary<string, NliLabelMetrics>();

        // ── 派生プロパティ ────────────────────────────────────────
        /// <summary>F1 が最も高いラベル (canonical 英語名)</summary>
        public string BestLabel => PerLabel.OrderByDescending(kv => kv.Value.F1).FirstOrDefault().Key ?? "";

        /// <summary>F1 が最も低いラベル (canonical 英語名)</summary>
        public string WorstLabel => PerLabel.OrderBy(kv => kv.Value.F1).FirstOrDefault().Key ?? "";

        // ── 表示ヘルパー ──────────────────────────────────────────
        /// <summary>1行サマリーを返します</summary>
        public string ToSummaryLine() =>
            $"N={Count} | MacroF1={MacroF1:F4} | TopK={TopKHitRate:F4} | " +
            $"Neutral={NeutralRate:F4} | FalseNeutral={FalseNeutralRate:F4}";

        /// <summary>ラベル別詳細を複数行で返します</summary>
        public string ToDetailText()
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine(ToSummaryLine());
            sb.AppendLine("--- per-label ---");
            foreach (var kv in PerLabel.OrderByDescending(x => x.Value.F1))
                sb.AppendLine($"  {kv.Key,-10}: {kv.Value}");
            return sb.ToString();
        }

        public override string ToString() => ToSummaryLine();
    }
}