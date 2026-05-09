using System;
using System.Collections.Generic;

namespace Yakumo.Affect.Config
{
    /// <summary>
    /// モデル設定
    /// </summary>
    public class ModelConfig
    {
        public string Name { get; set; } = "";
        public string ModelPath { get; set; } = "";
        public string Description { get; set; } = "";
        public bool RequiresTranslation { get; set; } = false;
        public string[] SupportedLanguages { get; set; } = Array.Empty<string>();
        public string ModelType { get; set; } = "";
    }

    /// <summary>
    /// Affect Engine で利用可能なモデルの定義と選択を管理する設定クラス
    /// </summary>
    public static class AffectModelConfiguration
    {
        /// <summary>
        /// 利用可能なモデルの定義
        /// </summary>
        public static readonly Dictionary<string, ModelConfig> AvailableModels = new()
        {
            ["bert_sentiment"] = new ModelConfig
            {
                Name = "BERT Sentiment",
                ModelPath = "bert_sentiment",
                Description = "多言語対応 BERT 感情分析モデル",
                RequiresTranslation = true,
                SupportedLanguages = new[] { "jp", "en" },
                ModelType = "BERT"
            },

            ["xnli_mbert"] = new ModelConfig
            {
                Name = "mBERT XNLI",
                ModelPath = "xnli_mbert",
                Description = "多言語 BERT による多言語推論モデル",
                RequiresTranslation = false,
                SupportedLanguages = new[] { "jp", "en", "zh", "ko", "fr", "de", "es" },
                ModelType = "BERT"
            },

            ["roberta-large-mnli"] = new ModelConfig
            {
                Name = "RoBERTa Large MNLI",
                ModelPath = "roberta-large-mnli",
                Description = "高精度用途向け RoBERTa 推論モデル",
                RequiresTranslation = true,
                SupportedLanguages = new[] { "en" },
                ModelType = "RoBERTa"
            }
        };

        /// <summary>
        /// 設定からモデル設定を取得します
        /// </summary>
        public static ModelConfig GetModelConfig()
        {
            try
            {
                string modelKey = AffectConfigManager.Get("nli_model", "Model", "bert_sentiment");

                if (AvailableModels.TryGetValue(modelKey, out ModelConfig? config))
                {
                    Console.WriteLine($"[INFO] 使用モデル: {config.Name} ({config.ModelType})");
                    return config;
                }

                // デフォルトモデルを返す
                Console.WriteLine($"[WARNING] 指定されたモデル '{modelKey}' が見つかりません。デフォルトモデルを使用します。");
                return AvailableModels["bert_sentiment"];
            }
            catch (Exception ex)
            {
                Console.WriteLine($"モデル設定読み込みエラー: {ex.Message}");
                return AvailableModels["bert_sentiment"];
            }
        }

        /// <summary>
        /// 利用可能なモデル一覧を表示します（デバッグ用）
        /// </summary>
        public static void DisplayAvailableModels()
        {
            Console.WriteLine("=== 利用可能なモデル ===");
            foreach (var model in AvailableModels)
            {
                Console.WriteLine($"キー: {model.Key}");
                Console.WriteLine($"  名前: {model.Value.Name}");
                Console.WriteLine($"  タイプ: {model.Value.ModelType}");
                Console.WriteLine($"  説明: {model.Value.Description}");
                Console.WriteLine($"  翻訳: {(model.Value.RequiresTranslation ? "必要" : "不要")}");
                Console.WriteLine($"  対応言語: {string.Join(", ", model.Value.SupportedLanguages)}");
                Console.WriteLine();
            }
            Console.WriteLine("=======================");
        }
    }
}
