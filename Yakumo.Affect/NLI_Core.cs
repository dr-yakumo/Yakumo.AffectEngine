using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System.Diagnostics;
// ＊ 追加が必要な場合 ＊

namespace Yakumo.Affect
{
    /// <summary>
    /// 発話者の役割を表す列挙型
    /// </summary>
    public enum SpeakerRole
    {
        /// <summary>
        /// ユーザー（人間）の発話
        /// </summary>
        User,

        /// <summary>
        /// AI（人工知能）の応答
        /// </summary>
        AI,

        /// <summary>
        /// システムメッセージ
        /// </summary>
        System
    }

    public static class AffectTuning
    {
        // 既存プロパティ
        public static double Temperature = 0.3;
        public static string ScoreMode = "ent_minus_neu";
        public static string Aggregate = "max";
        public static int MaxTotal = 256;
        public static bool PreTok = true;
        public static bool Norm = true;
        public static bool DebugMode = false;  // ← コンストラクタから反映予定
        public static string Language = "jp";

        // 感情フィルタリング設定
        public static bool EnableEmotionFiltering = true;

        // 極性グループ（RAG拡張を統合）
        public static readonly HashSet<string> PositiveEmotions = new()
        {
            // JP
            "喜び", "親しみ・愛情", "信頼", "期待", "欲望",
            // EN
            "joy", "affection", "trust", "anticipation", "desire"
        };

        public static readonly HashSet<string> NegativeEmotions = new()
        {
            // 既存
            "悲しみ", "怒り", "恐れ", "嫌悪",
            "sadness", "anger", "fear", "disgust",
            // RAG拡張
            "恨み", "落胆", "恥・罪悪感",
            "resentment", "disbelief", "shame"
        };

        public static readonly HashSet<string> SpecialEmotions = new()
        {
            "驚き", "中立", "surprise", "neutral"
        };

        // === 追加: 拡張感情スイッチ類（Config連動） ===
        public static string LabelSetMode = "extended";           // basic | extended | auto
        public static bool ProbeOnNeutral = true;                  // auto時、中立Topなら追試
        public static double ProbeTieMargin = 0.05;                // auto時、Top1-2が僅差なら追試
        public static int ExtendedPromptsPerLabel = 2;             // 拡張ラベルの仮説文数（軽量化）

        // 追加: basic のポジ感情を拡張するためのスイッチ
        // - 親しみ・愛情 / 期待 を basic の有効ラベルに含めるか
        // - 欲望 を basic/extended/auto で個別にON/OFF
        public static bool BasicPosIncludeAffection = true;        // default: ON（ポジ3系統化）
        public static bool BasicPosIncludeAnticipation = true;     // default: ON（ポジ3系統化）
        public static bool EnableDesireBasic = false;              // default: OFF（basicではデフォルト除外）
        public static bool EnableDesireExtended = true;            // default: ON
        public static bool EnableDesireAuto = true;                // default: ON

        // 基本7ラベル（canonical）
        private static readonly string[] BaseCanon = new[]
        {
            "surprise","anger","joy","sadness","fear","disgust","neutral"
        };

        // ラベル重み（RAG拡張含む）
        // 重み調整: surprise=0.85 / joy=0.95 (日英共通) 他は維持
        private static readonly Dictionary<string, Dictionary<string, double>> _labelWeightsData = new()
        {
            ["jp"] = new()
            {
                ["驚き"]=0.85, ["怒り"]=0.90, ["喜び"]=0.95, ["悲しみ"]=1.00, ["恐れ"]=1.00, ["嫌悪"]=1.00, ["中立"]=0.90,
                ["親しみ・愛情"]=1.00, ["信頼"]=1.00, ["恨み"]=1.00, ["落胆"]=1.00, ["恥・罪悪感"]=1.00,
                ["期待"]=1.00, ["欲望"]=1.00
            },
            ["en"] = new()
            {
                ["surprise"]=0.85, ["anger"]=0.90, ["joy"]=0.95, ["sadness"]=1.00, ["fear"]=1.00, ["disgust"]=1.00, ["neutral"]=0.90,
                ["affection"]=1.00, ["trust"]=1.00, ["resentment"]=1.00, ["disbelief"]=1.00, ["shame"]=1.00,
                ["anticipation"]=1.00, ["desire"]=1.00
            }
        };

        // ロール別重み（既存）
        private static readonly Dictionary<SpeakerRole, Dictionary<string, double>> _roleBasedWeights = new()
        {
            [SpeakerRole.AI] = new()
            {
                ["驚き"]=0.3, ["surprise"]=0.3,
                ["中立"]=1.5, ["neutral"]=1.5,
                ["喜び"]=0.8, ["joy"]=0.8
            },
            [SpeakerRole.User] = new(),
            [SpeakerRole.System] = new()
            {
                ["驚き"]=0.1, ["surprise"]=0.1,
                ["中立"]=2.0, ["neutral"]=2.0
            }
        };

        public static Dictionary<string, double> GetRoleBasedWeights(SpeakerRole role) =>
            _roleBasedWeights.TryGetValue(role, out var weights) ? weights : new Dictionary<string, double>();

        // 多言語プロンプト（RAG拡張 + 拡張行追加）
        private static readonly Dictionary<string, Dictionary<string, string[]>> _promptsData = new()
        {
            ["jp"] = new()
            {
                ["驚き"] = new[]{
                    "これは驚きの感情を表す文章だ。","この出来事に驚いている。","予想外で驚いた。","I am surprised by this."
                },
                ["怒り"] = new[]{
                    "これは怒りの感情を表す文章だ。","腹が立っている。","I feel angry.",
                    "不公平で腹立たしいと感じている。","不当な結果に苛立っている。","怒りが込み上げてきている。"
                },
                ["喜び"] = new[]{
                    "これは喜びの感情を表す文章だ。","うれしく感じている。","I feel joy."
                },
                ["悲しみ"] = new[]{
                    "このテキストは悲しみを表現している。","このことを悲しく思う。","悲しみを感じる。",
                    "深い悲しみと空虚感を覚える。", "この喪失は心を打ち砕く。",
                    "孤独と圧倒的な辛さを感じる。", "この痛みは耐え難いほどだ。",
                    "押しつぶされるような絶望感に襲われる。", "悲しみに溺れている。"
                },
                ["恐れ"] = new[]{
                    "このテキストは恐れを表現している。","このことを恐れる。","恐怖を感じる。",
                    "差し迫った危険を感じて恐怖に震える。", "何か悪いことが起きる気がして怖い。",
                    "迫り来る脅威にさらされている気がする。", "危険がすぐそばにある感覚だ。"
                },
                ["嫌悪"] = new[]{
                    "これは嫌悪の感情を表す文章だ。","嫌悪感を抱いている。","I feel disgust.",
                    "これに激しい嫌悪感を覚える。","これは気持ち悪く吐き気がする。","嫌悪で思わず身を引いた。"
                },
                ["中立"] = new[]{
                    "これは特定の感情を含まない中立的な文章だ。","This is neutral.",
                    "これは感情的な偏りを含まない情報文だ。","特別な感情表現はない。","これは単に事実を述べているだけだ。"
                },

                ["親しみ・愛情"] = new[] { "これは親しみや愛情を表す文章だ。", "相手に親近感や愛情を感じている。", "I feel affection or closeness." },
                ["信頼"] = new[] { "これは信頼の感情を表す文章だ。", "相手を信頼している。", "I feel trust." },
                ["恨み"] = new[] { "これは恨みの感情を表す文章だ。", "根に持っている。", "I feel resentment." },
                ["落胆"] = new[] { "これは落胆の感情を表す文章だ。", "期待外れでがっかりしている。", "I feel disappointed." },
                ["恥・罪悪感"] = new[] { "これは恥や罪悪感を表す文章だ。", "自分を責めている。", "I feel shame or guilt." },
                ["期待"] = new[] { "これは期待の感情を表す文章だ。", "良い結果を見込んでいる。", "I feel anticipation." },
                ["欲望"] = new[] { "これは欲望の感情を表す文章だ。", "強く望んでいる。", "I feel desire." }
            },
            ["en"] = new()
            {
                ["surprise"] = new[]{
                    "This text expresses surprise.","I am surprised by this event.","This is unexpected and surprising.","I am surprised by this."
                },
                ["anger"] = new[]{
                    // 改善: 客観的・三人称表現を先頭に配置
                    "The speaker expresses anger or frustration.",
                    "Someone is upset or outraged about something.",
                    "This text contains hostile or irritated language.",
                    "The tone is aggressive or confrontational.",
                    "There is expression of annoyance or rage.",
                    "I feel angry about this."
                },
                ["joy"] = new[]{
                    "This text expresses joy.","I feel happy about this.","I feel joy."
                },
                ["sadness"] = new[]{
                    // 元のプロンプトに戻す
                    "This text expresses sadness.","I feel sad about this.","I feel sadness.",
                    "I feel deeply sad and empty.", "This loss leaves me heartbroken.",
                    "I feel isolated and overwhelmed.", "This pain feels impossible to bear.",
                    "A crushing despair overwhelms me.", "I am drowning in sorrow."
                },
                ["fear"] = new[]{
                    // より具体的で限定的な表現に変更
                    "The speaker is afraid or scared.",
                    "Someone feels fearful about a specific threat.",
                    "This text expresses genuine worry or terror.",
                    "The person feels anxious about imminent danger.",
                },
                ["disgust"] = new[]{
                    // より具体的で限定的な表現に変更
                    "The speaker feels disgusted or repulsed.",
                    "Someone finds this revolting or gross.",
                    "This text expresses strong distaste or aversion.",
                    "The person is sickened or offended by this.",
                },
                ["neutral"] = new[]{
                    "This text contains no specific emotion.","This is neutral.",
                    "This text is emotionally neutral.","No particular emotion is expressed.","It conveys information only."
                },

                ["affection"] = new[] { "This text expresses affection or fondness.", "I feel affection.", "I feel close to them." },
                ["trust"] = new[] { "This text expresses trust.", "I trust them.", "I have confidence in them." },
                ["resentment"] = new[] { "This text expresses resentment.", "I feel resentful.", "I hold a grudge." },
                ["disbelief"] = new[] { "This text expresses disbelief or disappointment.", "I feel let down.", "I can't believe it." },
                ["shame"] = new[] { "This text expresses shame or guilt.", "I feel ashamed.", "I feel guilty." },
                ["anticipation"] = new[] { "This text expresses anticipation.", "I am looking forward to it.", "I expect a good result." },
                ["desire"] = new[] { "This text expresses desire.", "I strongly want this.", "I crave it." }
            }
        };

        // 感情ラベル辞書（canonical -> localized）
        private static readonly Dictionary<string, Dictionary<string, string>> _emotionLabels = new()
        {
            ["jp"] = new()
            {
                ["surprise"]="驚き", ["anger"]="怒り", ["joy"]="喜び", ["sadness"]="悲しみ", ["fear"]="恐れ", ["disgust"]="嫌悪", ["neutral"]="中立",
                ["affection"]="親しみ・愛情", ["trust"]="信頼", ["resentment"]="恨み",
                ["disbelief"]="落胆",            // ← ここを書き換え
                ["shame"]="恥・罪悪感",
                ["anticipation"]="期待", ["desire"]="欲望"
            },
            ["en"] = new()
            {
                ["surprise"]="surprise", ["anger"]="anger", ["joy"]="joy", ["sadness"]="sadness", ["fear"]="fear", ["disgust"]="disgust", ["neutral"]="neutral",
                ["affection"]="affection", ["trust"]="trust", ["resentment"]="resentment", ["disbelief"]="disbelief", ["shame"]="shame",
                ["anticipation"]="anticipation", ["desire"]="desire"
            }
        };

        // === 追加: Configから設定反映（コンストラクタから呼ぶ）===
        public static void ApplyConfigFromManager()
        {
            // 既存の基本設定
            LabelSetMode = Config.AffectConfigManager.Get("nli_emotion", "LabelSet", LabelSetMode);

            ProbeOnNeutral = Config.AffectConfigManager
                .Get("nli_emotion", "ExtendedProbe.OnNeutral", ProbeOnNeutral ? "true" : "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

            ProbeTieMargin = Config.AffectConfigManager
                .GetDouble("nli_emotion", "ExtendedProbe.TieMargin", ProbeTieMargin);

            ExtendedPromptsPerLabel = Config.AffectConfigManager
                .GetInt("nli_emotion", "ExtendedPromptsPerLabel", ExtendedPromptsPerLabel);

            // 追加: basic のポジを3系統にするためのスイッチ
            BasicPosIncludeAffection = Config.AffectConfigManager
                .Get("nli_emotion", "BasicPos.IncludeAffection", BasicPosIncludeAffection ? "true" : "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

            BasicPosIncludeAnticipation = Config.AffectConfigManager
                .Get("nli_emotion", "BasicPos.IncludeAnticipation", BasicPosIncludeAnticipation ? "true" : "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

            // 追加: 欲望（desire）ON/OFF（basic/extended/auto 別）
            EnableDesireBasic = Config.AffectConfigManager
                .Get("nli_emotion", "EnableDesire.Basic", EnableDesireBasic ? "true" : "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

            EnableDesireExtended = Config.AffectConfigManager
                .Get("nli_emotion", "EnableDesire.Extended", EnableDesireExtended ? "true" : "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

            EnableDesireAuto = Config.AffectConfigManager
                .Get("nli_emotion", "EnableDesire.Auto", EnableDesireAuto ? "true" : "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);

            // ★ 追加: ラベル重みオーバーライド読込
            // 全キーを走査 (ConfigManager にキー列挙 API がない前提 → 想定: label明示)
            string[] overrideTargets = {
                "surprise","anger","joy","sadness","fear","disgust","neutral",
                "affection","trust","resentment","disbelief","shame","anticipation","desire"
            };

            foreach (var key in overrideTargets)
            {
                string raw = Config.AffectConfigManager.Get("nli_emotion", $"LabelWeight.{key}", "");
                if (string.IsNullOrWhiteSpace(raw)) continue;
                if (!double.TryParse(raw, out double w) || w <= 0) continue;

                // jp側 localized 名
                if (_runtimeLabelWeights.TryGetValue("jp", out var jpMap) && _emotionLabels["jp"].TryGetValue(key, out var jpLoc))
                {
                    if (jpMap.ContainsKey(jpLoc)) jpMap[jpLoc] = w;
                }
                // en側 canonical
                if (_runtimeLabelWeights.TryGetValue("en", out var enMap) && _emotionLabels["en"].TryGetValue(key, out var enLoc))
                {
                    if (enMap.ContainsKey(enLoc)) enMap[enLoc] = w;
                }
            }

            // ★ 追加: debugMode 時の重みダンプ（言語別）
            if (DebugMode)
            {
                Console.WriteLine("[CONFIG] LabelWeights dump (runtime overrides applied):");
                foreach (var lang in _runtimeLabelWeights.Keys.OrderBy(k => k))
                {
                    Console.WriteLine($"  <{lang}>");
                    foreach (var kv in _runtimeLabelWeights[lang].OrderBy(kv => kv.Key))
                    {
                        Console.WriteLine($"    {kv.Key} = {kv.Value:0.###}");
                    }
                }
            }
        }

        // === 追加: 現在の言語における「基本ラベル（localized）」 ===
        // 注意: base セットは LabelSet=basic のときだけ拡張（親しみ・愛情/期待/欲望）
        private static HashSet<string> GetBaseLocalized()
        {
            if (!EmotionLabels.TryGetValue(Language, out var map))
            {
                map = EmotionLabels["jp"];
            }

            // まず canonical 基本7を localized に変換
            var baseLocalized = BaseCanon
                .Select(c => map.TryGetValue(c, out var loc) ? loc : c)
                .ToHashSet(StringComparer.Ordinal);

            // LabelSet が basic のときのみ、ポジ系を拡張（設定で制御）
            if (LabelSetMode.Equals("basic", StringComparison.OrdinalIgnoreCase))
            {
                // 親しみ・愛情（affection）
                if (BasicPosIncludeAffection)
                {
                    var aff = GetLocalizedLabel("affection");
                    baseLocalized.Add(aff);
                }

                // 期待（anticipation）
                if (BasicPosIncludeAnticipation)
                {
                    var ant = GetLocalizedLabel("anticipation");
                    baseLocalized.Add(ant);
                }

                // 欲望（desire）— basic 用フラグが ON の場合のみ追加
                if (EnableDesireBasic)
                {
                    var des = GetLocalizedLabel("desire");
                    baseLocalized.Add(des);
                }
            }

            return baseLocalized;
        }

        // === 追加: 現在有効なラベル集合（localized） ===
        public static string[] GetActiveLabelList()
        {
            var all = EmotionLabels.TryGetValue(Language, out var labels)
                ? labels.Values.ToArray()
                : EmotionLabels["jp"].Values.ToArray();

            // basic: 拡張済み base セットのみ返す
            if (LabelSetMode.Equals("basic", StringComparison.OrdinalIgnoreCase))
            {
                var baseSet = GetBaseLocalized();
                return all.Where(l => baseSet.Contains(l)).ToArray();
            }

            // extended/auto: 全ラベル。ただし欲望はフラグで除外可能
            var active = new List<string>(all);

            // 欲望（desire）の localized 名を取得
            var desireLoc = GetLocalizedLabel("desire");

            if (LabelSetMode.Equals("extended", StringComparison.OrdinalIgnoreCase) && !EnableDesireExtended)
            {
                active.RemoveAll(l => string.Equals(l, desireLoc, StringComparison.Ordinal));
            }

            if (LabelSetMode.Equals("auto", StringComparison.OrdinalIgnoreCase) && !EnableDesireAuto)
            {
                active.RemoveAll(l => string.Equals(l, desireLoc, StringComparison.Ordinal));
            }

            return active.ToArray();
        }

        // === 追加: 現在有効なプロンプト辞書（localizedキー） ===
        public static Dictionary<string, string[]> GetActivePromptsDict()
        {
            var prompts = PromptsData.TryGetValue(Language, out var p)
                ? p
                : PromptsData["jp"];

            var baseSet = GetBaseLocalized();
            var active = new Dictionary<string, string[]>(StringComparer.Ordinal);

            bool isBasicMode = LabelSetMode.Equals("basic", StringComparison.OrdinalIgnoreCase);
            bool isExtendedMode = LabelSetMode.Equals("extended", StringComparison.OrdinalIgnoreCase);
            bool isAutoMode = LabelSetMode.Equals("auto", StringComparison.OrdinalIgnoreCase);

            // extended/auto で「欲望」を除外するかどうか
            bool allowDesireThisMode =
                (isExtendedMode && EnableDesireExtended) ||
                (isAutoMode && EnableDesireAuto) ||
                isBasicMode; // basic は baseSet 側の制御に委ねる

            var desireLoc = GetLocalizedLabel("desire");

            // ★ 追加: プロンプト数オーバーライド (Prompts.Override.<label>=N)
            //   - localized と canonical 英語両方で一致を試みる
            //   - 設定値 N>=1 のとき、そのラベルのプロンプト配列を先頭 N 件に強制
            var overrides = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            // 現在言語の localized -> canonical 逆引き辞書
            var langMap = EmotionLabels.TryGetValue(Language, out var locMap) ? locMap : EmotionLabels["jp"];
            var inverseMap = langMap.ToDictionary(kv => kv.Value, kv => kv.Key, StringComparer.OrdinalIgnoreCase);

            foreach (var kv in prompts)
            {
                string localized = kv.Key;
                string canonical = inverseMap.TryGetValue(localized, out var c) ? c : localized;

                // localized 名で試行
                string rawLoc = Config.AffectConfigManager.Get("nli_emotion", "Prompts.Override." + localized);
                // canonical 名で試行（localized が日本語でも英語キー指定を許可）
                string rawCan = string.IsNullOrEmpty(rawLoc)
                    ? Config.AffectConfigManager.Get("nli_emotion", "Prompts.Override." + canonical)
                    : null;

                string raw = !string.IsNullOrEmpty(rawLoc) ? rawLoc : rawCan;

                if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw.Trim(), out int n) && n >= 1)
                {
                    overrides[localized] = n;
                }
            }

            foreach (var kv in prompts)
            {
                // basic: 基本ラベル（拡張済み base セット）のみ使用
                if (isBasicMode)
                {
                    if (baseSet.Contains(kv.Key))
                    {
                        var arrBasic = kv.Value;
                        // オーバーライド適用（basicでも有効）
                        if (overrides.TryGetValue(kv.Key, out int forcedBasic) && forcedBasic < arrBasic.Length)
                        {
                            arrBasic = arrBasic.Take(forcedBasic).ToArray();
                        }
                        active[kv.Key] = arrBasic;
                    }
                    continue;
                }

                // extended/auto: 全ラベル。但し「欲望」は設定で除外
                if (!allowDesireThisMode && string.Equals(kv.Key, desireLoc, StringComparison.Ordinal))
                {
                    continue;
                }

                var arr = kv.Value;

                // ★ 個別オーバーライド最優先
                if (overrides.TryGetValue(kv.Key, out int forced) && forced < arr.Length)
                {
                    arr = arr.Take(forced).ToArray();
                }
                else
                {
                    // 拡張ラベルは ExtendedPromptsPerLabel で件数制限
                    bool isBaseLabel = baseSet.Contains(kv.Key);
                    if (!isBaseLabel && ExtendedPromptsPerLabel > 0 && arr.Length > ExtendedPromptsPerLabel)
                    {
                        arr = arr.Take(ExtendedPromptsPerLabel).ToArray();
                      }
                }

                active[kv.Key] = arr;
            }

            return active;
        }

        // === 追加: 現在の有効なラベルに基づく極性判定 ===
        public static bool IsPositive(string label)
        {
            // 有効ラベルに含まれないものは無視
            if (!_isActive(label)) return false;
            return PositiveEmotions.Contains(label);
        }

        public static bool IsNegative(string label)
        {
            if (!_isActive(label)) return false;
            return NegativeEmotions.Contains(label);
        }

        private static bool _isActive(string label)
        {
            // 高頻度で呼ばれるので HashSet 化
            _activeCache ??= GetActiveLabelList().ToHashSet(StringComparer.Ordinal);
            return _activeCache.Contains(label);
        }

        private static HashSet<string>? _activeCache;
        public static void InvalidateActiveCache() => _activeCache = null;

        // 公開ビュー
        //public static Dictionary<string, Dictionary<string, double>> LabelWeightsData =>
        //    new Dictionary<string, Dictionary<string, double>>(_labelWeightsData);

        // 公開ビューを runtime 優先へ
        public static Dictionary<string, Dictionary<string, double>> LabelWeightsData =>
            new(_runtimeLabelWeights.ToDictionary(
                lang => lang.Key,
                lang => lang.Value.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal),
                StringComparer.Ordinal));

        public static Dictionary<string, Dictionary<string, string[]>> PromptsData =>
            new Dictionary<string, Dictionary<string, string[]>>(_promptsData);

        public static Dictionary<string, Dictionary<string, string>> EmotionLabels =>
            new Dictionary<string, Dictionary<string, string>>(_emotionLabels);

        // 現在の言語設定に基づいて感情ラベルを取得
        //public static Dictionary<string, double> LabelWeights =>
        //    LabelWeightsData.TryGetValue(Language, out var weights) ? weights : LabelWeightsData["jp"];

        public static Dictionary<string, double> LabelWeights =>
            LabelWeightsData.TryGetValue(Language, out var weights) ? weights : LabelWeightsData["jp"];

        // 現在の言語設定に基づいてプロンプトを取得
        public static Dictionary<string, string[]> Prompts =>
            PromptsData.TryGetValue(Language, out var prompts) ? prompts : PromptsData["jp"];

        // 現在の言語設定に基づいて感情ラベル配列を取得（localized名の配列）
        public static string[] GetEmotionLabels() =>
            EmotionLabels.TryGetValue(Language, out var labels) ? labels.Values.ToArray() : EmotionLabels["jp"].Values.ToArray();

        // 感情ラベルの翻訳（canonical→localized）
        public static string GetLocalizedLabel(string emotionKey) =>
            EmotionLabels.TryGetValue(Language, out var labels) && labels.TryGetValue(emotionKey, out var label)
                ? label
                : emotionKey;

        // ★ 追加: ラベル重み外部化用ランタイム辞書
        private static Dictionary<string, Dictionary<string, double>> _runtimeLabelWeights =
            new(_labelWeightsData.ToDictionary(
                lang => lang.Key,
                lang => lang.Value.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal),
                StringComparer.Ordinal));
    }


    public partial class AffectCore : IDisposable
    {
        // BERTの特殊トークンID（mBERT共通）
        private const int PAD = 0, UNK = 100, CLS = 101, SEP = 102, MASK = 103;
        private const int IDX_CONTRA = 0, IDX_NEUTRAL = 1, IDX_ENTAIL = 2;

        // インスタンス固有の設定
        private readonly string _language;
        private readonly bool _debugMode;
        private readonly double _temperature;
        private readonly string _scoreMode;
        private readonly string _aggregate;

        // 共有リソース（モデルファイルパスなど）
        private static readonly Dictionary<string, string> ModelPaths = new()
        {
            ["xnli_mbert"] = "xnli_mbert",
            ["bert_sentiment"] = "bert_sentiment"
        };

        private static readonly string ModelDir = Path.Combine(AppContext.BaseDirectory, "libs\\models", ModelPaths["bert_sentiment"]);
        private static readonly string OnnxPath = Path.Combine(ModelDir, "model.onnx");
        private static readonly string VocabPath = Path.Combine(ModelDir, "vocab.txt");

        // リソース管理
        private WordPieceTokenizer? _tokenizer;
        private InferenceSession? _session;
        private bool _disposed = false;

        private Calibration.EmotionCalibrationStore _calibrationStore = null!;

        /// <summary>
        /// NLI_Coreの新しいインスタンスを作成します
        /// </summary>
        /// <param name="language">言語設定 ("jp" または "en")</param>
        /// <param name="debugMode">デバッグモード</param>
        /// <param name="temperature">温度パラメータ</param>
        /// <param name="scoreMode">スコアモード</param>
        /// <param name="aggregate">集約方法</param>
        // 変更: コンストラクタでConfig優先にオーバーライド
       public AffectCore(string language = "jp", bool debugMode = false, double temperature = 0.3,
                        string scoreMode = "ent_minus_neu", string aggregate = "max")
        {
            // 言語設定の検証(jp/enのみ)
            if (language != "jp" && language != "en")
            {
                throw new ArgumentException("Supported languages are 'jp' and 'en'", nameof(language));
            }
            // Configから読めるものは上書き
            string cfgScoreMode = Config.AffectConfigManager.Get("nli_model", "ScoreMode", scoreMode);
            string cfgAggregate = Config.AffectConfigManager.Get("nli_model", "Aggregate", aggregate);
            double cfgTemperature = Config.AffectConfigManager.GetDouble("nli_model", "Temperature", temperature);

            _language = language;
            _debugMode = debugMode;
            AffectTuning.DebugMode = debugMode;  // ★ 追加: 静的側へ反映
            _temperature = cfgTemperature;
            _scoreMode = cfgScoreMode;
            _aggregate = cfgAggregate;

            // raw28 + jp の組み合わせ通知（起動時に1回だけ）
            if (ShouldUseGoEmotionsModel() && _language == "jp" && IsGoEmoRaw28Mode())
            {
                Console.WriteLine("[GoEmo] raw28モード起動: 14ラベルへの圧縮処理なし(GoEmotionsの28感情をそのまま出します)。翻訳は通常通り実施。");
                Console.WriteLine("[GoEmo] raw28 mode enabled: no 14-label compression (returns raw GoEmotions 28-emotion scores). Translation still runs as usual.");
            }

            // 設定ファイルから感情フィルタリング設定を読み込む
            try
            {
                // Config から読み込み (設定ファイルが存在する場合)
                AffectTuning.EnableEmotionFiltering = Config.AffectConfigManager.Get("nli_emotion", "EnableFiltering", "true").ToLower() == "true";

                if (_debugMode)
                {
                    DebugWriteLine($"感情フィルタリング: {(AffectTuning.EnableEmotionFiltering ? "有効" : "無効")}");

                    // 強制的にフィルタ無効化（デバッグ時のみ）
                    //AffectTuning.EnableEmotionFiltering = false;
                }
            }
            catch
            {
                // 設定ファイルがない場合はデフォルト値を使用
            }

            // === 追加: 拡張感情スイッチ読込 ===
            try
            {
                AffectTuning.ApplyConfigFromManager();
                AffectTuning.InvalidateActiveCache();

                if (_debugMode)
                {
                    var active = string.Join(", ", AffectTuning.GetActiveLabelList());
                    Console.WriteLine($"[CONFIG] nli_emotion.LabelSet = {AffectTuning.LabelSetMode}");
                    if (AffectTuning.LabelSetMode.Equals("auto", StringComparison.OrdinalIgnoreCase))
                    {
                        Console.WriteLine(
                            $"[MODE] 感情ラベル: 基本 → 条件付き拡張 " +
                            $"(OnNeutral={AffectTuning.ProbeOnNeutral}, " +
                            $"TieMargin={AffectTuning.ProbeTieMargin:F2}, " +
                            $"ExtPrompts={AffectTuning.ExtendedPromptsPerLabel})"
                        );
                    }
                    else
                    {
                        var isBasic = AffectTuning.LabelSetMode.Equals("basic", StringComparison.OrdinalIgnoreCase);
                        var modeText = isBasic ? "基本" : "拡張";
                        var activeCount = AffectTuning.GetActiveLabelList().Length;

                        Console.WriteLine($"[MODE] 感情ラベル: {modeText}（有効数={activeCount}）");
                    }
                }
            } catch {}

            if (!ShouldUseGoEmotionsModel() && !ShouldUseRobertaModel())
            {
                InitializeResources();
            }

            // キャリブレーションデータの読み込み(シングルトンを使用)
            _calibrationStore = Calibration.EmotionCalibrationStore.Instance;
            _calibrationStore.Load(verbose: _debugMode);
        }

        /// <summary>
        /// デフォルト設定でインスタンスを作成する静的ファクトリメソッド
        /// </summary>
        public static AffectCore CreateDefault() => new AffectCore();

        /// <summary>
        /// 英語設定でインスタンスを作成する静的ファクトリメソッド
        /// </summary>
        public static AffectCore CreateEnglish() => new AffectCore("en");
        //public static AffectCore CreateEnglish() => new AffectCore("en", debugMode: true);

        /// <summary>
        /// 日本語設定でインスタンスを作成する静的ファクトリメソッド
        /// </summary>
        public static AffectCore CreateJapanese() => new AffectCore("jp");

        /// <summary>
        /// 現在の言語設定を取得します
        /// </summary>
        public string Language => _language;

        /// <summary>
        /// 現在のデバッグモード設定を取得します
        /// </summary>
        public bool DebugMode => _debugMode;


        /// <summary>
        /// 現在 Config で設定されているモデル名を取得します
        /// </summary>
        public string ModelName =>
            Config.AffectConfigManager.Get("nli_model", "Model", "bert_sentiment");

        [CodeStatus(CodeStatus.Legacy,
            Note = "Initializes BERT-sentiment model resources (ONNX session, WordPiece tokenizer). " +
                   "Not used when GoEmotions is active. / " +
                   "BERT-sentimentモデルリソース（ONNXセッション・WordPieceトークナイザ）の初期化。" +
                   "GoEmotionsが有効な場合は呼び出されない。")]
        private void InitializeResources()
        {
            RequireFile(OnnxPath);
            RequireFile(VocabPath);

            // WordPiece を vocab.txt で生成
            var wpOptions = new WordPieceOptions
            {
                UnknownToken = "[UNK]",
                SpecialTokens = new Dictionary<string, int>
                {
                    ["[PAD]"] = 0,
                    ["[UNK]"] = 100,
                    ["[CLS]"] = 101,
                    ["[SEP]"] = 102,
                    ["[MASK]"] = 103
                }
            };
            _tokenizer = WordPieceTokenizer.Create(VocabPath, wpOptions);

            _session = new InferenceSession(OnnxPath, new SessionOptions
            {
                IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2),
                InterOpNumThreads = 1
            });
        }

        /// <summary>
        /// 単一のテキストを分析し、結果を返します（ロール指定あり）
        /// </summary>
        public ClassificationResult AnalyzeText(string text, SpeakerRole role = SpeakerRole.User, int k = 3, double threshold = 0.25)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AffectCore));

            return ClassifyByNli(text, _tokenizer!, _session!, role, k, threshold);
        }

        /// <summary>
        /// 単一のテキストを分析し、結果を返します（互換性のため）
        /// </summary>
        public ClassificationResult AnalyzeText(string text, int k = 3, double threshold = 0.25)
        {
            return AnalyzeText(text, SpeakerRole.User, k, threshold);
        }

        /// <summary>
        /// 複数のテキストを分析し、結果を返します（ロール指定あり）
        /// </summary>
        public NliAnalysisResults AnalyzeTexts(string[] texts, SpeakerRole role = SpeakerRole.User, int k = 3, double threshold = 0.25)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AffectCore));

            var stopwatch = Stopwatch.StartNew();
            var results = new NliAnalysisResults
            {
                // ここをファイル名から設定キーへ
                ModelUsed = GetCurrentModelKey()
            };

            foreach (var text in texts)
            {
                var result = AnalyzeText(text, role, k, threshold);
                results.Results.Add(result);
            }

            stopwatch.Stop();
            results.ProcessingTime = stopwatch.Elapsed;
            return results;
        }

        /// <summary>
        /// 複数のテキストを分析し、結果を返します（互換性のため）
        /// </summary>
        public NliAnalysisResults AnalyzeTexts(string[] texts, int k = 3, double threshold = 0.25)
        {
            return AnalyzeTexts(texts, SpeakerRole.User, k, threshold);
        }

        /// <summary>
        /// 単一のテキストを非同期で分析し、結果を返します（ロール指定あり）
        /// </summary>
        public async Task<ClassificationResult> AnalyzeTextAsync(string text, SpeakerRole role = SpeakerRole.User, int k = 3, double threshold = 0.25)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AffectCore));

            return await ClassifyByNliAsync(text, _tokenizer!, _session!, role, k, threshold);
        }

        /// <summary>
        /// 単一のテキストを非同期で分析し、結果を返します（互換性のため）
        /// </summary>
        public async Task<ClassificationResult> AnalyzeTextAsync(string text, int k = 3, double threshold = 0.25)
        {
            return await AnalyzeTextAsync(text, SpeakerRole.User, k, threshold);
        }

        /// <summary>
        /// 複数のテキストを非同期で分析し、結果を返します（ロール指定あり）
        /// </summary>
        public async Task<NliAnalysisResults> AnalyzeTextsAsync(string[] texts, SpeakerRole role = SpeakerRole.User, int k = 3, double threshold = 0.25)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AffectCore));

            var stopwatch = Stopwatch.StartNew();
            var results = new NliAnalysisResults
            {
                ModelUsed = GetCurrentModelKey()
            };

            // 並列処理でパフォーマンス向上
            var tasks = texts.Select(text => AnalyzeTextAsync(text, role, k, threshold)).ToArray();
            var analysisResults = await Task.WhenAll(tasks);

            results.Results.AddRange(analysisResults);

            stopwatch.Stop();
            results.ProcessingTime = stopwatch.Elapsed;
            return results;
        }

        /// <summary>
        /// 複数のテキストを非同期で分析し、結果を返します（互換性のため）
        /// </summary>
        public async Task<NliAnalysisResults> AnalyzeTextsAsync(string[] texts, int k = 3, double threshold = 0.25)
        {
            return await AnalyzeTextsAsync(texts, SpeakerRole.User, k, threshold);
        }

        [CodeStatus(CodeStatus.Legacy,
            Note = "BERT-sentiment asynchronous inference path. Retained as fallback/research. " +
                   "Primary path is GoEmotions (GoEmo_Core.cs). / " +
                   "BERT-sentiment非同期推論パス。研究・フォールバック用として保持。主系統はGoEmo_Core.cs。")]
        private async Task<ClassificationResult> ClassifyByNliAsync(string text, WordPieceTokenizer tokenizer, InferenceSession session, SpeakerRole role, int k = 3, double threshold = 0.25)
        {
            string originalText = text;
            var emotionLabels = GetEmotionLabels();
            var labelWeights = GetLabelWeights();
            var prompts = GetPrompts();

            // BERT感情モデル使用時は英語に翻訳（非同期）
            if (ModelDir.Contains("bert_sentiment"))
            {
                text = await TranslationService.Instance.TranslateAsync(text);
                DebugWriteLine($"翻訳: {originalText} -> {text}");
            }

            var scores = new Dictionary<string, double>(emotionLabels.Length);

            foreach (var label in emotionLabels)
            {
                if (!prompts.TryGetValue(label, out var promptSet))
                    promptSet = new[] { $"これは{label}の感情を表す文章だ。" };

                var perPrompt = new List<double>(promptSet.Length);

                foreach (var hyp in promptSet)
                {
                    var (idsArr, typeArr, maskArr) = BuildPairInputs(tokenizer, text, hyp,
                        maxTotal: AffectTuning.MaxTotal, preTok: AffectTuning.PreTok, norm: AffectTuning.Norm);

                    // ONNX実行は同期のまま（Microsoft.ML.OnnxRuntimeは非同期APIを提供していない）
                    var idsT = new DenseTensor<long>(new[] { 1, idsArr.Length });
                    var typT = new DenseTensor<long>(new[] { 1, typeArr.Length });
                    var mskT = new DenseTensor<long>(new[] { 1, maskArr.Length });

                    for (int i = 0; i < idsArr.Length; i++)
                    {
                        idsT[0, i] = idsArr[i];
                        typT[0, i] = typeArr[i];
                        mskT[0, i] = maskArr[i];
                    }

                    var inputs = new List<NamedOnnxValue>{
                        NamedOnnxValue.CreateFromTensor("input_ids", idsT),
                        NamedOnnxValue.CreateFromTensor("token_type_ids", typT),
                        NamedOnnxValue.CreateFromTensor("attention_mask", mskT)
                    };

                    // CPU集約的な処理を別スレッドで実行
                    var logits = await Task.Run(() => {
                        using var res = session.Run(inputs);
                        return res.First().AsEnumerable<float>().Select(v => (double)v).ToArray();
                    });

                    var probs = SoftmaxTemp(logits, _temperature);

                    // スコア計算行差し替え（ClassifyByNli / ClassifyByNliAsync 両方）
                    double score = ModelDir.Contains("bert_sentiment")
                        ? EmotionScoring.CalculateSentimentScore(probs, label, text, role)
                        : _scoreMode switch
                        {
                            "ent_minus_neu" => probs[IDX_ENTAIL] - probs[IDX_NEUTRAL],
                            "ent_only" => probs[IDX_ENTAIL],
                            "margin" => probs[IDX_ENTAIL] - Math.Max(probs[IDX_NEUTRAL], probs[IDX_CONTRA]),
                            "log_margin" => Math.Log(probs[IDX_ENTAIL] + 1e-8) - Math.Log(Math.Max(probs[IDX_NEUTRAL], probs[IDX_CONTRA]) + 1e-8),
                            _ => probs[IDX_ENTAIL] - probs[IDX_NEUTRAL]
                        };

                    perPrompt.Add(score);
                }

                double agg = (_aggregate == "mean") ? perPrompt.Average() : perPrompt.Max();
                double weighted = agg * (labelWeights.TryGetValue(label, out var w) ? w : 1.0);

                // ＊ ロール別の重み調整を適用 ＊
                var roleWeights = AffectTuning.GetRoleBasedWeights(role);
                if (roleWeights.TryGetValue(label, out var roleWeight))
                {
                    weighted *= roleWeight;
                    DebugWriteLine($"ロール調整 [{role}] {label}: {agg:F3} -> {weighted:F3} (×{roleWeight})");
                }

                scores[label] = weighted;
            }

            // ★ キャリブレーション適用（正規化付き）
            if (_calibrationStore.IsEnabled)
            {
                scores = _calibrationStore.CalibrateAllCanonical(scores);
                if (_debugMode) DebugWriteLine("[CALIB] スコアキャリブレーション適用済み（正規化）");
            }

            // 計算量が少ないので残りの処理は同期のまま続ける
            var topk = FilterEmotions(scores, k); // 新しいフィルタリング処理を使用
            // 驚き判定をConfig化＋差分条件追加
            var surpriseLabel = GetLocalizedSurpriseLabel();
            scores.TryGetValue(surpriseLabel, out var sur);

            double cfgThreshold = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseThreshold", threshold);
            double cfgDelta = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseDelta", 0.10);
            // ★ 追加: 相対比 / 2位差分の個別閾値（既存Configキー活用）
            double cfgRelativeMin = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseRelativeMin", 0.60);
            double cfgSecondDiffMin = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseSecondDiffMin", 0.10);

            bool surpriseInTopK = topk.Any(kv => kv.Key == surpriseLabel);
            double topScore = topk.Count > 0 ? topk[0].Value : 0.0;
            double secondScore = topk.Count > 1 ? topk[1].Value : double.NegativeInfinity;
            double surpriseRelativeScore = topScore > 0 ? (sur / topScore) : 0.0;

            // ★ AND化: すべての条件を満たす場合のみ驚き
            bool isSurprised =
                surpriseInTopK &&
                sur >= cfgThreshold &&
                surpriseRelativeScore >= cfgRelativeMin &&
                (sur - secondScore) >= cfgSecondDiffMin;

            return new ClassificationResult
            {
                Text = text,
                OriginalText = originalText,
                Scores = scores,
                TopK = topk,
                SurpriseScore = sur,
                IsSurprised = isSurprised,
                Threshold = cfgThreshold,
                Language = _language,
                Role = role
            };
        }

        /// <summary>
        /// Retrieves a localized label for the word "surprise" based on the current language setting.
        /// </summary>
        /// <returns>A string containing the localized label for "surprise". Returns "surprise" if the language is set to English
        /// ("en"), or "驚き" if the language is set to Japanese ("jp").
        /// JP:  enなら"surprise", jpなら"驚き"を返す
        /// </returns>
        private string GetLocalizedSurpriseLabel()
        {
            if (_language == "en")
                return "surprise";
            return "驚き";

        }

        // シングルトンを使用し、Load() で係数を読み込む
        private static void RequireFile(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"必要ファイルが見つかりません: {path}");
        }

        // BuildPairInputsメソッドをstaticに修正
        private static (long[] ids, long[] typeIds, long[] mask) BuildPairInputs(
            WordPieceTokenizer tokenizer, string a, string b,
            int maxTotal = 256, bool preTok = true, bool norm = true)
        {
            // 1) それぞれを "素のID列" に
            var aIds = tokenizer.EncodeToIds(a, preTok, norm).ToList();
            var bIds = tokenizer.EncodeToIds(b, preTok, norm).ToList();

            // 2) 長さ調整（[CLS] と [SEP]×2 の3トークンを確保）
            int room = Math.Max(0, maxTotal - 3);
            int takeA = Math.Min(aIds.Count, room);
            int takeB = Math.Min(bIds.Count, room - takeA);
            aIds = aIds.Take(takeA).ToList();
            bIds = bIds.Take(takeB).ToList();

            // 3) [CLS] A [SEP] B [SEP]
            var ids = new List<long>(1 + aIds.Count + 1 + bIds.Count + 1);
            ids.Add(CLS);
            ids.AddRange(aIds.Select(i => (long)i));
            ids.Add(SEP);
            ids.AddRange(bIds.Select(i => (long)i));
            ids.Add(SEP);

            // 4) token_type_ids : A側(含CLS/SEP)=0, B側(含終端SEP)=1
            var typeIds = Enumerable.Repeat(0L, 1 + aIds.Count + 1)
                                    .Concat(Enumerable.Repeat(1L, bIds.Count + 1))
                                    .ToArray();

            // 5) attention_mask : すべて1（パディング無し）
            var mask = Enumerable.Repeat(1L, ids.Count).ToArray();

            return (ids.ToArray(), typeIds, mask);
        }

        private static double[] SoftmaxTemp(double[] logits, double temp)
        {
            double t = Math.Max(1e-6, temp);
            double max = logits.Max();
            var exps = logits.Select(v => Math.Exp((v - max) / t)).ToArray();
            double sum = exps.Sum();
            for (int i = 0; i < exps.Length; i++) exps[i] /= sum;
            return exps;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                // WordPieceTokenizerはIDisposableを実装していないため、Disposeは不要
                _session?.Dispose(); // InferenceSessionのみDispose
                _robertaSession?.Dispose();   // RoBERTa-MNLI セッション
                _goemoSession?.Dispose();     // ★ Phase B: GoEmotions セッション
                _disposed = true;
            }
        }

        // ＊ 不足しているメソッドを追加 ＊

        // デバッグ出力用のヘルパーメソッド
        private void DebugWriteLine(string message)
        {
            if (_debugMode)
            {
                Console.WriteLine(message);
            }
        }

        // インスタンス設定に基づいて多言語対応データを取得
        private Dictionary<string, double> GetLabelWeights() =>
            AffectTuning.LabelWeights;

        private Dictionary<string, string[]> GetPrompts() =>
            AffectTuning.GetActivePromptsDict();

        private string[] GetEmotionLabels() =>
            AffectTuning.GetActiveLabelList();

        // 既存 partial class AffectCore 内の適当な private メソッド群の末尾付近に追加
        /// <summary>
        /// 現在選択されているモデルキー（設定値）を取得します（失敗時は bert_sentiment）。
        /// </summary>
        private string GetCurrentModelKey()
        {
            try
            {
                return Config.AffectConfigManager.Get("nli_model", "Model", "bert_sentiment");
            }
            catch
            {
                return "bert_sentiment";
            }
        }

        // ＊ 同期版のClassifyByNliメソッド（ロール対応）＊
        [CodeStatus(CodeStatus.Legacy,
            Note = "BERT-sentiment synchronous inference path. Retained as fallback/research. " +
                   "Primary path is GoEmotions (GoEmo_Core.cs). / " +
                   "BERT-sentiment同期推論パス。研究・フォールバック用として保持。主系統はGoEmo_Core.cs。")]
        private ClassificationResult ClassifyByNli(string text, WordPieceTokenizer tokenizer, InferenceSession session, SpeakerRole role, int k = 3, double threshold = 0.25)
        {
            string originalText = text;
            var emotionLabels = GetEmotionLabels();
            var labelWeights = GetLabelWeights();
            var prompts = GetPrompts();

            // BERT感情モデル使用時は英語に翻訳（同期版）
            if (ModelDir.Contains("bert_sentiment"))
            {
                text = Yakumo.Affect.TranslationService.Instance.Translate(text);
                DebugWriteLine($"翻訳: {originalText} -> {text}");
            }

            var scores = new Dictionary<string, double>(emotionLabels.Length);

            foreach (var label in emotionLabels)
            {
                if (!prompts.TryGetValue(label, out var promptSet))
                    promptSet = new[] { $"これは{label}の感情を表す文章だ。" };

                var perPrompt = new List<double>(promptSet.Length);

                foreach (var hyp in promptSet)
                {
                    var (idsArr, typeArr, maskArr) = BuildPairInputs(tokenizer, text, hyp,
                        maxTotal: AffectTuning.MaxTotal, preTok: AffectTuning.PreTok, norm: AffectTuning.Norm);

                    // ONNX実行（同期版）
                    var idsT = new DenseTensor<long>(new[] { 1, idsArr.Length });
                    var typT = new DenseTensor<long>(new[] { 1, typeArr.Length });
                    var mskT = new DenseTensor<long>(new[] { 1, maskArr.Length });

                    for (int i = 0; i < idsArr.Length; i++)
                    {
                        idsT[0, i] = idsArr[i];
                        typT[0, i] = typeArr[i];
                        mskT[0, i] = maskArr[i];
                    }

                    var inputs = new List<NamedOnnxValue>{
                        NamedOnnxValue.CreateFromTensor("input_ids", idsT),
                        NamedOnnxValue.CreateFromTensor("token_type_ids", typT),
                        NamedOnnxValue.CreateFromTensor("attention_mask", mskT)
                    };

                    // 同期実行
                    using var res = session.Run(inputs);
                    var logits = res.First().AsEnumerable<float>().Select(v => (double)v).ToArray();

                    var probs = SoftmaxTemp(logits, _temperature);

                    // スコア計算行差し替え（ClassifyByNli / ClassifyByNliAsync 両方）
                    double score = ModelDir.Contains("bert_sentiment")
                        ? EmotionScoring.CalculateSentimentScore(probs, label, text, role)
                        : _scoreMode switch
                        {
                            "ent_minus_neu" => probs[IDX_ENTAIL] - probs[IDX_NEUTRAL],
                            "ent_only" => probs[IDX_ENTAIL],
                            "margin" => probs[IDX_ENTAIL] - Math.Max(probs[IDX_NEUTRAL], probs[IDX_CONTRA]),
                            "log_margin" => Math.Log(probs[IDX_ENTAIL] + 1e-8) - Math.Log(Math.Max(probs[IDX_NEUTRAL], probs[IDX_CONTRA]) + 1e-8),
                            _ => probs[IDX_ENTAIL] - probs[IDX_NEUTRAL]
                        };

                    perPrompt.Add(score);
                }

                double agg = (_aggregate == "mean") ? perPrompt.Average() : perPrompt.Max();
                double weighted = agg * (labelWeights.TryGetValue(label, out var w) ? w : 1.0);

                // ＊ ロール別の重み調整を適用 ＊
                var roleWeights = AffectTuning.GetRoleBasedWeights(role);
                if (roleWeights.TryGetValue(label, out var roleWeight))
                {
                    weighted *= roleWeight;
                    DebugWriteLine($"ロール調整 [{role}] {label}: {agg:F3} -> {weighted:F3} (×{roleWeight})");
                }

                scores[label] = weighted;
            }

            // ★ キャリブレーション適用（正規化付き）
            if (_calibrationStore.IsEnabled)
            {
                scores = _calibrationStore.CalibrateAllCanonical(scores);
                if (_debugMode) DebugWriteLine("[CALIB] スコアキャリブレーション適用済み（正規化）");
            }

            // 計算量が少ないので残りの処理は同期のまま続ける
            var topk = FilterEmotions(scores, k); // 新しいフィルタリング処理を使用
            // 驚き判定をConfig化＋差分条件追加
            var surpriseLabel = GetLocalizedSurpriseLabel();
            scores.TryGetValue(surpriseLabel, out var sur);

            double cfgThreshold = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseThreshold", threshold);
            double cfgDelta = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseDelta", 0.10);
            // ★ 追加: 相対比 / 2位差分の個別閾値（既存Configキー活用）
            double cfgRelativeMin = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseRelativeMin", 0.60);
            double cfgSecondDiffMin = Config.AffectConfigManager.GetDouble("nli_model", "SurpriseSecondDiffMin", 0.10);

            bool surpriseInTopK = topk.Any(kv => kv.Key == surpriseLabel);
            double topScore = topk.Count > 0 ? topk[0].Value : 0.0;
            double secondScore = topk.Count > 1 ? topk[1].Value : double.NegativeInfinity;
            double surpriseRelativeScore = topScore > 0 ? (sur / topScore) : 0.0;

            // ★ AND化: すべての条件を満たす場合のみ驚き
            bool isSurprised =
                surpriseInTopK &&
                sur >= cfgThreshold &&
                surpriseRelativeScore >= cfgRelativeMin &&
                (sur - secondScore) >= cfgSecondDiffMin;


            return new ClassificationResult
            {
                Text = text,
                OriginalText = originalText,
                Scores = scores,
                TopK = topk,
                SurpriseScore = sur,
                IsSurprised = isSurprised,
                Threshold = cfgThreshold,
                Language = _language,
                Role = role
            };
        }

        /// <summary>
        /// プラグインまたはフォールバックで読み込まれた感情フィルタ
        /// </summary>
        private static readonly IEmotionFilter _emotionFilter = EmotionFilterLoader.Load();

        /// <summary>
        /// 感情の整合性を保つためのフィルタリング処理（プラグイン委譲）
        /// </summary>
        private List<KeyValuePair<string, double>> FilterEmotions(Dictionary<string, double> scores, int k)
        {
            // DebugWriteLine は内部で _debugMode を見て出力するため、そのまま渡してOK
            return _emotionFilter.Filter(scores, k, _language, msg => DebugWriteLine(msg));
        }
    }

    // 翻訳用のヘルパークラスを追加
    [CodeStatus(CodeStatus.Deprecated,
        Note = "Fully superseded by TranslationService (NLLB-200 Flask server). " +
               "Spawns a Python subprocess per call — do not use in new code. " +
               "Retained only for historical reference. / " +
               "TranslationService（NLLB-200 Flaskサーバー）に完全置換済み。" +
               "呼び出しごとにPythonサブプロセスを起動するため新規コードでは使用禁止。" +
               "経緯参照のためのみ残存。")]
    [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
    public static class PythonTranslator
    {
        private static readonly string PythonScriptPath = Path.Combine(AppContext.BaseDirectory, "translate_ja_en.py");

        static PythonTranslator()
        {
            // 初回実行時にPythonスクリプトを作成
            CreateTranslateScript();
        }

        public static string Translate(string japaneseText)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{PythonScriptPath}\" \"{japaneseText.Replace("\"", "\\\"")}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,  // ← これでプロンプトが表示されない
                    StandardOutputEncoding = System.Text.Encoding.UTF8
                };

                using var process = Process.Start(psi);
                string result = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    if (AffectTuning.DebugMode)
                        Console.WriteLine($"翻訳エラー: {error}");
                    return japaneseText; // エラー時は元のテキストを返す
                }

                return result.Trim();
            }
            catch (Exception ex)
            {
                if (AffectTuning.DebugMode)
                    Console.WriteLine($"翻訳処理でエラー: {ex.Message}");
                return japaneseText; // エラー時は元のテキストを返す
            }
        }

        private static void CreateTranslateScript()
        {
            if (File.Exists(PythonScriptPath)) return;

            string script = @"#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

try:
    from transformers import pipeline
    
    # 翻訳パイプライン（初回実行時はモデルDLに時間がかかる）
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-ja-en')
    
    # コマンドライン引数から日本語テキストを取得
    japanese_text = sys.argv[1]
    
    # 翻訳実行
    result = translator(japanese_text)[0]['translation_text']
    
    # 結果を出力（C#側で受け取る）
    print(result)
    
except Exception as e:
    print(f'翻訳エラー: {e}', file=sys.stderr)
    sys.exit(1)
";
            File.WriteAllText(PythonScriptPath, script, System.Text.Encoding.UTF8);
            if (AffectTuning.DebugMode)
                Console.WriteLine($"翻訳スクリプトを作成しました: {PythonScriptPath}");
        }
    }

    /// <summary>
    /// 感情分析の結果を表すクラス
    /// </summary>
    public class ClassificationResult
    {
        public string Text { get; set; } = "";
        public string OriginalText { get; set; } = ""; // 原文用
        public Dictionary<string, double> Scores { get; set; } = new();
        public List<KeyValuePair<string, double>> TopK { get; set; } = new();
        public double SurpriseScore { get; set; }
        public bool IsSurprised { get; set; }
        public double Threshold { get; set; }
        public string Language { get; set; } = "jp"; // 分析時の言語設定

        /// <summary>
        /// ＊ 発話者の役割を追加 ＊
        /// </summary>
        public SpeakerRole Role { get; set; } = SpeakerRole.User;

        /// <summary>
        /// 最も高いスコアの感情ラベルを取得
        /// </summary>
        public string TopEmotion => TopK.Count > 0 ? TopK[0].Key : GetNeutralLabel();

        /// <summary>
        /// 最も高いスコアの値を取得
        /// </summary>
        public double TopScore => TopK.Count > 0 ? TopK[0].Value : 0.0;

        /// <summary>
        /// 現在の言語設定に基づいて中立ラベルを取得
        /// </summary>
        private string GetNeutralLabel() => Language == "en" ? "neutral" : "中立";

        /// <summary>
        /// 結果の文字列表現を取得
        /// </summary>
        public override string ToString()
        {
            var topResults = string.Join(", ", TopK.Select(t => $"{t.Key}={t.Value:0.000}"));
            var surpriseLabel = Language == "en" ? "surprise" : "驚き";
            return $"[{Role}] Text: {Text}, Top: {topResults}, {surpriseLabel}: {IsSurprised}";
        }
    }

    /// <summary>
    /// 複数のテキスト分析結果をまとめるクラス
    /// </summary>
    public class NliAnalysisResults
    {
        public List<ClassificationResult> Results { get; set; } = new();
        public DateTime AnalyzedAt { get; set; } = DateTime.Now;
        public string ModelUsed { get; set; } = "";
        public TimeSpan ProcessingTime { get; set; }

        /// <summary>
        /// 総分析数
        /// </summary>
        public int TotalCount => Results.Count;

        /// <summary>
        /// 驚き判定されたテキストの数
        /// </summary>
        public int SurpriseCount => Results.Count(r => r.IsSurprised);

        /// <summary>
        /// 最も多い感情ラベル
        /// </summary>
        public string MostCommonEmotion => Results
            .GroupBy(r => r.TopEmotion)
            .OrderByDescending(g => g.Count())
            .FirstOrDefault()?.Key ?? "中立";

        /// <summary>
        /// 分析結果のサマリーを取得
        /// </summary>
        public string GetSummary()
        {
            return $"分析数: {TotalCount}, 驚き: {SurpriseCount}件, " +
                   $"最頻感情: {MostCommonEmotion}, 処理時間: {ProcessingTime.TotalMilliseconds:0}ms";
        }
    }
}
