using System;
using System.Collections.Generic;
using System.IO;

namespace Yakumo.Affect.Config
{
    /// <summary>
    /// INI形式の設定ファイルを読み込み、セクション・キーで値を取得できる静的クラス。
    /// </summary>
    public static class AffectConfigManager
    {
        private static readonly Dictionary<string, Dictionary<string, string>> _configData = new();
        private static bool _isLoaded = false;
        private static readonly object _lock = new();   // スレッドセーフなロックオブジェクト

        // どのファイルをどの順番で読み込んだか（デバッグ表示用）
        private static readonly List<string> _loadedFiles = new();
        public static IReadOnlyList<string> LoadedFiles => _loadedFiles.AsReadOnly();

        /// <summary>
        /// 既定の探索順でレイヤード読み込みする。
        /// 1) BaseDir/AIClient.Config              （下位: AIクライアント共通 ※affect.configが優先されるので無くてもよいです）
        /// 2) BaseDir/affect.config                （上位: Affect Engine 専用）
        ///    ※ affect.config が存在しない場合は NLI.Config にフォールバック（非推奨）
        /// 3) 環境変数 AFFECT_CONFIG 指定ファイル  （最上位）
        ///    ※ AFFECT_CONFIG が未設定の場合は NLI_CONFIG にフォールバック（非推奨）
        /// </summary>
        public static IReadOnlyList<string> LoadLayeredDefaults()
        {
            lock (_lock)
            {
                if (_isLoaded) return LoadedFiles;

                var files = new List<string>();
                var baseDir = AppContext.BaseDirectory;

                // 1) グローバル共通設定（最下位）
                var globalPath = Path.Combine(baseDir, "AIClient.Config");
                if (File.Exists(globalPath)) files.Add(globalPath);

                // 2) Affect Engine 専用設定（上書き）
                var affectPath = Path.Combine(baseDir, "affect.config");
                if (File.Exists(affectPath))
                {
                    files.Add(affectPath);
                }
                else
                {
                    // 旧 NLI.Config への後方互換フォールバック
                    var legacyPath = Path.Combine(baseDir, "NLI.Config");
                    if (File.Exists(legacyPath))
                    {
                        Console.WriteLine("[CONFIG][WARN] NLI.Config は非推奨です。affect.config.default を affect.config にコピーして使用してください。");
                        files.Add(legacyPath);
                    }
                }

                // 3) 環境変数（最優先）— AFFECT_CONFIG を優先、NLI_CONFIG を後方互換として維持
                var envPath = Environment.GetEnvironmentVariable("AFFECT_CONFIG")
                           ?? Environment.GetEnvironmentVariable("NLI_CONFIG");
                if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath!))
                {
                    if (Environment.GetEnvironmentVariable("AFFECT_CONFIG") is null)
                        Console.WriteLine("[CONFIG][WARN] 環境変数 NLI_CONFIG は非推奨です。AFFECT_CONFIG に移行してください。");
                    files.Add(envPath!);
                }

                LoadLayered(files);
                if (_loadedFiles.Count > 0)
                {
                    Console.WriteLine("[CONFIG] Layered load order (low → high):");
                    foreach (var f in _loadedFiles)
                        Console.WriteLine($"  - {f}");
                }
                else
                {
                    Console.WriteLine("[WARNING] 設定ファイルが見つかりません（AIClient.Config / affect.config / AFFECT_CONFIG）");
                    Console.WriteLine("[WARNING] affect.config.default を affect.config にコピーしてください。");
                }

                return LoadedFiles;
            }
        }

        /// <summary>
        /// 指定された複数ファイルを下位→上位の順にマージ読み込み
        /// </summary>
        public static void LoadLayered(IEnumerable<string> filePaths)
        {
            lock (_lock)
            {
                if (_isLoaded) return;
                _configData.Clear();
                _loadedFiles.Clear();

                foreach (var path in filePaths)
                {
                    MergeFromFile(path);
                }
                _isLoaded = true;
            }
        }

        /// <summary>
        /// 従来API：単一ファイル読み込み（後方互換）
        /// </summary>
        /// <param name="filePath">INIファイルのパス</param>
        public static void Load(string filePath)
        {
            lock (_lock)
            {
                if (_isLoaded) return;
                _configData.Clear();
                _loadedFiles.Clear();

                MergeFromFile(filePath);
                _isLoaded = true;
            }
        }

        // 実際のマージ処理
        private static void MergeFromFile(string filePath)
        {
            try
            {
                Dictionary<string, string>? currentSection = null;

                foreach (var line in File.ReadLines(filePath))
                {
                    var trimmedLine = line.Trim();
                    if (string.IsNullOrEmpty(trimmedLine) || trimmedLine.StartsWith(";") || trimmedLine.StartsWith("#"))
                        continue;

                    if (trimmedLine.StartsWith("[") && trimmedLine.EndsWith("]"))
                    {
                        var sectionName = trimmedLine[1..^1].Trim();
                        if (!_configData.TryGetValue(sectionName, out var existing))
                        {
                            existing = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                            _configData[sectionName] = existing;
                        }
                        currentSection = existing;
                        continue;
                    }

                    if (currentSection != null && trimmedLine.Contains('='))
                    {
                        var keyValue = trimmedLine.Split('=', 2);
                        var key = keyValue[0].Trim();
                        var value = keyValue[1].Trim();

                        // 行内コメントを除去
                        int commentIdx = value.IndexOf(';');
                        if (commentIdx >= 0) value = value.Substring(0, commentIdx).TrimEnd();
                        commentIdx = value.IndexOf('#');
                        if (commentIdx >= 0) value = value.Substring(0, commentIdx).TrimEnd();

                        currentSection[key] = value;
                    }
                }

                _loadedFiles.Add(filePath);
                Console.WriteLine($"[CONFIG] Loaded: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading config file '{filePath}': {ex.Message}");
            }
        }

        // まだ読み込んでいなければ既定探索で自動読み込み
        private static void EnsureLoaded()
        {
            if (_isLoaded) return;
            LoadLayeredDefaults();
        }

        /// <summary>
        /// セクション・キーで値を取得。見つからなければnull。
        /// </summary>
        public static string? Get(string section, string key)
        {
            EnsureLoaded();
            if (_configData.TryGetValue(section, out var sec) && sec.TryGetValue(key, out var value))
                return value;
            return null;
        }

        /// <summary>
        /// セクション・キーで値を取得。なければデフォルト値を返す。
        /// </summary>
        public static string Get(string section, string key, string defaultValue)
        {
            EnsureLoaded();
            var value = Get(section, key) ?? defaultValue;

            if (value == defaultValue)
            {
                Console.WriteLine($"[CONFIG] {section}.{key} = {value} (デフォルト値)");
            }
            else
            {
                Console.WriteLine($"[CONFIG] {section}.{key} = {value}");
            }

            return value;
        }

        /// <summary>
        /// セクション・キーでint値を取得。なければデフォルト値を返す。
        /// </summary>
        public static int GetInt(string section, string key, int defaultValue = 0)
        {
            EnsureLoaded();
            var str = Get(section, key);
            if (int.TryParse(str, out var v)) return v;
            return defaultValue;
        }

        /// <summary>
        /// セクション・キーで double 値を取得。なければデフォルト値を返す。
        /// </summary>
        public static double GetDouble(string section, string key, double defaultValue = 0.0)
        {
            EnsureLoaded();
            var str = Get(section, key);
            if (double.TryParse(str, out var v)) return v;
            return defaultValue;
        }
    }
}
