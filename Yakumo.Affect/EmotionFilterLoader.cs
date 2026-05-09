using System.Reflection;

namespace Yakumo.Affect
{
    /// <summary>
    /// [EN] Dynamically loads an <see cref="IEmotionFilter"/> implementation from an external DLL.
    /// Falls back to <see cref="DefaultTopKFilter"/> if the DLL is not found or fails to load.
    /// [JA] 外部DLLから <see cref="IEmotionFilter"/> 実装を動的ロードする。
    /// DLLが見つからない・ロード失敗時は <see cref="DefaultTopKFilter"/> にフォールバックする。
    /// </summary>
    public static class EmotionFilterLoader
    {
        /// <summary>
        /// DLLファイル名（出力ディレクトリからの相対パス）
        /// </summary>
        private const string PluginDllName = "Yakumo.Affect.PolarityGate.dll";

        /// <summary>
        /// [EN] Attempts to load the Polarity Gate plugin; returns <see cref="DefaultTopKFilter"/> on failure.
        /// [JA] 極性ゲートプラグインのロードを試み、失敗時は <see cref="DefaultTopKFilter"/> を返す。
        /// </summary>
        public static IEmotionFilter Load()
        {
            try
            {
                // 実行アセンブリと同じディレクトリから探す
                var baseDir = AppContext.BaseDirectory;
                var dllPath = Path.Combine(baseDir, PluginDllName);

                if (!File.Exists(dllPath))
                {
                    Console.WriteLine($"[EmotionFilterLoader] Plugin not found: {dllPath} — using DefaultTopKFilter");
                    return new DefaultTopKFilter();
                }

                var asm = Assembly.LoadFrom(dllPath);

                // IEmotionFilter を実装する最初の具象クラスを探す
                var filterType = asm.GetTypes()
                    .FirstOrDefault(t => typeof(IEmotionFilter).IsAssignableFrom(t)
                                        && t is { IsClass: true, IsAbstract: false });

                if (filterType is null)
                {
                    Console.WriteLine($"[EmotionFilterLoader] No IEmotionFilter implementation found in {PluginDllName} — using DefaultTopKFilter");
                    return new DefaultTopKFilter();
                }

                var instance = Activator.CreateInstance(filterType) as IEmotionFilter;

                if (instance is null)
                {
                    Console.WriteLine($"[EmotionFilterLoader] Failed to instantiate {filterType.FullName} — using DefaultTopKFilter");
                    return new DefaultTopKFilter();
                }

                Console.WriteLine($"[EmotionFilterLoader] Loaded plugin: {filterType.FullName}");
                return instance;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[EmotionFilterLoader] Plugin load error: {ex.Message} — using DefaultTopKFilter");
                return new DefaultTopKFilter();
            }
        }
    }
}