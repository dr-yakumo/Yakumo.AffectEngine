using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Yakumo.Affect
{
    /// <summary>
    /// 日本語テキストを英語に翻訳するためのサービスを提供します。
    /// ユーザー辞書や翻訳モデルの選択、サーバー管理、キャッシュ機能を備えています。
    /// シングルトンインスタンスとして利用できます。
    /// </summary>
    ///
    /// <remarks>このサービスは、内部で翻訳サーバー（Python/Flask）を自動的に起動し、HTTP経由で翻訳リクエストを処理します。
    /// ユーザーはフレーズ辞書や固有名詞マップ、翻訳後補正辞書を動的に追加でき、翻訳精度や一貫性を向上させることが可能です。
    /// 各翻訳のライセンスは、使用する翻訳モデルに依存します。
    /// </remarks>
    public class TranslationService : IDisposable
    {
        // 翻訳モデル設定
        private static readonly Dictionary<string, TranslationModelConfig> TranslationModels = new()
        {
            ["opus"] = new TranslationModelConfig
            {
                ModelName = "Helsinki-NLP/opus-mt-ja-en",
                Type = "standard",
                Description = "軽量・高速"
            },
            ["nllb"] = new TranslationModelConfig
            {
                ModelName = "facebook/nllb-200-distilled-600M",
                Type = "multilingual",
                SrcLang = "jpn_Jpan",
                TgtLang = "eng_Latn",
                Description = "高精度・文脈理解"
            },
            ["mt5"] = new TranslationModelConfig
            {
                ModelName = "google/mt5-small",
                Type = "t5",
                Description = "バランス型"
            }
        };

        private static TranslationService? _instance;
        private static readonly object _lock = new object();

        private readonly HttpClient _httpClient = new HttpClient();
        private Process? _serverProcess;
        private bool _serverStarted = false;

        // ── ブルーグリーン切替用のポート管理 ──
        // アクティブサーバーのメモリが閾値に達したら、裏でもう一方のポートに新プロセスを
        // 起動・ウォームアップし、準備完了後に無停止で切り替える(会話を止めない)。
        private readonly int[] _ports = new int[2];
        private volatile int _activePort;
        private string TranslateUrl(int port) => $"http://localhost:{port}/translate";
        private string HealthUrl(int port) => $"http://localhost:{port}/health";
        private int OtherPort(int port) => port == _ports[0] ? _ports[1] : _ports[0];

        // タイムアウト/リトライ設定（Config反映）
        private readonly int _timeoutSeconds;
        private readonly int _warmupTimeoutSeconds;
        private readonly int _retryCount;
        private readonly int _retryBackoffMs;

        private readonly Dictionary<string, string> _translationCache = new();
        private readonly TranslationModelConfig _activeModel;

        // 起動完了シグナル（初回呼び出しゲート用。サーバー再起動時に差し替えるため readonly ではない）
        private TaskCompletionSource<bool> _readyTcs
            = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private TaskCompletionSource<bool> _warmupTcs
            = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private volatile bool _isReady;

        // サーバーのライフサイクル操作(クラッシュ復旧の再起動 / ブルーグリーン切替)を
        // 同時に複数走らせないための直列化ロック
        private readonly SemaphoreSlim _serverLifecycleLock = new(1, 1);
        private readonly CancellationTokenSource _lifecycleCts = new();
        private volatile bool _disposed;

        // ── ユーザー辞書（インスタンスフィールド） ──────────────────────────
        // フレーズ辞書（完全一致 → 翻訳スキップ）
        private readonly Dictionary<string, string> _phraseDict = new();
        // 固有名詞マップ（翻訳前の部分置換）
        private readonly Dictionary<string, string> _properMap = new();
        // 英語後処理補正（翻訳後の部分置換）
        private readonly Dictionary<string, string> _corrections = new();

        /// <summary>辞書の読み書きを保護するロック</summary>
        private readonly ReaderWriterLockSlim _dictLock = new(LockRecursionPolicy.NoRecursion);

        // ── シングルトン ─────────────────────────────────────────────────────
        public static TranslationService Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_lock)
                    {
                        // nullなら翻訳サービスを初期化（スレッドセーフなダブルチェックロック）
                        _instance ??= new TranslationService();
                    }
                }
                return _instance;
            }
        }

        // ── コンストラクタ ───────────────────────────────────────────────────
        private TranslationService()
        {
            _activeModel = LoadModelFromConfig();
            Console.WriteLine("=== 翻訳サービス初期化 ===");
            Console.WriteLine($"翻訳モデル: {_activeModel.ModelName}");
            Console.WriteLine($"モデル種別: {_activeModel.Type}");
            Console.WriteLine($"特徴: {_activeModel.Description}");
            if (!string.IsNullOrEmpty(_activeModel.SrcLang))
                Console.WriteLine($"言語ペア: {_activeModel.SrcLang} → {_activeModel.TgtLang}");
            Console.WriteLine("========================");

            _timeoutSeconds       = Config.AffectConfigManager.GetInt("nli_translation", "TimeoutSeconds",       180);
            _warmupTimeoutSeconds = Config.AffectConfigManager.GetInt("nli_translation", "WarmupTimeoutSeconds",  30);
            _retryCount           = Config.AffectConfigManager.GetInt("nli_translation", "RetryCount",             1);
            _retryBackoffMs       = Config.AffectConfigManager.GetInt("nli_translation", "RetryBackoffMs",       500);

            int basePort = Config.AffectConfigManager.GetInt("nli_translation", "BasePort", 5000);
            _ports[0] = basePort;
            _ports[1] = basePort + 1;
            _activePort = _ports[0];

            _httpClient.Timeout = TimeSpan.FromSeconds(Math.Max(1, _timeoutSeconds));

            // デフォルト辞書ファイルの自動ロード（存在しない場合はスキップ）
            LoadDictionaryFromJson(Path.Combine(AppContext.BaseDirectory, "affect.dict.json"));

            AppDomain.CurrentDomain.ProcessExit += (_, __) => SafeShutdown();

            _serverProcess = LaunchServerProcess(_activePort);
            _serverStarted = _serverProcess != null;

            _ = Task.Run(async () =>
            {
                bool ready = _serverProcess != null &&
                             await WaitForServerReadyAsync(_activePort, TimeSpan.FromSeconds(Math.Max(_warmupTimeoutSeconds, 15)));
                if (ready)
                {
                    await WarmupAsync(_activePort);
                    _isReady = true;
                }
                else
                {
                    // 初回起動時はモデルダウンロードでウォームアップ制限時間を超えることがあるが、
                    // 翻訳リクエスト時に再試行されるため機能上の問題はない。文言も警告に留める
                    Console.WriteLine("[Warmup] WARNING: Server is still starting (first launch may download the model). Will retry on request / サーバー起動待ちタイムアウト（初回はモデルダウンロード中の可能性）。翻訳リクエスト時に自動再試行します");
                }
                _readyTcs.TrySetResult(ready);
                _warmupTcs.TrySetResult(ready);

                // 初回起動が完了してから、ブルーグリーン切替のためのメモリ監視を開始する
                StartMemoryMonitorLoop();
            });
        }

        // ── 辞書登録 API（Fluent） ────────────────────────────────────────────

        /// <summary>
        /// フレーズ辞書にエントリを追加します（完全一致 → 翻訳スキップ）。
        /// </summary>
        /// <param name="japanese">日本語の原文フレーズ</param>
        /// <param name="english">対応する英語訳</param>
        public TranslationService AddPhrase(string japanese, string english)
        {
            _dictLock.EnterWriteLock();
            try { _phraseDict[japanese] = english; }
            finally { _dictLock.ExitWriteLock(); }
            return this;
        }

        /// <summary>
        /// 固有名詞マップにエントリを追加します（翻訳前の部分一致置換）。
        /// </summary>
        /// <param name="japanese">日本語の固有名詞</param>
        /// <param name="english">英語表記</param>
        public TranslationService AddProperNoun(string japanese, string english)
        {
            _dictLock.EnterWriteLock();
            try { _properMap[japanese] = english; }
            finally { _dictLock.ExitWriteLock(); }
            return this;
        }

        /// <summary>
        /// 英語後処理補正にエントリを追加します（翻訳後の部分一致置換）。
        /// </summary>
        /// <param name="from">誤訳パターン（大文字小文字を無視）</param>
        /// <param name="to">正しい英語表現</param>
        public TranslationService AddCorrection(string from, string to)
        {
            _dictLock.EnterWriteLock();
            try { _corrections[from] = to; }
            finally { _dictLock.ExitWriteLock(); }
            return this;
        }

        /// <summary>
        /// JSON辞書ファイルをロードして各辞書にマージします。
        /// ファイルが存在しない場合はスキップします。
        /// </summary>
        /// <param name="filePath">
        /// <c>affect.dict.json</c> 形式のJSONファイルパス。
        /// スキーマは <see cref="UserDictionaryFile"/> を参照してください。
        /// </param>
        public TranslationService LoadDictionaryFromJson(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"[Dict] ファイルが見つかりません（スキップ）: {filePath}");
                return this;
            }

            try
            {
                var json = File.ReadAllText(filePath, Encoding.UTF8);
                var dict = JsonSerializer.Deserialize<UserDictionaryFile>(json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (dict is null) return this;

                int phraseCount = 0, nounCount = 0, corrCount = 0;

                _dictLock.EnterWriteLock();
                try
                {
                    foreach (var kv in dict.PhraseDict  ?? [])
                    {
                        // "$comment" キーはスキーマ説明用なのでスキップ
                        if (kv.Key.StartsWith("説明:", StringComparison.Ordinal) ||
                            kv.Key.StartsWith("$",    StringComparison.Ordinal)) continue;
                        _phraseDict[kv.Key] = kv.Value;
                        phraseCount++;
                    }
                    foreach (var kv in dict.ProperNouns ?? [])
                    {
                        if (kv.Key.StartsWith("説明:", StringComparison.Ordinal) ||
                            kv.Key.StartsWith("$",    StringComparison.Ordinal)) continue;
                        _properMap[kv.Key] = kv.Value;
                        nounCount++;
                    }
                    foreach (var kv in dict.Corrections ?? [])
                    {
                        if (kv.Key.StartsWith("説明:", StringComparison.Ordinal) ||
                            kv.Key.StartsWith("$",    StringComparison.Ordinal)) continue;
                        _corrections[kv.Key] = kv.Value;
                        corrCount++;
                    }
                }
                // 書き込みロックはまとめて取る（パフォーマンスのため）ので、例外が起きても必ず解放する
                finally { _dictLock.ExitWriteLock(); }

                Console.WriteLine($"[Dict] ロード完了: {filePath}");
                Console.WriteLine($"[Dict]  フレーズ={phraseCount} 固有名詞={nounCount} 補正={corrCount}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Dict] ロードエラー ({filePath}): {ex.Message}");
            }
            return this;
        }

        // ── TranslateAsync ───────────────────────────────────────────────────

        /// <summary>
        /// 指定された日本語テキストを非同期に英語に翻訳します。
        /// </summary>
        /// <remarks>
        /// フレーズ辞書（完全一致）→ 翻訳キャッシュ → 翻訳APIの順に処理します。
        /// 翻訳サーバーが起動していない場合は原文を返します。
        /// </remarks>
        /// <param name="japaneseText">翻訳する日本語テキスト</param>
        /// <param name="properNouns">
        /// 追加の固有名詞リスト。<c>"entity"</c> に置換されて翻訳精度を向上させます。
        /// </param>
        public async Task<string> TranslateAsync(string japaneseText, string[]? properNouns = null)
        {
            // フレーズ辞書を優先チェック（完全一致）
            string? phraseTranslation = null;
            _dictLock.EnterReadLock();
            try { _phraseDict.TryGetValue(japaneseText, out phraseTranslation); }
            finally { _dictLock.ExitReadLock(); }

            if (phraseTranslation != null)
            {
                Console.WriteLine($"フレーズ辞書から翻訳: {japaneseText} -> {phraseTranslation}");
                _translationCache[japaneseText] = phraseTranslation;
                return phraseTranslation;
            }

            // キャッシュ確認
            if (_translationCache.TryGetValue(japaneseText, out string? cachedTranslation))
            {
                Console.WriteLine($"キャッシュから翻訳: {japaneseText} -> {cachedTranslation}");
                return cachedTranslation;
            }

            if (!_serverStarted)
            {
                Console.WriteLine("翻訳サーバーが起動していません");
                return japaneseText;
            }

            // Warmup を待ってから通常翻訳へ
            if (!_isReady)
            {
                using var gateCts = new CancellationTokenSource(TimeSpan.FromSeconds(Math.Max(10, _warmupTimeoutSeconds)));
                await Task.WhenAny(_readyTcs.Task,  Task.Delay(Timeout.Infinite, gateCts.Token));
                await Task.WhenAny(_warmupTcs.Task, Task.Delay(Timeout.Infinite, gateCts.Token));
            }

            // 固有名詞置換（翻訳精度のための前処理）
            string preprocessed = ApplyProperNounMap(japaneseText, properNouns);

            // サーバーが(メモリ上限超過による自己終了等で)停止している場合は自動的に再起動する
            await EnsureServerAliveAsync();

            // リトライ付きでHTTP呼び出し
            int attempts = Math.Max(1, _retryCount + 1);
            int backoff   = Math.Max(100, _retryBackoffMs);

            for (int i = 1; i <= attempts; i++)
            {
                try
                {
                    // 毎回読み直す: ブルーグリーン切替がこの呼び出しの途中で起きた場合でも、
                    // リトライは常にその時点のアクティブポートへ送られるようにする
                    // (固定してしまうと、切替で退役したポートに向けて再送し続けてしまう)
                    int targetPort = _activePort;
                    using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(Math.Max(1, _timeoutSeconds)));

                    var content = new StringContent(
                        JsonSerializer.Serialize(new { text = preprocessed }),
                        Encoding.UTF8,
                        "application/json");

                    var response = await _httpClient.PostAsync(TranslateUrl(targetPort), content, cts.Token);
                    if (response.IsSuccessStatusCode)
                    {
                        var responseContent = await response.Content.ReadAsStringAsync(cts.Token);
                        var jsonResponse = JsonSerializer.Deserialize<TranslationResponse>(
                            responseContent,
                            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                        string translation = jsonResponse?.Translated ?? japaneseText;
                        translation = PostprocessTranslation(translation);

                        _translationCache[japaneseText] = translation;
                        Console.WriteLine($"翻訳API経由: {japaneseText} -> {translation}");
                        return translation;
                    }
                    else
                    {
                        Console.WriteLine($"翻訳HTTPエラー: {(int)response.StatusCode} {response.ReasonPhrase}");
                    }
                }
                catch (TaskCanceledException tcex)
                {
                    Console.WriteLine($"翻訳タイムアウト: {tcex.Message}");
                }
                catch (HttpRequestException hrex)
                {
                    Console.WriteLine($"翻訳通信エラー: {hrex.Message}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"翻訳例外: {ex.Message}");
                }

                if (i < attempts)
                {
                    Console.WriteLine($"[Retry] {i}/{attempts - 1} 失敗 -> {backoff}ms 待機して再試行");
                    await Task.Delay(backoff);
                    backoff *= 2;
                }
            }

            return japaneseText;
        }

        /// <summary>後方互換性のための同期ラッパー</summary>
        public string Translate(string japaneseText, string[]? properNouns = null)
            => TranslateAsync(japaneseText, properNouns).GetAwaiter().GetResult();

        // ── 内部処理 ─────────────────────────────────────────────────────────

        private string ApplyProperNounMap(string text, string[]? properNouns = null)
        {
            // 登録済み固有名詞マップを適用（読み取りロック）
            _dictLock.EnterReadLock();
            Dictionary<string, string> snapshot;
            try { snapshot = new Dictionary<string, string>(_properMap); }
            finally { _dictLock.ExitReadLock(); }

            foreach (var kv in snapshot)
                text = text.Replace(kv.Key, kv.Value, StringComparison.Ordinal);

            // 呼び出し元から渡された固有名詞は "entity" に置換
            if (properNouns != null)
            {
                foreach (var noun in properNouns)
                {
                    if (!string.IsNullOrWhiteSpace(noun))
                        text = text.Replace(noun, "entity", StringComparison.Ordinal);
                }
            }
            return text;
        }

        private string PostprocessTranslation(string englishText)
        {
            _dictLock.EnterReadLock();
            Dictionary<string, string> snapshot;
            try { snapshot = new Dictionary<string, string>(_corrections); }
            finally { _dictLock.ExitReadLock(); }

            string result = englishText;
            foreach (var kv in snapshot)
                result = result.Replace(kv.Key, kv.Value, StringComparison.OrdinalIgnoreCase);

            return result;
        }

        // ── サーバー管理 ─────────────────────────────────────────

        private TranslationModelConfig LoadModelFromConfig()
        {
            try
            {
                // デフォルトを opus に変更
                string modelKey = Config.AffectConfigManager.Get("nli_translation", "Model", "opus");
                string quality  = Config.AffectConfigManager.Get("nli_translation", "Quality", "high");

                if (quality == "fast" && !TranslationModels.ContainsKey(modelKey))
                    modelKey = "opus";
                else if (quality == "high" && !TranslationModels.ContainsKey(modelKey))
                    modelKey = "opus"; // デフォルトをnllb → opus に変更

                return TranslationModels.TryGetValue(modelKey, out var config)
                    ? config
                    : TranslationModels["opus"]; // フォールバックも opus に変更
            }
            catch (Exception ex)
            {
                Console.WriteLine($"設定読み込みエラー、デフォルトモデルを使用: {ex.Message}");
                return TranslationModels["opus"]; // フォールバックも opus に変更
            }
        }

        /// <summary>
        /// 翻訳サーバー(Pythonプロセス)を指定ポートで起動します。失敗時は null を返します。
        /// </summary>
        private Process? LaunchServerProcess(int port)
        {
            try
            {
                string scriptPath = Path.Combine(AppContext.BaseDirectory, "translate_server.py");
                CreateServerScript(scriptPath);

                // CPU推論はビーム探索の入力長に応じてアロケータが高水位マークを保持し続け、
                // OSに返さない性質がある(実測: 会話が続くほどWorkingSetが段階的に増加)。
                // ブルーグリーン切替の閾値に達する前に万一到達した場合の最終防波堤として、
                // この上限を超えたらPython側が自ら終了する(通常運用では発火しない想定)。
                int memLimitMb = Config.AffectConfigManager.GetInt("nli_translation", "ServerMemoryLimitMB", 1536);

                var proc = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "python",
                        // 引数: スクリプトパス, 親(自分)のPID(孤児化防止の生死監視用), メモリ上限MB(緊急自己終了用), ポート番号
                        Arguments = $"\"{scriptPath}\" {Environment.ProcessId} {memLimitMb} {port}",
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true
                    },
                    EnableRaisingEvents = true
                };

                // イベントハンドラーを設定して、サーバーの標準出力とエラー出力をリアルタイムでコンソールに表示
                proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine($"[PY:{port}] {e.Data}"); };
                proc.ErrorDataReceived  += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine($"[PY-ERR:{port}] {e.Data}"); };

                proc.Start();
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();

                Console.WriteLine($"翻訳サーバーを起動しました (port={port}, {_activeModel.ModelName})");

                Thread.Sleep(2000);
                return proc;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"翻訳サーバーの起動に失敗: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 翻訳サーバーが(クラッシュ等で)停止していた場合、自動的に同じポートで再起動する。
        /// 同時に複数の呼び出しが再起動を試みないよう直列化する。
        /// </summary>
        private async Task EnsureServerAliveAsync()
        {
            if (_serverProcess != null && !_serverProcess.HasExited) return;

            await _serverLifecycleLock.WaitAsync();
            try
            {
                if (_serverProcess != null && !_serverProcess.HasExited) return; // 他の呼び出しが既に再起動済み

                Console.WriteLine("[Translation] 翻訳サーバーが停止していたため再起動します");
                _isReady = false;
                _readyTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                _warmupTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

                _serverProcess = LaunchServerProcess(_activePort);
                _serverStarted = _serverProcess != null;

                bool ready = _serverProcess != null &&
                             await WaitForServerReadyAsync(_activePort, TimeSpan.FromSeconds(Math.Max(_warmupTimeoutSeconds, 15)));
                if (ready)
                {
                    await WarmupAsync(_activePort);
                    _isReady = true;
                }
                _readyTcs.TrySetResult(ready);
                _warmupTcs.TrySetResult(ready);
            }
            finally
            {
                _serverLifecycleLock.Release();
            }
        }

        /// <summary>
        /// アクティブサーバーのメモリ使用量を定期監視し、閾値超過時にブルーグリーン切替を行うループ。
        /// </summary>
        private void StartMemoryMonitorLoop()
        {
            int swapThresholdMb = Config.AffectConfigManager.GetInt("nli_translation", "BlueGreenSwapThresholdMB", 1024);
            int intervalSec     = Config.AffectConfigManager.GetInt("nli_translation", "BlueGreenCheckIntervalSeconds", 10);
            var token = _lifecycleCts.Token;

            _ = Task.Run(async () =>
            {
                while (!_disposed && !token.IsCancellationRequested)
                {
                    try
                    {
                        await Task.Delay(TimeSpan.FromSeconds(Math.Max(3, intervalSec)), token);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }

                    try
                    {
                        var proc = _serverProcess;
                        if (proc == null || proc.HasExited) continue;

                        proc.Refresh();
                        double mb = proc.WorkingSet64 / 1024.0 / 1024.0;
                        if (mb >= swapThresholdMb)
                        {
                            Console.WriteLine($"[BlueGreen] アクティブサーバー(port={_activePort})が{mb:F1}MB(閾値{swapThresholdMb}MB)に到達。裏で新プロセスを準備します");
                            await SwapToFreshServerAsync();
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[BlueGreen] 監視エラー: {ex.Message}");
                    }
                }
            }, token);
        }

        /// <summary>
        /// 現在のアクティブサーバーとは別ポートで新しい翻訳サーバーを起動・ウォームアップし、
        /// 準備が整い次第、会話を止めずに切り替える(旧プロセスは切替後に少し猶予を置いて終了)。
        /// </summary>
        private async Task SwapToFreshServerAsync()
        {
            // クラッシュ復旧の再起動処理などと競合しないよう、既に何か処理中ならこのサイクルは見送る
            if (!await _serverLifecycleLock.WaitAsync(0)) return;
            try
            {
                int oldPort = _activePort;
                int newPort = OtherPort(oldPort);
                var oldProcess = _serverProcess;

                var newProcess = LaunchServerProcess(newPort);
                if (newProcess == null)
                {
                    Console.WriteLine("[BlueGreen] 新サーバーの起動に失敗。既存サーバーを継続使用します");
                    return;
                }

                bool ready = await WaitForServerReadyAsync(newPort, TimeSpan.FromSeconds(Math.Max(_warmupTimeoutSeconds, 15)));
                if (!ready)
                {
                    Console.WriteLine("[BlueGreen] 新サーバーの準備確認に失敗。既存サーバーを継続使用し、新プロセスは破棄します");
                    KillServerProcess(newProcess);
                    return;
                }
                await WarmupAsync(newPort);

                // 切り替え: 以降の新規リクエストは新ポートへ送られる
                _activePort = newPort;
                _serverProcess = newProcess;
                _serverStarted = true;
                Console.WriteLine($"[BlueGreen] port={newPort} の新サーバーに切り替えました (旧 port={oldPort})");

                // 切替直前に発行済みのリクエストが完了する猶予を与えてから旧プロセスを終了する
                _ = Task.Run(async () =>
                {
                    try { await Task.Delay(5000, _lifecycleCts.Token); } catch (OperationCanceledException) { }
                    KillServerProcess(oldProcess);
                    Console.WriteLine($"[BlueGreen] 旧サーバー(port={oldPort})を終了しました");
                });
            }
            finally
            {
                _serverLifecycleLock.Release();
            }
        }

        /// <summary>プロセスを穏当に、ダメなら強制的に終了させます。</summary>
        private static void KillServerProcess(Process? proc)
        {
            try
            {
                if (proc != null && !proc.HasExited)
                {
                    proc.CloseMainWindow();
                    if (!proc.WaitForExit(1500))
                    {
                        proc.Kill(entireProcessTree: true);
                        proc.WaitForExit(1500);
                    }
                }
            }
            catch { /* 終了処理中の例外は握りつぶす */ }
            finally
            {
                try { proc?.Dispose(); } catch { }
            }
        }

        private void CreateServerScript(string path)
        {
            string script = $@"#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import ctypes
import threading
from flask import Flask, request, jsonify
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

app = Flask(__name__)

translator = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()

# ── 親プロセス監視ウォッチドッグ ──
# 親(C#側)が正常終了/クラッシュ/強制終了のいずれであっても、このPythonプロセスが
# 孤児化して動き続けることを防ぐ。親のPIDをコマンドライン引数で受け取り、
# 一定間隔で生死を確認して、消えていれば自分も終了する。
_PARENT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else None

# ── メモリ上限監視 ──
# CPU推論はビーム探索の入力長に応じてアロケータが高水位マークを保持し続け、OSに返さない
# 性質があり、対話が長く続くほどWorkingSetが段階的に増加していく。上限を超えたら自ら終了し、
# C#側(TranslationService.EnsureServerAliveAsync)が次回リクエスト時に新しいプロセスを
# 起動し直すことで、際限のない増加を防ぐ。
_MEMORY_LIMIT_MB = int(sys.argv[2]) if len(sys.argv) > 2 else None

# ── 待受ポート ──
# ブルーグリーン切替のため、C#側から明示的に割り当てられたポートで待ち受ける。
_PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ('cb', ctypes.c_ulong),
        ('PageFaultCount', ctypes.c_ulong),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
ctypes.windll.psapi.GetProcessMemoryInfo.restype = ctypes.c_int

def _own_working_set_mb():
    # argtypes/restype を明示しないと GetCurrentProcess の疑似ハンドル(64bit)が
    # ctypes既定の32bit int型で誤ってマーシャリングされ、GetProcessMemoryInfoが
    # 常に失敗する(戻り値0、WorkingSetSizeも常に0のまま)ため、必ず明示する。
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / (1024.0 * 1024.0)

def _parent_alive(pid):
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        STILL_ACTIVE = 259
        return exit_code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)

def _watchdog():
    while True:
        time.sleep(5)

        if _PARENT_PID is not None and not _parent_alive(_PARENT_PID):
            # 親が死ぬとこのプロセスの標準出力パイプの読み取り側も一緒に失われているため、
            # print自体が壊れたパイプへの書き込みで例外を出すことがある。
            # os._exit(0) は print の成否に関わらず必ず実行されるようにする。
            try:
                print('[WATCHDOG] parent process (pid=%d) is gone, shutting down' % _PARENT_PID, flush=True)
            except Exception:
                pass
            os._exit(0)

        if _MEMORY_LIMIT_MB is not None:
            try:
                ws_mb = _own_working_set_mb()
            except Exception:
                ws_mb = 0
            if ws_mb >= _MEMORY_LIMIT_MB:
                # 推論処理中(ロック取得中)ならリクエストを壊さないよう次のサイクルまで見送る
                if _infer_lock.acquire(False):
                    _infer_lock.release()
                    try:
                        print('[WATCHDOG] working set %.1fMB exceeded limit %dMB, restarting' % (ws_mb, _MEMORY_LIMIT_MB), flush=True)
                    except Exception:
                        pass
                    os._exit(0)

# ── MKLバッファの明示解放 ──
# MKL(PyTorchのCPU線形代数バックエンド)は性能優先で内部バッファを保持し続け、
# プロセス終了までOSに返さない設計になっている(Intel公式ドキュメントにも明記)。
# mkl_free_buffers()で明示的に解放を指示する。スレッドローカルな制限があるため、
# 全リクエストを同一スレッドで処理する(threaded=False)ことと組み合わせて使う。
try:
    _mkl_rt = ctypes.CDLL('mkl_rt.dll')
    _mkl_rt.mkl_free_buffers.restype = None
except Exception as _mkl_e:
    _mkl_rt = None
    print('[MKL] mkl_rt.dll not available, buffer freeing disabled:', _mkl_e, flush=True)

def _mkl_free_buffers():
    if _mkl_rt is not None:
        try:
            _mkl_rt.mkl_free_buffers()
        except Exception:
            pass

def ensure_translator():
    global translator
    if translator is not None:
        return
    with _init_lock:
        if translator is not None:
            return
        from transformers import pipeline
        {GetModelInitCode(_activeModel)}
        try:
            if translator is not None:
                _ = translator('テスト翻訳用ショートテキスト'[:16])
        except Exception as e:
            print('[INIT-WARN] Warmup failed:', e)

@app.route('/health', methods=['GET'])
def health():
    try:
        ensure_translator()
        return jsonify({{'status':'ready'}}), 200
    except Exception as e:
        return jsonify({{'status':'error','error':str(e)}}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    ensure_translator()
    data = request.get_json(silent=True) or {{}}
    text = data.get('text','')
    if not text:
        return jsonify({{'translated': ''}})

    with _infer_lock:
        try:
            try:
                {GetSafeCallCode(_activeModel)}
            except RuntimeError as re:
                print('[PY-ERR] RuntimeError first attempt:', re)
                try:
                    {GetSafeCallCode(_activeModel)}
                except Exception as e2:
                    print('[PY-ERR] Fatal translation error after retry:', e2)
                    result = text
            except Exception as e:
                print('[PY-ERR] General translation error:', e)
                result = text
        finally:
            # 成功/失敗どちらの経路でも必ずMKLバッファ解放を試みる
            _mkl_free_buffers()

    return jsonify({{'translated': result}})

if __name__ == '__main__':
    if _PARENT_PID is not None:
        threading.Thread(target=_watchdog, daemon=True).start()
    # threaded=False: 推論は_infer_lockで元々完全直列化されているため並行処理の恩恵はなく、
    # 逆にリクエスト毎に新規スレッドが立つとMKLのスレッドローカルバッファ解放が
    # 効かなくなる(mkl_free_buffersは呼び出しスレッド自身のバッファしか解放しない)。
    # 常に同一スレッドで処理することで解放を確実に効かせる。
    app.run(port=_PORT, threaded=False)
";
            File.WriteAllText(path, script, Encoding.UTF8);
        }

        private string GetSafeCallCode(TranslationModelConfig config)
        {
            if (config.Type == "multilingual")
            {
                return @"result = translator(text,
                                   num_beams=4,
                                   do_sample=False,
                                   no_repeat_ngram_size=4,
                                   encoder_no_repeat_ngram_size=4,
                                   repetition_penalty=1.30,
                                   length_penalty=1.2,
                                   early_stopping=True,
                                   max_new_tokens=64)[0]['translation_text']";
            }
            if (config.Type == "t5")
            {
                return @"prompt = 'translate Japanese to English: ' + text
result = translator(prompt,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.30,
                    length_penalty=1.2,
                    early_stopping=True,
                    max_new_tokens=64)[0]['generated_text']";
            }
            return @"result = translator(text,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.30,
                    length_penalty=1.2,
                    early_stopping=True,
                    max_new_tokens=64)[0]['translation_text']";
        }

        /// <summary>
        /// 選択された翻訳モデル種別に応じて Python 側の初期化コードを返します。
        /// </summary>
        private string GetModelInitCode(TranslationModelConfig config)
        {
            string type = string.IsNullOrWhiteSpace(config.Type) ? "standard" : config.Type;

            if (type == "multilingual")
            {
                return
@"translator = pipeline(
    'translation',
    model='" + config.ModelName + @"',
    src_lang='" + config.SrcLang + @"',
    tgt_lang='" + config.TgtLang + @"'
)";
            }

            if (type == "t5")
            {
                return
@"translator = pipeline(
    'text2text-generation',
    model='" + config.ModelName + @"'
)";
            }

            return
@"translator = pipeline(
    'translation',
    model='" + config.ModelName + @"'
)";
        }

        /// <summary>
        /// 翻訳サーバー（Flask）の /health をポーリングして、準備完了になるまで待機します。
        /// - 成功: true を返す
        /// - タイムアウト: false を返す
        /// </summary>
        /// <param name="port">対象サーバーのポート番号</param>
        /// <param name="timeout">待機の総タイムアウト時間</param>
        private async Task<bool> WaitForServerReadyAsync(int port, TimeSpan timeout)
        {
            // 期限（デッドライン）を計算
            DateTime deadlineUtc = DateTime.UtcNow + timeout;

            // ヘルスチェックURIを生成
            var healthUri = new Uri(HealthUrl(port));

            // 期限に達するまでポーリングを繰り返す
            while (DateTime.UtcNow < deadlineUtc)
            {
                try
                {
                    // 単回のヘルスチェックは短いタイムアウトで実行（5秒）
                    using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

                    // GET /health にアクセス
                    var response = await _httpClient.GetAsync(healthUri, cts.Token);

                    // 200なら準備完了
                    if (response.IsSuccessStatusCode)
                    {
                        Console.WriteLine($"[Health] 翻訳サーバー Ready (port={port})");
                        return true;
                    }

                    // 200以外は未準備として少し待って再試行
                    Console.WriteLine($"[Health] NotReady: {(int)response.StatusCode}");
                }
                catch (TaskCanceledException)
                {
                    // 単回タイムアウト(5秒)。起動直後はモデルロード中で応答が遅いため正常な待機状態
                    Console.WriteLine("[Health] WARNING: Waiting for server response / サーバーの応答を待っています");
                }
                catch (HttpRequestException)
                {
                    // ポート未オープン(接続拒否)も起動直後の正常な待機状態。短時間待って再試行
                    Console.WriteLine("[Health] WARNING: Waiting for server response / サーバーの応答を待っています");
                }
                catch (Exception ex)
                {
                    // その他のエラー
                    Console.WriteLine($"[Health] エラー: {ex.Message}");
                }

                // 500ms 待機して再試行
                await Task.Delay(500);
            }

            // 規定時間以内に Ready にならなかった
            return false;
        }

        /// <summary>
        /// コールドスタート対策のダミー翻訳（ウォームアップ）を1回行います。
        /// - 成功/失敗に関わらず致命扱いしません（ログのみ）。呼び出し元がTCS等の状態管理を行います。
        /// </summary>
        /// <param name="port">対象サーバーのポート番号</param>
        private async Task<bool> WarmupAsync(int port)
        {
            try
            {
                int warmupSeconds = Math.Max(5, _warmupTimeoutSeconds);
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(warmupSeconds));

                var payload = new StringContent(
                    JsonSerializer.Serialize(new { text = "テスト" }),
                    Encoding.UTF8,
                    "application/json");

                DateTime start = DateTime.UtcNow;
                var response = await _httpClient.PostAsync(TranslateUrl(port), payload, cts.Token);
                double elapsedMs = (DateTime.UtcNow - start).TotalMilliseconds;

                if (response.IsSuccessStatusCode)
                {
                    Console.WriteLine($"[Warmup] 成功 (port={port}, {elapsedMs:0} ms)");
                    return true;
                }
                Console.WriteLine($"[Warmup] 失敗: Status={(int)response.StatusCode} (port={port}, {elapsedMs:0} ms)");
                return false;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Warmup] 例外 (port={port}): {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 親プロセス終了/Dispose 時の安全なサーバー終了処理
        /// </summary>
        private void SafeShutdown()
        {
            lock (_lock)
            {
                try
                {
                    if (_serverProcess != null && !_serverProcess.HasExited)
                    {
                        // 可能なら穏当に閉じる → 閉じなければ Kill(true)
                        _serverProcess.CloseMainWindow();
                        if (!_serverProcess.WaitForExit(1500))
                        {
                            _serverProcess.Kill(entireProcessTree: true);
                            _serverProcess.WaitForExit(1500);
                        }
                    }
                }
                catch { /* 例外は握りつぶす（終了中） */ }
                finally
                {
                    try { _serverProcess?.Dispose(); } catch { }
                    _serverProcess = null;
                    _serverStarted = false;
                }
            }
        }

        /// <summary>
        /// Disposeパターンの実装
        /// </summary>
        public void Dispose()
        {
            try
            {
                _disposed = true;
                try { _lifecycleCts.Cancel(); } catch { }
                _lifecycleCts.Dispose();
                _httpClient?.Dispose();
                _dictLock?.Dispose();
                SafeShutdown();
                _instance = null;
            }
            catch { }
        }
    }

    /// <summary>
    /// Gets or sets the translated text resulting from a translation operation.
    /// JA: 翻訳操作の結果得られた翻訳されたテキストを表すクラス。
    /// </summary>
    internal class TranslationResponse
    {
        public string? Translated { get; set; }
    }
    /// <summary>
    /// JA: 翻訳モデルの設定を表すクラス。
    /// EN: Class representing the configuration for a translation model.
    /// </summary>
    class TranslationModelConfig
    {
        public string ModelName   { get; set; } = "";
        public string Type        { get; set; } = "";
        public string SrcLang     { get; set; } = "";
        public string TgtLang     { get; set; } = "";
        public string Description { get; set; } = "";
    }
}