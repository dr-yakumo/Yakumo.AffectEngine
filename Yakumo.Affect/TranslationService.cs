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
        // 翻訳API/ヘルスチェックのURL
        private readonly string _serverUrl = "http://localhost:5000/translate";
        private readonly string _healthUrl = "http://localhost:5000/health";

        // タイムアウト/リトライ設定（Config反映）
        private readonly int _timeoutSeconds;
        private readonly int _warmupTimeoutSeconds;
        private readonly int _retryCount;
        private readonly int _retryBackoffMs;

        private readonly Dictionary<string, string> _translationCache = new();
        private readonly TranslationModelConfig _activeModel;

        // 起動完了シグナル（初回呼び出しゲート用）
        private readonly TaskCompletionSource<bool> _readyTcs
            = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource<bool> _warmupTcs
            = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private volatile bool _isReady;

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

            _httpClient.Timeout = TimeSpan.FromSeconds(Math.Max(1, _timeoutSeconds));

            // デフォルト辞書ファイルの自動ロード（存在しない場合はスキップ）
            LoadDictionaryFromJson(Path.Combine(AppContext.BaseDirectory, "affect.dict.json"));
            
            AppDomain.CurrentDomain.ProcessExit += (_, __) => SafeShutdown();
            StartServer();

            _ = Task.Run(async () =>
            {
                bool ready = await WaitForServerReadyAsync(TimeSpan.FromSeconds(Math.Max(_warmupTimeoutSeconds, 15)));
                if (ready)
                {
                    await WarmupAsync();
                    _warmupTcs.TrySetResult(true);
                    _isReady = true;
                    _readyTcs.TrySetResult(true);
                }
                else
                {
                    _readyTcs.TrySetResult(false);
                    _warmupTcs.TrySetResult(false);
                    Console.WriteLine("[Warmup] 翻訳サーバーの準備確認に失敗（タイムアウト）");
                }
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

            // リトライ付きでHTTP呼び出し
            int attempts = Math.Max(1, _retryCount + 1);
            int backoff   = Math.Max(100, _retryBackoffMs);

            for (int i = 1; i <= attempts; i++)
            {
                try
                {
                    using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(Math.Max(1, _timeoutSeconds)));

                    var content = new StringContent(
                        JsonSerializer.Serialize(new { text = preprocessed }),
                        Encoding.UTF8,
                        "application/json");

                    var response = await _httpClient.PostAsync(_serverUrl, content, cts.Token);
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

        // ── サーバー管理（変更なし） ─────────────────────────────────────────

        private TranslationModelConfig LoadModelFromConfig()
        {
            try
            {
                string modelKey = Config.AffectConfigManager.Get("nli_translation", "Model",   "nllb");
                string quality  = Config.AffectConfigManager.Get("nli_translation", "Quality", "high");

                if (quality == "fast" && !TranslationModels.ContainsKey(modelKey))
                    modelKey = "opus";
                else if (quality == "high" && !TranslationModels.ContainsKey(modelKey))
                    modelKey = "nllb";

                return TranslationModels.TryGetValue(modelKey, out var config)
                    ? config
                    : TranslationModels["nllb"];
            }
            catch (Exception ex)
            {
                Console.WriteLine($"設定読み込みエラー、デフォルトモデルを使用: {ex.Message}");
                return TranslationModels["nllb"];
            }
        }

        private void StartServer()
        {
            try
            {
                string scriptPath = Path.Combine(AppContext.BaseDirectory, "translate_server.py");
                CreateServerScript(scriptPath);

                _serverProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"\"{scriptPath}\"",
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true
                    },
                    EnableRaisingEvents = true
                };

                // イベントハンドラーを設定して、サーバーの標準出力とエラー出力をリアルタイムでコンソールに表示
                _serverProcess.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine($"[PY] {e.Data}"); };
                _serverProcess.ErrorDataReceived  += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Console.WriteLine($"[PY-ERR] {e.Data}"); };

                _serverProcess.Start();
                _serverProcess.BeginOutputReadLine();
                _serverProcess.BeginErrorReadLine();

                Console.WriteLine($"翻訳サーバーを起動しました ({_activeModel.ModelName})");

                Thread.Sleep(2000);
                _serverStarted = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"翻訳サーバーの起動に失敗: {ex.Message}");
            }
        }

        private void CreateServerScript(string path)
        {
            string script = $@"#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import threading
from flask import Flask, request, jsonify
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

app = Flask(__name__)

translator = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()

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
            {GetSafeCallCode(_activeModel)}
        except RuntimeError as re:
            try:
                print('[PY-ERR] RuntimeError first attempt:', re)
                {GetSafeCallCode(_activeModel)}
            except Exception as e2:
                print('[PY-ERR] Fatal translation error after retry:', e2)
                return jsonify({{'translated': text}})
        except Exception as e:
            print('[PY-ERR] General translation error:', e)
            return jsonify({{'translated': text}})

    return jsonify({{'translated': result}})

if __name__ == '__main__':
    app.run(port=5000, threaded=True)
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
        /// <param name="timeout">待機の総タイムアウト時間</param>
        private async Task<bool> WaitForServerReadyAsync(TimeSpan timeout)
        {
            // 期限（デッドライン）を計算
            DateTime deadlineUtc = DateTime.UtcNow + timeout;

            // ヘルスチェックURIを生成
            var healthUri = new Uri(_healthUrl);

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
                        Console.WriteLine("[Health] 翻訳サーバー Ready");
                        return true;
                    }

                    // 200以外は未準備として少し待って再試行
                    Console.WriteLine($"[Health] NotReady: {(int)response.StatusCode}");
                }
                catch (Exception ex)
                {
                    // ポート未オープンなど起動直後は例外になりやすい。短時間待って再試行
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
        /// - 成功/失敗に関わらず致命扱いしません（ログのみ）。
        /// </summary>
        private async Task WarmupAsync()
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
                var response = await _httpClient.PostAsync(_serverUrl, payload, cts.Token);
                double elapsedMs = (DateTime.UtcNow - start).TotalMilliseconds;

                if (response.IsSuccessStatusCode)
                    Console.WriteLine($"[Warmup] 成功 ({elapsedMs:0} ms)");
                else
                    Console.WriteLine($"[Warmup] 失敗: Status={(int)response.StatusCode} ({elapsedMs:0} ms)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Warmup] 例外: {ex.Message}");
            }
            finally
            {
                // 例外/失敗でもウォームアップ完了シグナルは返す（先へ進む）
                _warmupTcs.TrySetResult(true);
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