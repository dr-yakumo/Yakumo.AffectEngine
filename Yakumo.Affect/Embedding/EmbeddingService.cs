using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Yakumo.Affect.Embedding
{
    /// <summary>
    /// all-MiniLM-L6-v2 モデルを使用してテキストを384次元ベクトルに変換するサービス
    /// Phase 5-4 Embedding 実装 - Step 1
    /// </summary>
    public class EmbeddingService : IDisposable
    {
        private static readonly string ModelDir = Path.Combine(AppContext.BaseDirectory, "libs", "models", "all-MiniLM-L6-v2");
        private static readonly string ModelPath = Path.Combine(ModelDir, "model.onnx");
        private static readonly string TokenizerPath = Path.Combine(ModelDir, "tokenizer.json");

        /// <summary>
        /// 出力ベクトルの次元数 (all-MiniLM-L6-v2 = 384)
        /// </summary>
        public const int EmbeddingDimension = 384;

        private InferenceSession? _session;
        private BertWordPieceTokenizer? _tokenizer;  // ← 変更: BERT用トークナイザー
        private bool _initialized = false;
        private bool _disposed = false;
        private bool _useDirectML;
        private readonly int _deviceId;

        /// <summary>
        /// シングルトンインスタンス
        /// </summary>
        public static EmbeddingService Instance { get; } = new();

        public EmbeddingService()
        {
            _useDirectML = Config.AffectConfigManager.Get("embedding", "UseDirectML", "false")
                .Equals("true", StringComparison.OrdinalIgnoreCase);
            _deviceId = Config.AffectConfigManager.GetInt("embedding", "DirectMLDeviceId", 0);
        }

        /// <summary>
        /// 初期化状態を確認
        /// </summary>
        public bool IsInitialized => _initialized && !_disposed;

        /// <summary>
        /// モデルを非同期で初期化
        /// </summary>
        public async Task InitializeAsync()
        {
            if (_initialized) return;

            await Task.Run(() =>
            {
                try
                {
                    Console.WriteLine("[EMBED] all-MiniLM-L6-v2 初期化中...");

                    if (!File.Exists(ModelPath))
                    {
                        throw new FileNotFoundException(
                            $"Embedding モデルが見つかりません: {ModelPath}\n" +
                            "README の指示に従ってモデルをダウンロードしてください。");
                    }

                    if (!File.Exists(TokenizerPath))
                    {
                        throw new FileNotFoundException(
                            $"Tokenizer が見つかりません: {TokenizerPath}");
                    }

                    // BERT WordPiece Tokenizer をロード
                    _tokenizer = BertWordPieceTokenizer.Load(TokenizerPath);

                    // セッションオプションの設定
                    var opt = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                    };

                    if (_useDirectML)
                    {
                        // ★ 他の DirectML セッション（RoBERTa等）との競合を回避
                        bool otherDmlActive = IsOtherDirectMLSessionActive();
                        if (otherDmlActive)
                        {
                            Console.WriteLine("[EMBED][WARN] 他の DirectML セッションを検出 - CPU にフォールバック");
                            _useDirectML = false;
                            ConfigureCpuOptions(opt);
                        }
                        else
                        {
                            try
                            {
                                opt.AppendExecutionProvider_DML(_deviceId);
                                Console.WriteLine($"[EMBED] DirectML (GPU deviceId={_deviceId}) 有効");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[EMBED][WARN] DirectML 利用不可、CPU にフォールバック: {ex.Message}");
                                _useDirectML = false;
                                ConfigureCpuOptions(opt);
                            }
                        }
                    }
                    else
                    {
                        ConfigureCpuOptions(opt);
                    }

                    _session = new InferenceSession(ModelPath, opt);
                    _initialized = true;

                    Console.WriteLine($"[EMBED] 初期化完了 (dim={EmbeddingDimension}, DirectML={_useDirectML})");
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Embedding モデル初期化失敗: {ex.Message}", ex);
                }
            });
        }

        private static void ConfigureCpuOptions(SessionOptions opt)
        {
            opt.IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount);
            opt.InterOpNumThreads = 1;
        }

        /// <summary>
        /// 単一テキストを embedding ベクトルに変換
        /// </summary>
        public async Task<float[]> EmbedAsync(string text)
        {
            if (!_initialized)
                await InitializeAsync();

            if (_disposed)
                throw new ObjectDisposedException(nameof(EmbeddingService));

            return await Task.Run(() => EmbedInternal(text));
        }

        /// <summary>
        /// 複数テキストをバッチで embedding 変換（真のバッチ推論）
        /// </summary>
        public async Task<float[][]> EmbedBatchAsync(IList<string> texts, int batchSize = 32)
        {
            if (!_initialized)
                await InitializeAsync();

            if (_disposed)
                throw new ObjectDisposedException(nameof(EmbeddingService));

            var results = new float[texts.Count][];
            int totalBatches = (texts.Count + batchSize - 1) / batchSize;

            for (int i = 0; i < texts.Count; i += batchSize)
            {
                int currentBatch = i / batchSize + 1;
                int end = Math.Min(i + batchSize, texts.Count);
                var batch = texts.Skip(i).Take(end - i).ToList();

                // ★ 真のバッチ推論に変更
                var batchResults = await Task.Run(() => EmbedBatchInternal(batch));
                Array.Copy(batchResults, 0, results, i, batchResults.Length);

                if (currentBatch % 10 == 0 || currentBatch == totalBatches)
                {
                    Console.WriteLine($"[EMBED] バッチ {currentBatch}/{totalBatches} 完了 ({end}/{texts.Count} 件)");
                }
            }

            return results;
        }

        /// <summary>
        /// 内部: 単一テキストの embedding 生成
        /// </summary>
        private float[] EmbedInternal(string text)
        {
            if (_session == null || _tokenizer == null)
                throw new InvalidOperationException("サービスが初期化されていません");

            // トークン化 (最大128トークン - MiniLM のデフォルト)
            const int maxLength = 128;
            var inputIdArray = _tokenizer.Encode(text, maxLength);
            int seqLen = inputIdArray.Length;

            // 入力テンソルを作成
            var inputIds = new DenseTensor<long>(new[] { 1, seqLen });
            var attentionMask = new DenseTensor<long>(new[] { 1, seqLen });
            var tokenTypeIds = new DenseTensor<long>(new[] { 1, seqLen });

            for (int i = 0; i < seqLen; i++)
            {
                inputIds[0, i] = inputIdArray[i];
                attentionMask[0, i] = 1;
                tokenTypeIds[0, i] = 0;
            }

            // 推論実行
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
            };

            using var outputs = _session.Run(inputs);

            // 出力から embedding を抽出 (Mean Pooling)
            var lastHiddenState = outputs.First().AsTensor<float>();
            return MeanPooling(lastHiddenState, attentionMask);
        }

        /// <summary>
        /// 内部: バッチ推論（複数テキストを同時処理）
        /// </summary>
        private float[][] EmbedBatchInternal(List<string> texts)
        {
            if (_session == null || _tokenizer == null)
                throw new InvalidOperationException("サービスが初期化されていません");

            const int maxLength = 128;
            int batchSize = texts.Count;

            // 1. 全テキストをトークン化
            var allInputIds = new List<long[]>();
            var allAttentionMask = new List<long[]>();
            int maxSeqLen = 0;

            foreach (var text in texts)
            {
                var inputIdArray = _tokenizer.Encode(text, maxLength);
                maxSeqLen = Math.Max(maxSeqLen, inputIdArray.Length);
                allInputIds.Add(inputIdArray);
            }

            // 2. パディングしてテンソル作成
            var inputIds = new DenseTensor<long>(new[] { batchSize, maxSeqLen });
            var attentionMask = new DenseTensor<long>(new[] { batchSize, maxSeqLen });
            var tokenTypeIds = new DenseTensor<long>(new[] { batchSize, maxSeqLen });

            for (int b = 0; b < batchSize; b++)
            {
                var ids = allInputIds[b];
                for (int t = 0; t < maxSeqLen; t++)
                {
                    if (t < ids.Length)
                    {
                        inputIds[b, t] = ids[t];
                        attentionMask[b, t] = 1;
                        tokenTypeIds[b, t] = 0;
                    }
                    else
                    {
                        inputIds[b, t] = 0;  // PAD
                        attentionMask[b, t] = 0;
                        tokenTypeIds[b, t] = 0;
                    }
                }
            }

            // 3. バッチ推論実行
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
            };

            using var outputs = _session.Run(inputs);

            // 4. バッチ出力から各文の embedding を抽出
            var lastHiddenState = outputs.First().AsTensor<float>();
            var results = new float[batchSize][];

            for (int b = 0; b < batchSize; b++)
            {
                results[b] = MeanPoolingBatch(lastHiddenState, attentionMask, b);
            }

            return results;
        }

        /// <summary>
        /// Mean Pooling: 全トークンの hidden state を平均してセンテンス埋め込みを生成
        /// </summary>
        private static float[] MeanPooling(Tensor<float> hiddenState, Tensor<long> attentionMask)
        {
            var dims = hiddenState.Dimensions;
            int seqLen = dims[1];
            int hiddenDim = dims[2];

            var result = new float[hiddenDim];
            float tokenCount = 0;

            for (int t = 0; t < seqLen; t++)
            {
                if (attentionMask[0, t] == 1)
                {
                    for (int d = 0; d < hiddenDim; d++)
                    {
                        result[d] += hiddenState[0, t, d];
                    }
                    tokenCount++;
                }
            }

            if (tokenCount > 0)
            {
                for (int d = 0; d < hiddenDim; d++)
                {
                    result[d] /= tokenCount;
                }
            }

            Normalize(result);
            return result;
        }

        /// <summary>
        /// バッチ版 Mean Pooling
        /// </summary>
        private static float[] MeanPoolingBatch(Tensor<float> hiddenState, Tensor<long> attentionMask, int batchIndex)
        {
            // hiddenState: [batch, seq_len, hidden_dim]
            var dims = hiddenState.Dimensions;
            int seqLen = dims[1];
            int hiddenDim = dims[2];

            var result = new float[hiddenDim];
            float tokenCount = 0;

            for (int t = 0; t < seqLen; t++)
            {
                if (attentionMask[batchIndex, t] == 1)
                {
                    for (int d = 0; d < hiddenDim; d++)
                    {
                        result[d] += hiddenState[batchIndex, t, d];
                    }
                    tokenCount++;
                }
            }

            if (tokenCount > 0)
            {
                for (int d = 0; d < hiddenDim; d++)
                {
                    result[d] /= tokenCount;
                }
            }

            Normalize(result);
            return result;
        }

        /// <summary>
        /// L2 正規化 (in-place)
        /// </summary>
        public static void Normalize(float[] vector)
        {
            float norm = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                norm += vector[i] * vector[i];
            }
            norm = MathF.Sqrt(norm);

            if (norm > 1e-12f)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] /= norm;
                }
            }
        }

        /// <summary>
        /// コサイン類似度を計算 (正規化済みベクトル前提)
        /// </summary>
        public static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("ベクトルの次元が一致しません");

            float dot = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
            }
            return dot;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _session = null;
                _disposed = true;
            }
        }

        /// <summary>
        /// 他の DirectML セッションがアクティブかチェック
        /// （AffectCore の RoBERTa セッションとの競合回避）
        /// </summary>
        private static bool IsOtherDirectMLSessionActive()
        {
            try
            {
                return Config.AffectConfigManager.Get("nli_model", "UseDirectML", "false")
                    .Equals("true", StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }
    }
}