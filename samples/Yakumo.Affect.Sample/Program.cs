using Yakumo.Affect;
using System.Text;

// ================================================================
// Yakumo Affect Engine — Interactive emotion-analysis sample
// Yakumo Affect Engine — 対話型感情分析サンプル
//
// Usage / 使い方:
//   dotnet run
//   dotnet run -- --lang=en
//   dotnet run -- --lang=jp --debug
//
// Options / オプション:
//   --lang=jp|en   Language mode (default: jp)
//                  言語モード (デフォルト: jp)
//   --debug        Show verbose logs and translation results
//                  詳細ログ + 翻訳結果を表示
// ================================================================

Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding  = Encoding.UTF8;

// --- Parse arguments / 引数パース ---
string lang  = "jp";
bool   debug = false;

foreach (var arg in args)
{
    if (arg.StartsWith("--lang=", StringComparison.OrdinalIgnoreCase))
        lang = arg.Split('=', 2)[1].Trim().ToLower();
    else if (arg.Equals("--debug", StringComparison.OrdinalIgnoreCase))
        debug = true;
}

// UI messages follow the selected language mode
// UI メッセージは選択した言語モードに追従します
string L(string jp, string en) => lang == "en" ? en : jp;

// --- Banner / バナー ---
Console.WriteLine("=".PadRight(60, '='));
Console.WriteLine("  Yakumo Affect Engine — Interactive Sample");
Console.WriteLine($"  {L("言語", "Language")}: {lang.ToUpper()}  {L("デバッグ", "Debug")}: {(debug ? "ON" : "OFF")}");
Console.WriteLine("=".PadRight(60, '='));
Console.WriteLine();
Console.WriteLine(L("  モデル設定は affect.config で変更できます",
                    "  Model settings can be changed in affect.config"));
Console.WriteLine(L("  翻訳モデルの初回起動は数分かかる場合があります",
                    "  The first launch may take a few minutes while the translation model loads"));
Console.WriteLine();

// --- Initialize engine / エンジン初期化 ---
Console.Write(L("エンジンを初期化中...", "Initializing engine..."));

AffectCore engine;
try
{
    engine = new AffectCore(language: lang, debugMode: debug);
}
catch (Exception ex)
{
    Console.WriteLine();
    Console.WriteLine($"[ERROR] {L("初期化に失敗しました", "Initialization failed")}: {ex.Message}");
    Console.WriteLine(L("  install.ps1 が完了しているか確認してください",
                        "  Make sure install.ps1 has completed successfully"));
    return;
}

Console.WriteLine(L(" 完了", " done"));
Console.WriteLine();
Console.WriteLine(L("テキストを入力してください。\"quit\" または \"exit\" で終了。",
                    "Type some text to analyze. Enter \"quit\" or \"exit\" to leave."));
Console.WriteLine("-".PadRight(60, '-'));

var role = SpeakerRole.User;

// --- Interactive loop / 対話ループ ---
while (true)
{
    Console.Write("> ");
    var input = Console.ReadLine();

    if (input is null
        || input.Equals("quit", StringComparison.OrdinalIgnoreCase)
        || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
        break;

    if (string.IsNullOrWhiteSpace(input))
        continue;

    try
    {
        var result = await engine.AnalyzeTextWithAutoModelAsync(input, role);

        // Translated text (debug mode, when translation happened)
        // 翻訳結果 (debug モード時、翻訳が発生した場合)
        if (debug && result.Text != result.OriginalText)
            Console.WriteLine($"  {L("翻訳", "Trans")} : {result.Text}");

        // Top emotion / Top 感情
        string surprised = result.IsSurprised ? "  ★ Surprised" : "";
        Console.WriteLine($"  Top  : {result.TopEmotion} ({result.TopScore:F3}){surprised}");

        // Top-K ranking / TopK ランキング
        var topkStr = string.Join("  ", result.TopK.Select((kv, i) => $"{i + 1}.{kv.Key}={kv.Value:F3}"));
        Console.WriteLine($"  TopK : {topkStr}");
        Console.WriteLine();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  [ERROR] {ex.Message}");
        if (debug) Console.WriteLine(ex.StackTrace);
        Console.WriteLine();
    }
}

engine.Dispose();
Console.WriteLine(L("終了しました。", "Bye."));
