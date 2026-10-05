#Requires -Version 5.1
<#
.SYNOPSIS
    Yakumo Affect Engine — Installer / インストーラ

.DESCRIPTION
    Checks prerequisites, selects the translation model, installs Python packages,
    downloads/exports the ONNX models, and sets up the configuration files.
    前提条件の確認、翻訳モデルの選択、Python パッケージのインストール、
    ONNX モデルのダウンロード/エクスポート、設定ファイルのセットアップを行います。

.PARAMETER SkipEmbedding
    Skip the all-MiniLM-L6-v2 (embedding model) setup.
    all-MiniLM-L6-v2 (Embedding モデル) のセットアップをスキップします。

.PARAMETER SkipVerify
    Skip the verification step.
    動作検証をスキップします。

.PARAMETER Gpu
    Also generate the fp16 model for DirectML (GPU) use (Windows only).
    DirectML (GPU) 用に fp16 モデルも生成します (Windows 専用)。

.PARAMETER Translation
    Select the translation model non-interactively (opus | nllb | mt5).
    翻訳モデルを非対話で指定します (opus | nllb | mt5)。

.PARAMETER Lang
    Installer display language (en | jp). Default: auto-detect from the OS,
    with an interactive prompt.
    インストーラの表示言語 (en | jp)。省略時は OS 言語から自動判定し、
    対話プロンプトで確認します。

.PARAMETER InstallDir
    Target installation directory (default: same directory as this script).
    インストール先ディレクトリを指定します (デフォルト: スクリプトと同じディレクトリ)。

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Translation opus -SkipEmbedding
    .\install.ps1 -Gpu -Lang en
    .\install.ps1 -Translation nllb -SkipVerify -InstallDir "C:\Yakumo"
#>

[CmdletBinding()]
param(
    [switch]$SkipEmbedding,
    [switch]$SkipVerify,
    [switch]$Gpu,
    [ValidateSet("opus", "nllb", "mt5", "")]
    [string]$Translation = "",
    [ValidateSet("en", "jp", "")]
    [string]$Lang = "",
    [string]$InstallDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ================================================================
# 初期化 / Initialization
# ================================================================
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($InstallDir -eq "") {
    $InstallDir = $SCRIPT_DIR
}

$MODELS_DIR       = Join-Path $InstallDir "libs\models"
$GOEMO_MODEL_DIR  = Join-Path $MODELS_DIR "goemo-roberta-base"
$MINILM_MODEL_DIR = Join-Path $MODELS_DIR "all-MiniLM-L6-v2"

# 検証済みモデルリビジョン / Verified model revisions
#
# 公表スコアを測定したときの HuggingFace のコミット。上流で main が更新されても、
# 測定時と同じ重みがインストールされる。
# 値を変えるとモデルの重みが変わり、スコアも公表値と一致しなくなる。
# 翻訳モデル側の固定値は TranslationService.cs の VerifiedRevisions にある。
$GOEMO_REVISION   = "58b6c5b44a7a12093f782442969019c7e2982299"
$CONFIG_SRC       = Join-Path $InstallDir "affect.config.default"
$CONFIG_DEST      = Join-Path $InstallDir "affect.config"
# リリース時は make_release.ps1 が -Version の値でこの行を書き換える。
# ここの値はリポジトリから直接実行した場合のフォールバック。
# 行の形式を変えると make_release.ps1 の置換が失敗して停止するので注意。
$VERSION          = "1.2"

# ================================================================
# 言語選択 / Language selection
# ================================================================
$UiLang = $Lang
if ($UiLang -eq "") {
    $langDefault = if ((Get-Culture).TwoLetterISOLanguageName -eq "ja") { "jp" } else { "en" }
    Write-Host ""
    Write-Host "  Select language / 言語を選択してください:" -ForegroundColor White
    Write-Host "    [1] English"
    Write-Host "    [2] 日本語 (Japanese)"
    $langChoice = Read-Host "  Choice [1/2] (Enter = $langDefault)"
    switch ($langChoice) {
        "1"     { $UiLang = "en" }
        "2"     { $UiLang = "jp" }
        default { $UiLang = $langDefault }
    }
}

# [EN] Returns the string matching the selected UI language.
# [JP] 選択された表示言語に対応する文字列を返します。
function L {
    param([string]$Jp, [string]$En)
    if ($UiLang -eq "en") { $En } else { $Jp }
}

# [EN] PS 5.1's Write-Progress (used by Invoke-WebRequest etc.) corrupts
#      full-width (CJK) console text while redrawing the progress bar.
#      Suppress progress bars in Japanese UI mode; English output is
#      ASCII-only, so the bar is kept there.
# [JP] PS5.1 の Write-Progress (Invoke-WebRequest 等が内部使用) はプログレス
#      バー再描画時に全角文字の表示を壊す (文字が二重に見える) ため、
#      日本語表示モードでは全体的に抑止する。英語モードは半角のみなので
#      影響が無く、プログレスバーを残す。
if ($UiLang -eq "jp") {
    $ProgressPreference = 'SilentlyContinue'
}

# ================================================================
# ユーティリティ関数 / Utility functions
# ================================================================
function Write-Header {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  Yakumo Affect Engine -- Installer v$VERSION" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Num, [string]$Title)
    Write-Host ""
    Write-Host "[Step $Num] $Title" -ForegroundColor Yellow
    Write-Host ("-" * 56) -ForegroundColor DarkGray
}

function Write-OK   { param([string]$Msg); Write-Host "  OK  $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg); Write-Host "  !!  $Msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$Msg); Write-Host "  NG  $Msg" -ForegroundColor Red }
function Write-Info { param([string]$Msg); Write-Host "      $Msg" -ForegroundColor White }

# INI の指定セクション内の指定キーの値を更新する
# Updates the value of a key inside a specific INI section
function Set-IniValue {
    param(
        [string]$FilePath,
        [string]$Section,
        [string]$Key,
        [string]$Value
    )
    $lines = Get-Content $FilePath -Encoding UTF8
    $inSection = $false
    $replaced  = $false
    $result    = @()
    foreach ($line in $lines) {
        if ($line -match '^\[(.+)\]') {
            $inSection = ($Matches[1].Trim() -eq $Section)
        }
        if ($inSection -and -not $replaced -and $line -match "^$([regex]::Escape($Key))\s*=") {
            $line = "$Key = $Value"
            $replaced = $true
        }
        $result += $line
    }
    Set-Content -Path $FilePath -Value $result -Encoding UTF8
    return $replaced
}

# ----------------------------------------------------------------
# 利用者ファイルのバックアップ / Backing up the user's files
# ----------------------------------------------------------------
# affect.config と affect.dict.json は、利用者が自分の環境に合わせて調整したもの。
# インストーラーが手を入れる前に、必ず backups\<日時>-<乱数5桁>\ へ複製する。
# フォルダは1回の実行につき1つで、最初に必要になった時点で作る。
#
# affect.config / affect.dict.json hold the user's own tuning. Before the installer
# touches them, they are always copied to backups\<timestamp>-<5 random digits>\.
# One folder per run, created the first time it is needed.
$script:BackupDir = $null

function Get-BackupDir {
    if ($null -ne $script:BackupDir) {
        return $script:BackupDir
    }

    $backupRoot = Join-Path $InstallDir "backups"
    if (-not (Test-Path $backupRoot)) {
        New-Item -ItemType Directory -Path $backupRoot | Out-Null
    }

    # 衝突するのは同じ秒に同じ乱数が出たときだけだが、既にあれば引き直す
    # A clash needs the same second and the same random number; retry if it happens
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        $stamp     = Get-Date -Format "yyyyMMdd-HHmmss"
        $suffix    = Get-Random -Minimum 10000 -Maximum 100000
        $candidate = Join-Path $backupRoot "$stamp-$suffix"
        if (-not (Test-Path $candidate)) {
            New-Item -ItemType Directory -Path $candidate | Out-Null
            $script:BackupDir = $candidate
            return $candidate
        }
    }

    throw (L "バックアップフォルダを作成できませんでした: $backupRoot" `
             "Could not create a backup folder under: $backupRoot")
}

function Backup-UserFile {
    param([string]$FilePath)
    $backupDir = Get-BackupDir
    $fileName  = Split-Path $FilePath -Leaf
    $dest      = Join-Path $backupDir $fileName
    Copy-Item $FilePath $dest -Force
    Write-OK (L "$fileName をバックアップしました: $dest" "Backed up $fileName to: $dest")
}

# ================================================================
# Step 1: 前提条件チェック / Prerequisites
# ================================================================
Write-Header
Write-Step "1" (L "前提条件チェック" "Checking prerequisites")

# .NET 8.0 Runtime
Write-Info (L ".NET 8.0 Runtime を確認中..." "Checking .NET 8.0 Runtime...")
try {
    $runtimes = & dotnet --list-runtimes 2>&1
    if ($runtimes | Where-Object { $_ -match "Microsoft\.NETCore\.App 8\." }) {
        Write-OK (L ".NET 8.0 Runtime が見つかりました" ".NET 8.0 Runtime found")
    } else {
        Write-Fail (L ".NET 8.0 Runtime が見つかりません" ".NET 8.0 Runtime not found")
        Write-Info (L "インストール先: https://dotnet.microsoft.com/download/dotnet/8.0" `
                      "Download from: https://dotnet.microsoft.com/download/dotnet/8.0")
        exit 1
    }
} catch {
    Write-Fail (L "dotnet コマンドが見つかりません。.NET 8.0 SDK/Runtime をインストールしてください" `
                  "The dotnet command was not found. Please install the .NET 8.0 SDK/Runtime")
    Write-Info (L "インストール先: https://dotnet.microsoft.com/download/dotnet/8.0" `
                  "Download from: https://dotnet.microsoft.com/download/dotnet/8.0")
    Write-Warn (L "インストール直後の場合は PC を再起動してから再実行してください" `
                  "If you have just installed it, restart your PC and run this installer again")
    exit 1
}

# Python 3.9+
Write-Info (L "Python 3.9+ を確認中..." "Checking Python 3.9+...")
$pythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 9) {
                $pythonCmd = $cmd
                Write-OK (L "Python $major.$minor が見つかりました ($cmd)" "Python $major.$minor found ($cmd)")
                break
            }
        }
    } catch { }
}
if ($null -eq $pythonCmd) {
    Write-Fail (L "Python 3.9 以上が見つかりません" "Python 3.9 or later was not found")
    Write-Info (L "インストール先: https://www.python.org/downloads/" `
                  "Download from: https://www.python.org/downloads/")
    Write-Warn (L "インストール直後の場合は PC を再起動してから再実行してください" `
                  "If you have just installed it, restart your PC and run this installer again")
    exit 1
}

# Visual C++ 再頒布可能パッケージ / Visual C++ Redistributable
# (PyTorch の c10.dll / ONNX Runtime のネイティブ DLL が依存。素の Windows には入っていない)
Write-Info (L "Visual C++ 再頒布可能パッケージ (x64) を確認中..." "Checking the Visual C++ Redistributable (x64)...")
$vcOk = $false
foreach ($vcRegPath in @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)) {
    try {
        $vcKey = Get-ItemProperty -Path $vcRegPath -ErrorAction Stop
        if ($vcKey.Installed -eq 1) { $vcOk = $true; break }
    } catch { }
}
if ($vcOk) {
    Write-OK (L "Visual C++ 再頒布可能パッケージが見つかりました" "Visual C++ Redistributable found")
} else {
    Write-Warn (L "Visual C++ 再頒布可能パッケージ (x64) が見つかりません" "Visual C++ Redistributable (x64) not found")
    Write-Info (L "PyTorch / ONNX Runtime の実行に必要です" "It is required by PyTorch / ONNX Runtime")
    $vcAns = Read-Host (L "  自動でダウンロードしてインストールしますか? [Y/n]" "  Download and install it automatically? [Y/n]")
    if ($vcAns -notmatch "^[nN]") {
        $vcExe = Join-Path $env:TEMP "vc_redist.x64.exe"
        Write-Info (L "ダウンロード中: https://aka.ms/vs/17/release/vc_redist.x64.exe" `
                      "Downloading: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        # Write-Progress のプログレスバーは PS5.1 + 全角文字でコンソール表示を壊す
        # (文字が二重に見える描画バグ) うえ、Invoke-WebRequest を大幅に遅くするため抑止する
        $prevProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try {
            Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcExe -UseBasicParsing
        } finally {
            $ProgressPreference = $prevProgress
        }
        Write-Info (L "インストール中 (管理者権限の確認が表示されたら許可してください)..." `
                      "Installing (approve the elevation prompt if shown)...")
        $vcProc = Start-Process -FilePath $vcExe -ArgumentList "/install", "/quiet", "/norestart" -Wait -PassThru
        if ($vcProc.ExitCode -in @(0, 1638)) {
            Write-OK (L "Visual C++ 再頒布可能パッケージをインストールしました" "Visual C++ Redistributable installed")
        } elseif ($vcProc.ExitCode -eq 3010) {
            Write-OK (L "Visual C++ 再頒布可能パッケージをインストールしました" "Visual C++ Redistributable installed")
            Write-Warn (L "再起動が必要です。PC を再起動してからインストーラーを再実行してください" `
                          "A restart is required. Please reboot and run this installer again")
            exit 1
        } else {
            Write-Fail (L "インストールに失敗しました (ExitCode=$($vcProc.ExitCode))" `
                          "Installation failed (ExitCode=$($vcProc.ExitCode))")
            Write-Info (L "手動でインストールしてください: https://aka.ms/vs/17/release/vc_redist.x64.exe" `
                          "Please install it manually: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            exit 1
        }
    } else {
        Write-Fail (L "Visual C++ 再頒布可能パッケージが必要です。手動でインストール後に再実行してください: https://aka.ms/vs/17/release/vc_redist.x64.exe" `
                      "The Visual C++ Redistributable is required. Install it manually and run this installer again: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        exit 1
    }
}

# インターネット接続 / Internet connectivity
Write-Info (L "インターネット接続を確認中..." "Checking internet connectivity...")
try {
    $null = Invoke-WebRequest -Uri "https://huggingface.co" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
    Write-OK (L "インターネット接続 OK (huggingface.co に到達できます)" `
                "Internet connection OK (huggingface.co is reachable)")
} catch {
    Write-Warn (L "HuggingFace への接続確認に失敗しました" "Could not reach HuggingFace")
    Write-Info (L "プロキシ環境の場合、モデルのダウンロードで問題が発生する可能性があります" `
                  "If you are behind a proxy, model downloads may fail")
}

# ================================================================
# Step 2: 翻訳モデルの選択 / Translation model selection
# ================================================================
Write-Step "2" (L "翻訳モデルの選択" "Selecting the translation model")

$selectedModel = ""

if ($Translation -ne "") {
    $selectedModel = $Translation
    Write-OK (L "翻訳モデルを引数で指定: $selectedModel" "Translation model specified via argument: $selectedModel")
} else {
    Write-Host ""
    Write-Host (L "  翻訳モデルを選択してください:" "  Please select a translation model:") -ForegroundColor White
    Write-Host ""
    Write-Host "  [1] opus  — Helsinki-NLP/opus-mt-ja-en" -ForegroundColor White
    Write-Host (L "       ライセンス : Apache-2.0  (商用可)" "       License    : Apache-2.0  (commercial use OK)") -ForegroundColor Green
    Write-Host (L "       特徴       : 軽量・高速 (約 300MB)  ← 推奨" "       Profile    : light & fast (approx. 300MB)  <- recommended") -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  [2] nllb  — facebook/nllb-200-distilled-600M" -ForegroundColor White
    Write-Host (L "       ライセンス : CC-BY-NC-4.0  (非商用のみ)" "       License    : CC-BY-NC-4.0  (NON-COMMERCIAL only)") -ForegroundColor Yellow
    Write-Host (L "       特徴       : 高精度 (約 1.2GB)" "       Profile    : high accuracy (approx. 1.2GB)") -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  [3] mt5   — google/mt5-small" -ForegroundColor DarkGray
    Write-Host (L "       ライセンス : Apache 2.0  (商用可)" "       License    : Apache 2.0  (commercial use OK)") -ForegroundColor DarkGray
    Write-Host (L "       特徴       : 実験的・動作未保証" "       Profile    : EXPERIMENTAL / UNSUPPORTED") -ForegroundColor Yellow
    Write-Host (L "                    翻訳用に調整されていないため、現状まともな翻訳を出力しません" `
                  "                    Not fine-tuned for translation; produces no usable output.") -ForegroundColor Yellow
    Write-Host (L "                    将来の差し替え用に経路のみ残しています。opus を選んでください" `
                  "                    The path is kept for future replacement. Please choose opus.") -ForegroundColor Yellow
    Write-Host ""
    Write-Host (L "  !! nllb を商用利用する場合はライセンスをご確認ください" `
                  "  !! Check the license before using nllb commercially") -ForegroundColor Yellow
    Write-Host "     https://creativecommons.org/licenses/by-nc/4.0/" -ForegroundColor DarkGray
    Write-Host ""

    do {
        $choice = Read-Host (L "  選択 [1/2/3] (Enter = 1 = opus)" "  Choice [1/2/3] (Enter = 1 = opus)")
        switch ($choice) {
            ""  { $selectedModel = "opus"; break }
            "1" { $selectedModel = "opus"; break }
            "2" { $selectedModel = "nllb"; break }
            "3" { $selectedModel = "mt5";  break }
            default { Write-Host (L "  1、2、3 のいずれかを入力してください" "  Please enter 1, 2 or 3") -ForegroundColor Red }
        }
    } while ($selectedModel -eq "")
}

Write-OK (L "翻訳モデル: $selectedModel" "Translation model: $selectedModel")

# ================================================================
# Step 3: Python 依存パッケージのインストール / Python dependencies
# ================================================================
Write-Step "3" (L "Python 依存パッケージのインストール" "Installing Python dependencies")

$reqFile = Join-Path $InstallDir "requirements.txt"
if (Test-Path $reqFile) {
    Write-Info (L "requirements.txt からインストール中..." "Installing from requirements.txt...")
    & $pythonCmd -m pip install -r $reqFile
} else {
    Write-Info (L "pip install (翻訳サーバー用パッケージ)..." "pip install (translation-server packages)...")
    # sacremoses は必須。Marian トークナイザーの正規化に使われるため、
    # 欠けると翻訳結果が変わり、公表スコアと違う挙動になる。
    # requirements.txt が無いときのフォールバックなので、内容を一致させておくこと。
    # sacremoses is required: the Marian tokenizer uses it for normalization,
    # and translation output differs without it. Keep this list in sync with requirements.txt.
    & $pythonCmd -m pip install flask transformers torch sentencepiece protobuf sacremoses
    if ($LASTEXITCODE -ne 0) {
        Write-Fail (L "pip install 失敗 (基本パッケージ)" "pip install failed (base packages)")
        exit 1
    }

    Write-Info (L "pip install (ONNX エクスポート用パッケージ)..." "pip install (ONNX export packages)...")
    & $pythonCmd -m pip install "optimum[onnxruntime]" onnx
}

if ($LASTEXITCODE -ne 0) {
    Write-Fail (L "pip install に失敗しました" "pip install failed")
    exit 1
}
Write-OK (L "Python パッケージのインストール完了" "Python packages installed")

# ================================================================
# Step 4: goemo-roberta-base のエクスポート / goemo-roberta-base export
# ================================================================
Write-Step "4" (L "goemo-roberta-base モデルのエクスポート" "Exporting the goemo-roberta-base model")

$goemoOnnx = Join-Path $GOEMO_MODEL_DIR "model.onnx"

if (Test-Path $goemoOnnx) {
    Write-Warn (L "model.onnx が既に存在します — スキップします" "model.onnx already exists -- skipping")
    Write-Info (L "再エクスポートする場合は $goemoOnnx を削除してから再実行してください" `
                  "To re-export, delete $goemoOnnx and run this installer again")
} else {
    New-Item -ItemType Directory -Force -Path $GOEMO_MODEL_DIR | Out-Null
    Write-Info (L "SamLowe/roberta-base-go_emotions を HuggingFace からエクスポート中..." `
                  "Exporting SamLowe/roberta-base-go_emotions from HuggingFace...")
    Write-Info (L "(初回は数分かかります。ダウンロードサイズ: 約 500MB)" `
                  "(The first run takes a few minutes. Download size: approx. 500MB)")

    # ── 取得と変換を分ける（リビジョンを固定するため）──
    #
    # optimum の exporter にはリビジョンを指定する手段が無い。
    #   --revision                          → 存在しない引数のためエラーになる
    #   --model-kwargs '{"revision":...}'   → 受け付けるが無視される
    # （詳細は scripts\fetch_goemo_model.py の冒頭コメント）
    #
    # そこで snapshot_download でリビジョンを固定して先に取得し、
    # そのローカルパスを exporter に渡す。
    # 上流で main が更新されても、公表スコアを測定したときと同じ重みが入る。
    #
    # Fetch first with the revision pinned, then export from that local path.
    # The optimum exporter has no way to pin a revision.
    $fetchScript = Join-Path $InstallDir "scripts\fetch_goemo_model.py"
    if (-not (Test-Path $fetchScript)) {
        Write-Fail (L "取得スクリプトが見つかりません: $fetchScript" `
                      "Fetch script not found: $fetchScript")
        exit 1
    }

    $modelPath = & $pythonCmd $fetchScript $GOEMO_REVISION | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($modelPath)) {
        Write-Fail (L "モデルの取得に失敗しました (リビジョン: $GOEMO_REVISION)" `
                      "Failed to fetch the model (revision: $GOEMO_REVISION)")
        exit 1
    }
    Write-OK (L "リビジョン固定で取得しました" "Fetched with the revision pinned")

    # optimum-cli は PATH 依存のため、python -m で直接起動する
    # optimum-cli depends on PATH, so invoke it directly via python -m
    & $pythonCmd -m optimum.exporters.onnx `
        --model "$modelPath" `
        --task text-classification `
        --framework pt `
        "$GOEMO_MODEL_DIR"

    if ($LASTEXITCODE -ne 0) {
        Write-Fail (L "ONNX エクスポートに失敗しました" "ONNX export failed")
        exit 1
    }

    # 形状確認スクリプトがあれば実行 / Run the shape-verification script if present
    $verifyScript = Join-Path $GOEMO_MODEL_DIR "verify_goemo_onnx.py"
    if (Test-Path $verifyScript) {
        Write-Info (L "ONNX モデル形状を確認中 (期待値: logits[batch, 28])..." `
                      "Verifying ONNX model shape (expected: logits[batch, 28])...")
        & $pythonCmd $verifyScript
    }

    Write-OK (L "goemo-roberta-base エクスポート完了" "goemo-roberta-base export complete")
}

# --gpu オプション: fp16 変換 / -Gpu option: fp16 conversion
if ($Gpu) {
    Write-Info (L "--Gpu オプション: fp16 モデルを生成中..." "-Gpu option: generating the fp16 model...")
    # ZIP では scripts\ に入る。従来はモデルディレクトリ直下だったので両方を見る。
    $fp16Script = Join-Path $InstallDir "scripts\export_goemo_fp16.py"
    if (-not (Test-Path $fp16Script)) {
        $fp16Script = Join-Path $GOEMO_MODEL_DIR "export_goemo_fp16.py"
    }
    if (Test-Path $fp16Script) {
        & $pythonCmd $fp16Script --model-dir $GOEMO_MODEL_DIR
        if ($LASTEXITCODE -ne 0) {
            Write-Warn (L "fp16 変換に失敗しました — fp32 で続行します" "fp16 conversion failed -- continuing with fp32")
        } else {
            Write-OK (L "fp16 モデル生成完了 (model_fp16.onnx)" "fp16 model generated (model_fp16.onnx)")
        }
    } else {
        Write-Warn (L "export_goemo_fp16.py が見つかりません — fp16 変換をスキップします" `
                      "export_goemo_fp16.py not found -- skipping fp16 conversion")
    }
}

# ================================================================
# Step 5: all-MiniLM-L6-v2 のセットアップ (任意) / all-MiniLM-L6-v2 setup (optional)
# ================================================================
Write-Step "5" (L "all-MiniLM-L6-v2 のセットアップ (Embedding / kNN 再スコアリング)" `
                  "Setting up all-MiniLM-L6-v2 (embedding / kNN rescoring)")

if ($SkipEmbedding) {
    Write-Info (L "-SkipEmbedding を指定したためスキップします" "-SkipEmbedding specified -- skipping")
    Write-Info (L "後から有効化する手順:" "To enable it later:")
    Write-Info (L "  1. scripts\make_all_MiniLM_L6_v2.py を実行" "  1. Run scripts\make_all_MiniLM_L6_v2.py")
    Write-Info (L "  2. affect.config の Rescoring.Enabled = true に変更" "  2. Set Rescoring.Enabled = true in affect.config")
} elseif (Test-Path (Join-Path $MINILM_MODEL_DIR "model.onnx")) {
    Write-Warn (L "all-MiniLM-L6-v2/model.onnx が既に存在します — スキップします" `
                  "all-MiniLM-L6-v2/model.onnx already exists -- skipping")
} else {
    # スクリプトを探す (scripts/ または Yakumo-NLI/ 配下など)
    # Locate the script (under scripts/ or Yakumo-NLI/ etc.)
    $makeScript = $null
    foreach ($candidate in @(
        (Join-Path $InstallDir "scripts\make_all_MiniLM_L6_v2.py"),
        (Join-Path $SCRIPT_DIR "scripts\make_all_MiniLM_L6_v2.py"),
        (Join-Path $SCRIPT_DIR "Yakumo-NLI\make_all_MiniLM_L6_v2.py"),
        (Join-Path $InstallDir "make_all_MiniLM_L6_v2.py")
    )) {
        if (Test-Path $candidate) { $makeScript = $candidate; break }
    }

    if ($null -eq $makeScript) {
        Write-Warn (L "make_all_MiniLM_L6_v2.py が見つかりません — Embedding セットアップをスキップします" `
                      "make_all_MiniLM_L6_v2.py not found -- skipping the embedding setup")
    } else {
        New-Item -ItemType Directory -Force -Path $MINILM_MODEL_DIR | Out-Null
        Write-Info (L "all-MiniLM-L6-v2 を HuggingFace からエクスポート中..." `
                      "Exporting all-MiniLM-L6-v2 from HuggingFace...")

        # スクリプト内の output_dir をインストール先に合わせて上書き
        # Override the script's output_dir to match the install directory
        $env:MINILM_OUTPUT_DIR = $MINILM_MODEL_DIR
        & $pythonCmd $makeScript

        if ($LASTEXITCODE -ne 0) {
            Write-Warn (L "all-MiniLM-L6-v2 のセットアップに失敗しました — Embedding 機能は無効のまま続行します" `
                          "all-MiniLM-L6-v2 setup failed -- continuing with the embedding feature disabled")
        } else {
            Write-OK (L "all-MiniLM-L6-v2 セットアップ完了" "all-MiniLM-L6-v2 setup complete")
        }
    }
}

# ================================================================
# Step 6: 設定ファイルのセットアップ / Configuration file setup
# ================================================================
Write-Step "6" (L "設定ファイルのセットアップ" "Setting up the configuration file")

if (-not (Test-Path $CONFIG_SRC)) {
    Write-Fail (L "affect.config.default が見つかりません: $CONFIG_SRC" "affect.config.default not found: $CONFIG_SRC")
    exit 1
}

$doWriteConfig = $true
$configExisted = Test-Path $CONFIG_DEST
if ($configExisted) {
    # 保持する場合も下で Model キーを書き換えるので、どちらを選んでも先に複製する
    # Back up first either way: the Model key below is rewritten even when keeping the file
    Backup-UserFile $CONFIG_DEST
    Write-Warn (L "affect.config が既に存在します（バックアップ済み）" `
                  "affect.config already exists (backed up)")
    Write-Info (L "Y : 今の設定をそのまま使います（翻訳モデルの指定だけ更新）" `
                  "Y : keep your current settings (only the translation model is updated)")
    Write-Info (L "n : v$VERSION の既定値で入れ直します（今の設定はバックアップに残ります）" `
                  "n : replace them with the v$VERSION defaults (your settings stay in the backup)")
    # 保持を既定にする。n と入力されたときだけ入れ直す（押し間違えても失わない側に倒す）
    # Keeping is the default; only an explicit "n" replaces the file
    $keepConfig = Read-Host (L "  今の設定を保持しますか? [Y/n]" "  Keep your current settings? [Y/n]")
    if ($keepConfig -notmatch "^[nN]$") {
        Write-Info (L "今の affect.config を保持します — 翻訳モデル設定のみ更新します" `
                      "Keeping your affect.config -- only the translation model setting will be updated")
        $doWriteConfig = $false
    }
}

if ($doWriteConfig) {
    Copy-Item $CONFIG_SRC $CONFIG_DEST -Force
    Write-OK (L "affect.config.default → affect.config にコピーしました" `
                "Copied affect.config.default -> affect.config")
    if ($configExisted) {
        Write-Info (L "以前の設定はバックアップフォルダに残っています" `
                      "Your previous settings are kept in the backup folder")
    }
}

# [nli_translation] セクションの Model キーのみ更新
# Update only the Model key in the [nli_translation] section
$updated = Set-IniValue -FilePath $CONFIG_DEST -Section "nli_translation" -Key "Model" -Value $selectedModel
if ($updated) {
    Write-OK (L "翻訳モデルを [nli_translation] Model = $selectedModel に設定しました" `
                "Set [nli_translation] Model = $selectedModel")
} else {
    Write-Warn (L "[nli_translation] セクションに Model キーが見つかりませんでした — 手動で確認してください" `
                  "The Model key was not found in the [nli_translation] section -- please check manually")
}

# ================================================================
# Step 6b: 翻訳辞書ファイルのセットアップ / Translation dictionary setup
# ================================================================
Write-Step "6b" (L "翻訳辞書ファイルのセットアップ" "Setting up the translation dictionary")

$dictSrc  = Join-Path $InstallDir "affect.dict.json.example"
$dictDest = Join-Path $InstallDir "affect.dict.json"

if (-not (Test-Path $dictSrc)) {
    Write-Warn (L "affect.dict.json.example が見つかりません — 辞書セットアップをスキップします" `
                  "affect.dict.json.example not found -- skipping the dictionary setup")
} elseif (Test-Path $dictDest) {
    Backup-UserFile $dictDest
    Write-Warn (L "affect.dict.json が既に存在します（バックアップ済み）" `
                  "affect.dict.json already exists (backed up)")
    Write-Info (L "Y : 今の辞書をそのまま使います" `
                  "Y : keep your current dictionary")
    Write-Info (L "n : v$VERSION の辞書で入れ直します（今の辞書はバックアップに残ります）" `
                  "n : replace it with the v$VERSION dictionary (yours stays in the backup)")
    # 保持を既定にする。n と入力されたときだけ入れ直す（押し間違えても失わない側に倒す）
    # Keeping is the default; only an explicit "n" replaces the file
    $keepDict = Read-Host (L "  今の辞書を保持しますか? [Y/n]" "  Keep your current dictionary? [Y/n]")
    if ($keepDict -match "^[nN]$") {
        Copy-Item $dictSrc $dictDest -Force
        Write-OK (L "affect.dict.json.example → affect.dict.json にコピーしました" `
                    "Copied affect.dict.json.example -> affect.dict.json")
        Write-Info (L "以前の辞書はバックアップフォルダに残っています" `
                      "Your previous dictionary is kept in the backup folder")
    } else {
        Write-Info (L "今の affect.dict.json を保持します" `
                      "Keeping your affect.dict.json")
    }
} else {
    Copy-Item $dictSrc $dictDest -Force
    Write-OK (L "affect.dict.json.example → affect.dict.json にコピーしました" `
                "Copied affect.dict.json.example -> affect.dict.json")
    Write-Info (L "affect.dict.json はユーザーが自由に編集できます（phraseDict / properNouns / corrections）" `
                  "You can freely edit affect.dict.json (phraseDict / properNouns / corrections)")
}

# ================================================================
# Step 7: PolarityGate DLL の確認 / PolarityGate DLL check
# ================================================================
Write-Step "7" (L "PolarityGate DLL の確認" "Checking the PolarityGate DLL")

$polarityDll = Join-Path $InstallDir "Yakumo.Affect.PolarityGate.dll"
if (Test-Path $polarityDll) {
    Write-OK (L "Yakumo.Affect.PolarityGate.dll が見つかりました" "Yakumo.Affect.PolarityGate.dll found")
} else {
    Write-Warn (L "Yakumo.Affect.PolarityGate.dll が見つかりません" "Yakumo.Affect.PolarityGate.dll not found")
    Write-Info (L "PolarityGate なしで動作します (DefaultTopKFilter が使用されます)" `
                  "The engine will run without PolarityGate (DefaultTopKFilter will be used)")
}

# ================================================================
# Step 8: 動作検証 / Verification
# ================================================================
Write-Step "8" (L "動作検証" "Verification")

if ($SkipVerify) {
    Write-Info (L "-SkipVerify を指定したためスキップします" "-SkipVerify specified -- skipping")
} else {
    Write-Info (L "翻訳サーバーはライブラリ初回呼び出し時に自動起動されます" `
                  "The translation server starts automatically on the library's first call")
    Write-Info (L "goemo-roberta-base モデルファイルの存在を確認中..." `
                  "Checking that the goemo-roberta-base model files exist...")

    $requiredFiles = @("model.onnx", "tokenizer.json", "vocab.json", "merges.txt", "config.json")
    $allOk = $true
    foreach ($f in $requiredFiles) {
        $path = Join-Path $GOEMO_MODEL_DIR $f
        if (Test-Path $path) {
            Write-OK "$f"
        } else {
            Write-Fail (L "$f が見つかりません" "$f not found")
            $allOk = $false
        }
    }

    if (-not $allOk) {
        Write-Warn (L "一部ファイルが不足しています。Step 4 のエクスポートを再確認してください" `
                      "Some files are missing. Please re-check the Step 4 export")
    } else {
        Write-OK (L "モデルファイルの確認完了" "Model files verified")
    }
}

# ================================================================
# 完了 / Done
# ================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host (L "  Yakumo Affect Engine のセットアップが完了しました" `
              "  Yakumo Affect Engine setup is complete") -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
# 八雲（雲）とロゴ文字。形は ASCII だけで作る（日本語コンソールでも幅がずれないように）
# Clouds over the Yakumo wordmark, ASCII only so the width is the same on any console
Write-Host '          .--.                    .--.' -ForegroundColor White
Write-Host '       .-(    )-.              .-(    )-.' -ForegroundColor White
Write-Host '      (___.__)___)            (___.__)___)' -ForegroundColor Gray
Write-Host ' __   __        _' -ForegroundColor Magenta
Write-Host ' \ \ / / __ _  | |__  _  _   _ __    ___' -ForegroundColor Magenta
Write-Host '  \ V / / _` | | / / | || | | ''  \  / _ \' -ForegroundColor Magenta
Write-Host '   |_|  \__,_| |_\_\  \_,_| |_|_|_| \___/' -ForegroundColor Magenta
Write-Host ""
# 英語 UI では「八雲」を出さない（英語版 Windows のコンソールでは漢字が表示できないことがあるため）
# The kanji is omitted in the English UI: an English console font may not have it
Write-Host "   " -NoNewline
if ($UiLang -ne "en") {
    Write-Host "八雲 " -NoNewline -ForegroundColor Cyan
}
Write-Host "affect engine " -NoNewline -ForegroundColor White
Write-Host "v$VERSION" -NoNewline -ForegroundColor DarkMagenta
Write-Host "  --  ready" -ForegroundColor White
Write-Host ""
Write-Host (L "  翻訳モデル    : $selectedModel" "  Translation model : $selectedModel") -ForegroundColor White
Write-Host (L "  設定ファイル  : $CONFIG_DEST" "  Config file       : $CONFIG_DEST") -ForegroundColor White
Write-Host (L "  モデルフォルダ: $MODELS_DIR" "  Models folder     : $MODELS_DIR") -ForegroundColor White
if ($null -ne $script:BackupDir) {
    Write-Host (L "  バックアップ  : $script:BackupDir" "  Backup folder     : $script:BackupDir") -ForegroundColor White
}
Write-Host ""
Write-Host (L "  次のステップ:" "  Next steps:") -ForegroundColor Cyan
Write-Host (L "   - サンプルコードを実行して動作確認してください" `
              "   - Run the sample app to verify everything works") -ForegroundColor White
Write-Host (L "   - GPU を使用する場合は affect.config の UseDirectML = true に変更" `
              "   - To use the GPU, set UseDirectML = true in affect.config") -ForegroundColor White
Write-Host (L "   - Embedding 再スコアリングを有効化: Rescoring.Enabled = true" `
              "   - To enable embedding rescoring: set Rescoring.Enabled = true") -ForegroundColor White
Write-Host ""
