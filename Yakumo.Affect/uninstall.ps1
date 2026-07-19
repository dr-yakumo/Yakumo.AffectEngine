#Requires -Version 5.1
<#
.SYNOPSIS
    Yakumo Affect Engine — Uninstaller / アンインストーラ

.DESCRIPTION
    Deletes the model files and backs up the configuration file.
    Python packages are NOT removed automatically (to avoid conflicts with other tools).
    モデルファイルの削除、設定ファイルのバックアップを行います。
    Python パッケージは自動削除しません (他ツールとの競合を避けるため)。

.PARAMETER Lang
    Uninstaller display language (en | jp). Default: auto-detect from the OS,
    with an interactive prompt.
    アンインストーラの表示言語 (en | jp)。省略時は OS 言語から自動判定し、
    対話プロンプトで確認します。

.PARAMETER InstallDir
    Target directory to uninstall from (default: same directory as this script).
    アンインストール対象のディレクトリを指定します (デフォルト: スクリプトと同じディレクトリ)。

.EXAMPLE
    .\uninstall.ps1
    .\uninstall.ps1 -Lang en
    .\uninstall.ps1 -InstallDir "C:\Yakumo"
#>

[CmdletBinding()]
param(
    [ValidateSet("en", "jp", "")]
    [string]$Lang = "",
    [string]$InstallDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ================================================================
# 初期化 / Initialization
# ================================================================
$SCRIPT_DIR    = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($InstallDir -eq "") {
    $InstallDir = $SCRIPT_DIR
}

$MODELS_DIR    = Join-Path $InstallDir "libs\models"
$CONFIG_DEST   = Join-Path $InstallDir "affect.config"
$CONFIG_BACKUP = Join-Path $InstallDir "affect.config.bak"

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

# ================================================================
# ユーティリティ関数 / Utility functions
# ================================================================
function Write-Header {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  Yakumo Affect Engine -- Uninstaller" -ForegroundColor Cyan
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
function Write-Info { param([string]$Msg); Write-Host "      $Msg" -ForegroundColor White }

# ================================================================
# Step 1: モデルファイルの削除 / Deleting the model files
# ================================================================
Write-Header
Write-Step "1" (L "モデルファイルの削除" "Deleting the model files")

if (Test-Path $MODELS_DIR) {
    # フォルダサイズを概算表示 / Show the approximate folder size
    $sizeBytes = (Get-ChildItem -Recurse -File $MODELS_DIR | Measure-Object -Property Length -Sum).Sum
    $sizeMB    = [math]::Round($sizeBytes / 1MB, 1)
    Write-Info (L "削除対象: $MODELS_DIR" "Target: $MODELS_DIR")
    Write-Info (L "合計サイズ: 約 ${sizeMB} MB" "Total size: approx. ${sizeMB} MB")
    Write-Host ""
    $confirm = Read-Host (L "  モデルファイルを削除しますか? [y/N]" "  Delete the model files? [y/N]")
    if ($confirm -match "^[yY]$") {
        Remove-Item -Recurse -Force $MODELS_DIR
        Write-OK (L "モデルファイルを削除しました" "Model files deleted")
    } else {
        Write-Info (L "スキップしました" "Skipped")
    }
} else {
    Write-Info (L "モデルフォルダが見つかりません (スキップ): $MODELS_DIR" `
                  "Models folder not found (skipping): $MODELS_DIR")
}

# ================================================================
# Step 2: 設定ファイルのバックアップ → 削除 / Backing up & removing the config
# ================================================================
Write-Step "2" (L "設定ファイルのバックアップ" "Backing up the configuration file")

if (Test-Path $CONFIG_DEST) {
    if (Test-Path $CONFIG_BACKUP) {
        Write-Warn (L "affect.config.bak が既に存在します — 上書きします" `
                      "affect.config.bak already exists -- it will be overwritten")
    }
    Copy-Item $CONFIG_DEST $CONFIG_BACKUP -Force
    Remove-Item $CONFIG_DEST -Force
    Write-OK (L "affect.config を affect.config.bak にバックアップして削除しました" `
                "affect.config was backed up to affect.config.bak and removed")
} else {
    Write-Info (L "affect.config が見つかりません (スキップ)" "affect.config not found (skipping)")
}

# ================================================================
# Step 3: Python パッケージの削除案内 / About Python packages
# ================================================================
Write-Step "3" (L "Python パッケージについて" "About the Python packages")

Write-Info (L "Python パッケージは自動削除しません" "Python packages are NOT removed automatically")
Write-Info (L "(他のツールとの競合を避けるため)" "(to avoid conflicts with other tools)")
Write-Host ""
Write-Host (L "  手動で削除する場合は以下を実行してください:" `
              "  To remove them manually, run:") -ForegroundColor White
Write-Host ""
Write-Host "    pip uninstall -y flask transformers torch sentencepiece protobuf ``" -ForegroundColor DarkGray
Write-Host "                     optimum onnx onnxconverter-common onnxruntime ``" -ForegroundColor DarkGray
Write-Host "                     sacremoses hf_xet" -ForegroundColor DarkGray
Write-Host ""
Write-Info (L "HuggingFace モデルキャッシュ (~\.cache\huggingface\) も手動削除が必要です" `
              "The HuggingFace model cache (~\.cache\huggingface\) also needs manual removal")

# ================================================================
# 完了 / Done
# ================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host (L "  アンインストール完了" "  Uninstall complete") -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
if (Test-Path $CONFIG_BACKUP) {
    Write-Host (L "  設定バックアップ: $CONFIG_BACKUP" "  Config backup: $CONFIG_BACKUP") -ForegroundColor White
}
Write-Host ""
