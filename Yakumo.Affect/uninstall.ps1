#Requires -Version 5.1
<#
.SYNOPSIS
    Yakumo Affect Engine — Uninstaller / アンインストーラ

.DESCRIPTION
    Cleans up what the installer downloaded and generated: the model files (optional)
    and affect.config (backed up first). The program files, affect.dict.json and the
    backups\ folder are left in place; delete the folder itself to remove everything.
    Python packages are NOT removed automatically (to avoid conflicts with other tools).
    インストーラーが取得・生成したもの（モデルファイル〔任意〕と affect.config〔バックアップ後〕）を
    片付けます。プログラム本体・affect.dict.json・backups\ フォルダは残します。
    すべて削除するにはフォルダごと削除してください。
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
$DICT_DEST     = Join-Path $InstallDir "affect.dict.json"

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

# ----------------------------------------------------------------
# 利用者ファイルのバックアップ / Backing up the user's files
# ----------------------------------------------------------------
# install.ps1 と同じ形式。backups\<日時>-<乱数5桁>\ に複製し、過去のバックアップは上書きしない。
# Same scheme as install.ps1: copies go to backups\<timestamp>-<5 random digits>\,
# so an earlier backup is never overwritten.
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
# Step 1: モデルファイルの削除 / Deleting the model files
# ================================================================
Write-Header
Write-Info (L "ダウンロードしたモデルと設定ファイルを片付けます。" `
              "This cleans up the downloaded models and the configuration file.")
Write-Info (L "プログラム本体・affect.dict.json・backups\ は削除しません（最後に案内します）。" `
              "The program files, affect.dict.json and backups\ are not deleted (see the end).")

# 完了画面で「削除したもの / 残したもの」を実際の結果から表示するための記録
# Records what actually happened, for the summary at the end
$modelsExisted = $false
$modelsDeleted = $false
$configDeleted = $false

Write-Step "1" (L "モデルファイルの削除" "Deleting the model files")

if (Test-Path $MODELS_DIR) {
    $modelsExisted = $true
    # フォルダサイズを概算表示 / Show the approximate folder size
    $sizeBytes = (Get-ChildItem -Recurse -File $MODELS_DIR | Measure-Object -Property Length -Sum).Sum
    $sizeMB    = [math]::Round($sizeBytes / 1MB, 1)
    Write-Info (L "削除対象: $MODELS_DIR" "Target: $MODELS_DIR")
    Write-Info (L "合計サイズ: 約 ${sizeMB} MB" "Total size: approx. ${sizeMB} MB")
    Write-Host ""
    $confirm = Read-Host (L "  モデルファイルを削除しますか? [y/N]" "  Delete the model files? [y/N]")
    if ($confirm -match "^[yY]$") {
        Remove-Item -Recurse -Force $MODELS_DIR
        $modelsDeleted = $true
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
    Backup-UserFile $CONFIG_DEST
    Remove-Item $CONFIG_DEST -Force
    $configDeleted = $true
    Write-OK (L "affect.config を削除しました" "affect.config was removed")
} else {
    Write-Info (L "affect.config が見つかりません (スキップ)" "affect.config not found (skipping)")
}

# 辞書は削除しない。利用者が育てたものなので、残したままにする
# The dictionary is never deleted: it is the user's own work
if (Test-Path $DICT_DEST) {
    Write-Info (L "affect.dict.json は削除せずに残します: $DICT_DEST" `
                  "affect.dict.json is kept, not deleted: $DICT_DEST")
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

# 実際に行った内容から「削除したもの / 残したもの」を組み立てる
# Build the removed / kept lists from what actually happened
$removedItems = @()
$keptItems    = @()

if ($modelsDeleted) {
    $removedItems += (L "モデルファイル" "model files")
} elseif ($modelsExisted) {
    $keptItems += (L "モデルファイル" "model files")
}

if ($configDeleted) {
    $removedItems += (L "affect.config（バックアップ済み）" "affect.config (backed up)")
}

if (Test-Path $DICT_DEST) {
    $keptItems += "affect.dict.json"
}

if (Test-Path (Join-Path $InstallDir "backups")) {
    $keptItems += "backups\"
}

$keptItems += (L "プログラム本体" "program files")

$noneLabel   = L "なし" "nothing"
$removedText = if ($removedItems.Count -gt 0) { $removedItems -join " / " } else { $noneLabel }
$keptText    = $keptItems -join " / "

Write-Host (L "  削除したもの : $removedText" "  Removed : $removedText") -ForegroundColor White
Write-Host (L "  残したもの   : $keptText" "  Kept    : $keptText") -ForegroundColor White
Write-Host (L "                 Python パッケージ / HuggingFace のキャッシュ（上記の手順で削除できます）" `
              "            Python packages / HuggingFace cache (see the steps above to remove them)") -ForegroundColor White
if ($null -ne $script:BackupDir) {
    Write-Host (L "  設定バックアップ: $script:BackupDir" "  Config backup: $script:BackupDir") -ForegroundColor White
}
Write-Host ""
Write-Host (L "  完全に削除するには、このフォルダごと削除してください:" `
              "  To remove everything, delete this folder:") -ForegroundColor Cyan
Write-Host "    $InstallDir" -ForegroundColor Cyan
Write-Host (L "  （affect.dict.json と backups\ が必要なら、先に別の場所へ移してください）" `
              "  (Move affect.dict.json and backups\ elsewhere first if you want to keep them)") -ForegroundColor Cyan
Write-Host ""
