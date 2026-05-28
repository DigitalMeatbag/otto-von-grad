param(
    [string]$RawDir,
    [string]$OutFile
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $RawDir) {
    $RawDir = Join-Path $RepoRoot 'data/import/raw/philosophy'
}
if (-not $OutFile) {
    $OutFile = Join-Path $RepoRoot 'data/text/philosophy.txt'
}

$RawDir = [System.IO.Path]::GetFullPath($RawDir)
$OutFile = [System.IO.Path]::GetFullPath($OutFile)
$manifestPath = Join-Path $RawDir 'manifest.csv'

if (-not (Test-Path -LiteralPath $RawDir -PathType Container)) {
    throw "raw import directory not found: $RawDir"
}

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "raw import manifest not found: $manifestPath"
}

$rawFiles = Get-ChildItem -LiteralPath $RawDir -Filter '*.txt' -File | Sort-Object Name
if ($rawFiles.Count -eq 0) {
    throw "no raw text files found in: $RawDir"
}

$outDir = Split-Path -Parent $OutFile
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Output "raw directory: $RawDir"
Write-Output "raw files: $($rawFiles.Count)"
Write-Output "output file: $OutFile"
Write-Output 'TODO: strip source headers/footers, normalize text, and write the training corpus.'
