param(
    [string]$CsvPath,
    [string]$OutDir,
    [double]$DelaySeconds = 2.0,
    [int]$TimeoutSec = 60,
    [int]$MaxDownloadsPerRun = 100,
    [string]$Contact = '',
    [switch]$AcceptProjectGutenbergTerms,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $CsvPath) {
    $CsvPath = Join-Path $RepoRoot 'data/import/philosophy.csv'
}
if (-not $OutDir) {
    $OutDir = Join-Path $RepoRoot 'data/import/raw/philosophy'
}

$CsvPath = [System.IO.Path]::GetFullPath($CsvPath)
$OutDir = [System.IO.Path]::GetFullPath($OutDir)
$UserAgent = 'otto-von-grad-data-import/0.1'
if (-not [string]::IsNullOrWhiteSpace($Contact)) {
    $UserAgent = "$UserAgent (+$Contact)"
}

function ConvertTo-Slug {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [int]$MaxLength = 80
    )

    $normalized = $Value.Normalize([System.Text.NormalizationForm]::FormD)
    $chars = New-Object System.Collections.Generic.List[char]
    foreach ($ch in $normalized.ToCharArray()) {
        $category = [System.Globalization.CharUnicodeInfo]::GetUnicodeCategory($ch)
        if ($category -ne [System.Globalization.UnicodeCategory]::NonSpacingMark) {
            $chars.Add($ch)
        }
    }

    $slug = (-join $chars).ToLowerInvariant()
    $slug = $slug -replace '[^a-z0-9]+', '-'
    $slug = $slug.Trim('-')
    if ($slug.Length -gt $MaxLength) {
        $slug = $slug.Substring(0, $MaxLength).Trim('-')
    }
    if (-not $slug) {
        return 'untitled'
    }
    return $slug
}

function ConvertTo-YearLabel {
    param([Parameter(Mandatory = $true)][string]$Year)

    $parsed = 0
    if ([int]::TryParse($Year, [ref]$parsed)) {
        if ($parsed -lt 0) {
            return ('{0:D4}bce' -f [Math]::Abs($parsed))
        }
        return ('{0:D4}ce' -f $parsed)
    }
    return ConvertTo-Slug -Value $Year -MaxLength 16
}

function Get-RawFilename {
    param(
        [int]$Index,
        [Parameter(Mandatory = $true)]$Row
    )

    $year = ConvertTo-YearLabel $Row.year
    $author = ConvertTo-Slug -Value $Row.author -MaxLength 40
    $title = ConvertTo-Slug -Value $Row.title -MaxLength 80
    return ('{0:D3}_{1}_{2}_{3}.txt' -f $Index, $year, $author, $title)
}

$rows = Import-Csv -LiteralPath $CsvPath
$requiredColumns = @('year', 'author', 'title', 'uri')
$actualColumns = @()
if ($rows.Count -gt 0) {
    $actualColumns = @($rows[0].PSObject.Properties.Name)
} else {
    $header = (Get-Content -LiteralPath $CsvPath -TotalCount 1)
    $actualColumns = @($header -split ',')
}

foreach ($column in $requiredColumns) {
    if ($actualColumns -notcontains $column) {
        throw "$CsvPath is missing required column '$column'"
    }
}

for ($i = 0; $i -lt $rows.Count; $i++) {
    foreach ($column in $requiredColumns) {
        if ([string]::IsNullOrWhiteSpace($rows[$i].$column)) {
            $lineNumber = $i + 2
            throw "$CsvPath line $lineNumber has empty required column '$column'"
        }
    }
}

if ($DryRun) {
    for ($i = 0; $i -lt $rows.Count; $i++) {
        $filename = Get-RawFilename -Index ($i + 1) -Row $rows[$i]
        Write-Output "$($rows[$i].uri) -> $(Join-Path $OutDir $filename)"
    }
    exit 0
}

if (-not $AcceptProjectGutenbergTerms) {
    throw @"
Project Gutenberg asks automated downloaders to follow its Terms of Use and robot access guidance:
  https://www.gutenberg.org/policy/terms_of_use.html
  https://www.gutenberg.org/policy/robot_access.html

Review those terms, then rerun with -AcceptProjectGutenbergTerms if this use is appropriate.
"@
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$downloadedAt = [DateTimeOffset]::UtcNow.ToString('o')
$manifest = New-Object System.Collections.Generic.List[object]
$toDownload = 0
$usesMainGutenbergHost = $false

foreach ($row in $rows) {
    $uri = [Uri]$row.uri
    if ($uri.Host -eq 'www.gutenberg.org' -or $uri.Host -eq 'gutenberg.org') {
        $usesMainGutenbergHost = $true
    }
}

for ($i = 0; $i -lt $rows.Count; $i++) {
    $filename = Get-RawFilename -Index ($i + 1) -Row $rows[$i]
    $target = Join-Path $OutDir $filename
    if ($Force -or -not (Test-Path -LiteralPath $target)) {
        $toDownload += 1
    }
}

if ($usesMainGutenbergHost -and $toDownload -gt $MaxDownloadsPerRun) {
    throw @"
This run would download $toDownload files from Project Gutenberg's main site.
Project Gutenberg says more than about 100 books/day should come from a mirror, not the main site.
Use cached files, lower the batch size, or point the CSV at a Project Gutenberg mirror before retrying.
"@
}

if ($usesMainGutenbergHost -and [string]::IsNullOrWhiteSpace($Contact)) {
    Write-Warning 'No -Contact was supplied. Consider adding a contact URL or email to the User-Agent for polite automated access.'
}

Write-Output "planned downloads: $toDownload"
Write-Output "delay between downloads: $DelaySeconds seconds"

for ($i = 0; $i -lt $rows.Count; $i++) {
    $row = $rows[$i]
    $filename = Get-RawFilename -Index ($i + 1) -Row $row
    $target = Join-Path $OutDir $filename
    $status = ''

    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        $bytes = [System.IO.File]::ReadAllBytes($target)
        $status = 'cached'
        Write-Output "cached  $filename"
    } else {
        $response = Invoke-WebRequest `
            -Uri $row.uri `
            -OutFile $target `
            -UserAgent $UserAgent `
            -TimeoutSec $TimeoutSec `
            -MaximumRedirection 5 `
            -UseBasicParsing

        $httpStatus = 200
        if ($null -ne $response -and $response.PSObject.Properties.Name -contains 'StatusCode') {
            $httpStatus = [int]$response.StatusCode
        }
        if ($httpStatus -ne 200) {
            throw "failed to download $($row.uri): HTTP $httpStatus"
        }
        if (-not (Test-Path -LiteralPath $target)) {
            throw "failed to download $($row.uri): no output file was created"
        }

        $bytes = [System.IO.File]::ReadAllBytes($target)
        if ($bytes.Length -eq 0) {
            throw "failed to download $($row.uri): output file is empty"
        }
        $status = [string]$httpStatus
        Write-Output "saved   $filename ($($bytes.Length) bytes)"

        if ($DelaySeconds -gt 0 -and $i -lt ($rows.Count - 1)) {
            Start-Sleep -Milliseconds ([int]($DelaySeconds * 1000))
        }
    }

    $sha = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
    ).Replace('-', '').ToLowerInvariant()

    $manifest.Add([pscustomobject]@{
        filename = $filename
        year = $row.year
        author = $row.author
        title = $row.title
        uri = $row.uri
        status = $status
        bytes = $bytes.Length
        sha256 = $sha
        downloaded_at = $downloadedAt
    })
}

$manifestPath = Join-Path $OutDir 'manifest.csv'
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
Write-Output "wrote   $manifestPath"
