# Dot-source from a harness launcher: . (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
# Mirrors bpm_grid_paths.py: sets $BpmGridRepo, $BpmGridDataset, $BpmGridConfig.
# Machine paths = the first ```json block under "## Machine paths" in the dataset README.
$BpmGridRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
$BpmGridDataset = if ($env:SONARA_BPM_GRID_DATA) { $env:SONARA_BPM_GRID_DATA } else {
    Join-Path $BpmGridRepo 'dev-docs\bench\datasets\bpm-grid-2026-10'
}
$bpmGridConfigFile = Join-Path $BpmGridDataset 'README.md'
$BpmGridConfig = [pscustomobject]@{}
if (Test-Path -LiteralPath $bpmGridConfigFile) {
    $bpmGridReadme = Get-Content -LiteralPath $bpmGridConfigFile -Raw -Encoding utf8
    $bpmGridMatch = [regex]::Match($bpmGridReadme, '(?sm)^## Machine paths\b.*?^```json[ \t]*\r?\n(.*?)^```')
    if ($bpmGridMatch.Success) { $BpmGridConfig = $bpmGridMatch.Groups[1].Value | ConvertFrom-Json }
}

function Get-BpmGridPath([string]$Key) {
    $value = $BpmGridConfig.$Key
    if (-not $value) { throw "README.md machine paths have no '$Key' ($bpmGridConfigFile)" }
    [Environment]::ExpandEnvironmentVariables($value)
}
