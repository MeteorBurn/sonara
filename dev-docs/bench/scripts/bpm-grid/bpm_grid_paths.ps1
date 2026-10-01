# Dot-source from a harness launcher: . (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
# Mirrors bpm_grid_paths.py: sets $BpmGridRepo, $BpmGridDataset, $BpmGridConfig.
$BpmGridRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
$BpmGridDataset = if ($env:SONARA_BPM_GRID_DATA) { $env:SONARA_BPM_GRID_DATA } else {
    Join-Path $BpmGridRepo 'locals\datasets\bpm-grid'
}
$bpmGridConfigFile = Join-Path $BpmGridDataset 'paths.json'
$BpmGridConfig = if (Test-Path -LiteralPath $bpmGridConfigFile) {
    Get-Content -LiteralPath $bpmGridConfigFile -Raw | ConvertFrom-Json
} else { [pscustomobject]@{} }

function Get-BpmGridPath([string]$Key) {
    $value = $BpmGridConfig.$Key
    if (-not $value) { throw "paths.json has no '$Key' ($bpmGridConfigFile)" }
    [Environment]::ExpandEnvironmentVariables($value)
}
