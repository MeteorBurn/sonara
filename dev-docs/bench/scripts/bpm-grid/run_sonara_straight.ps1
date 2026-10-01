#Requires -Version 7
# Sonara 0.3.7 analysis of the straight set, read from the SSD copy.
# Same script and preset as the broken set (MIK range 79-192, all features,
# separate time_signature pass) -> <dataset>\data\straight_1000.sqlite and
# <dataset>\data\straight_1000_json\ (one JSON per track, numbered by playlist position).
# Resumable: rerun to continue; already analysed tracks are skipped.
param(
    [int]$Workers = 4,
    [int]$Threads = 4,
    [int]$Limit = 0
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
$python = Get-BpmGridPath 'sonara_python'
$pyArgs = @(
    (Join-Path $PSScriptRoot 'extract_reference.py'),
    '--playlist', (Join-Path (Get-BpmGridPath 'ssd_root') 'djts-playlist-straight.txt'),
    '--name', 'straight_1000',
    '--workers', $Workers,
    '--threads', $Threads
)
if ($Limit -gt 0) { $pyArgs += @('--limit', $Limit) }
& $python @pyArgs
exit $LASTEXITCODE
