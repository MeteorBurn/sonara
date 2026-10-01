#Requires -Version 7
# Beat This! (final0, GPU) analysis of the straight set, read from the SSD copy.
# Same script as the broken set (BPM = 60 / median beat interval, folded into
# 79-192) -> <dataset>\data\beat_this_straight\ (one JSON per track + _bpm.json).
# Resumable: rerun to continue; tracks that already have a JSON are skipped.
param(
    [int]$Decoders = 3,
    [int]$Limit = 0
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
$python = Get-BpmGridPath 'beat_this_python'
$pyArgs = @(
    (Join-Path $PSScriptRoot 'beat_this_bpm.py'),
    '--playlist', (Join-Path (Get-BpmGridPath 'ssd_root') 'djts-playlist-straight.txt'),
    '--out-dir', (Join-Path $BpmGridDataset 'data\beat_this_straight'),
    '--decoders', $Decoders
)
if ($Limit -gt 0) { $pyArgs += @('--limit', $Limit) }
& $python @pyArgs
exit $LASTEXITCODE
