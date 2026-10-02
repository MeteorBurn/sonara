#Requires -Version 7
# Sonara analysis of the straight set, read from the SSD copy.
# Same script and preset as the broken set (MIK range 79-192, all features,
# separate time_signature pass) -> <dataset>\data\1_sonara\run-<Name>.sqlite and
# <dataset>\data\1_sonara\json_<Name>\ (one JSON per track, numbered by playlist position).
# -Name: straight plus a revision suffix (the 0.3.7 baseline already holds json_straight).
# -Python: interpreter of the analysing environment (default: machine path sonara_python).
# -Revision, -Wheel: provenance passed to extract_reference.py as --revision / --wheel.
# Resumable: rerun to continue; already analysed tracks are skipped.
param(
    [Parameter(Mandatory)][string]$Name,
    [int]$Workers = 4,
    [int]$Threads = 4,
    [int]$Limit = 0,
    [string]$Python = '',
    [string]$Revision = '',
    [string]$Wheel = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
$python = if ($Python) { $Python } else { Get-BpmGridPath 'sonara_python' }
$pyArgs = @(
    (Join-Path $PSScriptRoot 'extract_reference.py'),
    '--playlist', (Join-Path (Get-BpmGridPath 'ssd_root') 'djts-playlist-straight.txt'),
    '--name', $Name,
    '--workers', $Workers,
    '--threads', $Threads
)
if ($Limit -gt 0) { $pyArgs += @('--limit', $Limit) }
if ($Revision) { $pyArgs += @('--revision', $Revision) }
if ($Wheel) { $pyArgs += @('--wheel', $Wheel) }
& $python @pyArgs
exit $LASTEXITCODE
