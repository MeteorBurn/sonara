#Requires -Version 7
# Copy the broken and straight evaluation playlists to one flat folder per set:
#   <ssd_root>\broken\<file name>   and   <ssd_root>\straight\<file name>
# (ssd_root from the dataset's paths.json unless -DestRoot is given). Writes
# SSD-side playlists in the same order (playlist index stays aligned with
# MIK, Beat This! and Sonara results) and a source -> copy map per set.
# Resumable: files already copied with the same size are skipped.
# Stops on a file-name collision instead of overwriting.
param(
    [string]$DestRoot = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
if (-not $DestRoot) { $DestRoot = Get-BpmGridPath 'ssd_root' }

$sets = [ordered]@{
    broken   = Join-Path $BpmGridDataset 'playlists\djts-playlist-broken.txt'
    straight = Join-Path $BpmGridDataset 'playlists\djts-playlist-straight.txt'
}

$destRootFull = [System.IO.Path]::GetFullPath($DestRoot).TrimEnd('\')
if ($destRootFull -match '^[A-Za-z]:$') { throw "Refusing a drive root as destination: $destRootFull" }

foreach ($set in $sets.Keys) {
    $sources = [System.IO.File]::ReadAllLines($sets[$set]) |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith('#') }
    $setRoot = Join-Path $destRootFull $set
    [void][System.IO.Directory]::CreateDirectory($setRoot)

    $seenNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $playlistLines = [System.Collections.Generic.List[string]]::new()
    $map = [System.Collections.Generic.List[object]]::new()
    $copied = 0; $skipped = 0; $missing = 0; [long]$copiedBytes = 0
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    for ($i = 0; $i -lt $sources.Count; $i++) {
        $source = [System.IO.Path]::GetFullPath($sources[$i])          # M:/a/b.flac -> M:\a\b.flac
        $name = [System.IO.Path]::GetFileName($source)
        if (-not $seenNames.Add($name)) { throw "file-name collision in '$set': $name" }
        $target = Join-Path $setRoot $name
        $sourceInfo = [System.IO.FileInfo]::new($source)

        if (-not $sourceInfo.Exists) {
            $missing++
            Write-Warning "missing, kept original path in playlist: $source"
            $playlistLines.Add($sources[$i])
            $map.Add([pscustomobject]@{ idx = $i + 1; source = $sources[$i]; copy = '' })
            continue
        }
        $targetInfo = [System.IO.FileInfo]::new($target)
        if ($targetInfo.Exists -and $targetInfo.Length -eq $sourceInfo.Length) {
            $skipped++
        } else {
            [System.IO.File]::Copy($source, $target, $true)
            if ([System.IO.FileInfo]::new($target).Length -ne $sourceInfo.Length) {
                throw "size mismatch after copy: $target"
            }
            $copied++
            $copiedBytes += $sourceInfo.Length
        }
        $playlistLines.Add($target)
        $map.Add([pscustomobject]@{ idx = $i + 1; source = $sources[$i]; copy = $target })

        $seconds = [math]::Max($timer.Elapsed.TotalSeconds, 0.001)
        $speed = $copiedBytes / 1MB / $seconds
        Write-Progress -Activity "Copy $set" -PercentComplete (100 * ($i + 1) / $sources.Count) `
            -Status ('{0}/{1}  copied {2}  skipped {3}  {4:N0} MB/s' -f ($i + 1), $sources.Count, $copied, $skipped, $speed)
        if (($i + 1) % 50 -eq 0 -or ($i + 1) -eq $sources.Count) {
            '{0,-8} {1,4}/{2}  copied {3,4}  skipped {4,4}  missing {5}  {6,6:N1} GB  {7,5:N0} MB/s' -f `
                $set, ($i + 1), $sources.Count, $copied, $skipped, $missing, ($copiedBytes / 1GB), $speed
        }
    }
    Write-Progress -Activity "Copy $set" -Completed

    $playlistPath = Join-Path $destRootFull "djts-playlist-$set.txt"
    [System.IO.File]::WriteAllLines($playlistPath, $playlistLines)       # UTF-8 without BOM
    $map | Export-Csv -LiteralPath (Join-Path $destRootFull "copy-map-$set.csv") -NoTypeInformation
    "{0}: done in {1:hh\:mm\:ss}; copied {2}, skipped {3}, missing {4}; playlist {5}" -f `
        $set, $timer.Elapsed, $copied, $skipped, $missing, $playlistPath
}
