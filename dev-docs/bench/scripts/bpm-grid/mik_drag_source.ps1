#Requires -Version 7
# Drag source for Mixed In Key: a small window that carries every file of a
# playlist as one Explorer-style file drop. Drag the green panel onto the
# Mixed In Key window to add all tracks at once.
param(
    [string]$Playlist = ''
)
$ErrorActionPreference = 'Stop'
if (-not $Playlist) {
    . (Join-Path $PSScriptRoot 'bpm_grid_paths.ps1')
    $Playlist = Join-Path $BpmGridDataset 'playlists\djts-playlist-broken.txt'
}
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$files = [System.Collections.Specialized.StringCollection]::new()
$missing = 0
foreach ($line in [System.IO.File]::ReadAllLines($Playlist)) {
    $entry = $line.Trim()
    if (-not $entry -or $entry.StartsWith('#')) { continue }
    $full = [System.IO.Path]::GetFullPath($entry)
    if ([System.IO.File]::Exists($full)) { [void]$files.Add($full) } else { $missing++ }
}

$form = [System.Windows.Forms.Form]::new()
$form.Text = 'MIK drag source'
$form.Size = [System.Drawing.Size]::new(360, 200)
$form.StartPosition = 'Manual'
$form.Location = [System.Drawing.Point]::new(40, 40)
$form.TopMost = $true

$panel = [System.Windows.Forms.Label]::new()
$panel.Dock = 'Fill'
$panel.TextAlign = 'MiddleCenter'
$panel.BackColor = [System.Drawing.Color]::FromArgb(46, 160, 67)
$panel.ForeColor = [System.Drawing.Color]::White
$panel.Font = [System.Drawing.Font]::new('Segoe UI', 12, [System.Drawing.FontStyle]::Bold)
$panel.Text = "Drag me onto Mixed In Key`n$($files.Count) files" +
    $(if ($missing) { "`n($missing missing)" } else { '' })
$panel.Add_MouseDown({
    $data = [System.Windows.Forms.DataObject]::new()
    $data.SetFileDropList($files)
    $result = $panel.DoDragDrop($data, [System.Windows.Forms.DragDropEffects]::Copy)
    $panel.Text = "Dropped: $result`n$($files.Count) files"
})
$form.Controls.Add($panel)
[void]$form.ShowDialog()
