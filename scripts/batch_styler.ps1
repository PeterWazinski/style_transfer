<#
.SYNOPSIS
    Apply all gallery styles to one image and write a PDF contact sheet.

.DESCRIPTION
    Delegates all work to scripts/batch_styler.py.
    Requires the project virtual environment (.venv) to be present in the
    repository root.

.PARAMETER ImagePath
    Path to the source image file (JPEG or PNG).

.PARAMETER TileSize
    Tile size for ONNX inference in pixels (default: 1024).

.PARAMETER Overlap
    Tile overlap in pixels (default: 128).

.PARAMETER Strength
    Style blend strength 0.0 – 1.0 (default: 1.0).

.PARAMETER Float16
    Switch: enable float16 inference (faster on GPU/DML).

.EXAMPLE
    # From PowerShell:
    .\scripts\batch_styler.ps1 my_dir\photo.jpg

    # From Windows cmd.exe:
    powershell -ExecutionPolicy RemoteSigned -File scripts\batch_styler.ps1 my_dir\photo.jpg

.EXAMPLE
    .\scripts\batch_styler.ps1 my_dir\photo.jpg -Strength 0.85 -TileSize 512
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ImagePath,

    [int]$TileSize = 1024,
    [int]$Overlap  = 128,
    [double]$Strength = 1.0,
    [switch]$Float16
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Split-Path -Parent $ScriptDir
$Python    = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PyScript  = Join-Path $ScriptDir "batch_styler.py"

if (-not (Test-Path $Python)) {
    Write-Error "Python venv not found at: $Python"
    Write-Error "Run: python -m venv .venv  then install dependencies."
    exit 1
}

if (-not (Test-Path $PyScript)) {
    Write-Error "Helper script not found: $PyScript"
    exit 1
}

$PyArgs = @(
    $PyScript,
    $ImagePath,
    "--tile-size", $TileSize,
    "--overlap",   $Overlap,
    "--strength",  $Strength
)
if ($Float16) {
    $PyArgs += "--float16"
}

& $Python @PyArgs
exit $LASTEXITCODE
