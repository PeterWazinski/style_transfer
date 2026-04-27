<#
.SYNOPSIS
    Batch style transfer: apply all gallery styles to one image.

.DESCRIPTION
    Delegates all work to scripts/batch_styler.py.
    Requires the project virtual environment (.venv) at the repository root.

    Exactly one of -pdfoverview or -fullimage must be specified.

.PARAMETER PdfOverview
    Create a DIN-A4 landscape PDF contact sheet.
    The original image appears top-left; every style follows.
    Output: <image-dir>\<stem>_thumbnails.pdf

.PARAMETER FullImage
    Save a full-resolution styled JPEG for every style.
    Output: <image-dir>\<stem>_<stylename>.jpg  (one file per style)

.PARAMETER ImagePath
    Path to the source image file (JPEG or PNG).

.PARAMETER TileSize
    Tile size for ONNX inference in pixels (default: 1024).

.PARAMETER Overlap
    Tile overlap in pixels (default: 128).

.PARAMETER Strength
    Style blend strength 0.0-1.0 (default: 1.0).

.PARAMETER Float16
    Switch: enable float16 inference (faster on GPU/DML).

.EXAMPLE
    # PDF contact sheet (PowerShell):
    .\scripts\batch_styler.ps1 -pdfoverview photos\portrait.jpg

    # Full-resolution images (PowerShell):
    .\scripts\batch_styler.ps1 -fullimage photos\portrait.jpg -Strength 0.85

    # From Windows cmd.exe:
    powershell -ExecutionPolicy RemoteSigned -File scripts\batch_styler.ps1 -pdfoverview photos\portrait.jpg
#>

param(
    [Parameter(Position = 0)]
    [string]$ImagePath,

    [switch]$PdfOverview,
    [switch]$FullImage,

    [int]$TileSize = 1024,
    [int]$Overlap  = 128,
    [double]$Strength = 1.0,
    [switch]$Float16
)

# ── Validate mode ────────────────────────────────────────────────────────────
if (-not $PdfOverview -and -not $FullImage) {
    Write-Host ""
    Write-Host "ERROR: No mode specified. You must pass -pdfoverview or -fullimage." -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  batch_styler.ps1 -pdfoverview <image>  [options]"
    Write-Host "  batch_styler.ps1 -fullimage   <image>  [options]"
    Write-Host ""
    Write-Host "Modes:"
    Write-Host "  -pdfoverview   DIN-A4 landscape PDF with all styles + original in top-left"
    Write-Host "  -fullimage     One full-resolution JPEG per style"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -TileSize N    Tile size in pixels (default: 1024)"
    Write-Host "  -Overlap  N    Tile overlap in pixels (default: 128)"
    Write-Host "  -Strength F    Blend strength 0.0-1.0 (default: 1.0)"
    Write-Host "  -Float16       Use float16 inference (faster on GPU/DML)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\batch_styler.ps1 -pdfoverview photos\portrait.jpg"
    Write-Host "  .\batch_styler.ps1 -fullimage   photos\portrait.jpg -Strength 0.85"
    Write-Host ""
    exit 1
}

if (-not $ImagePath) {
    Write-Error "No image path specified."
    exit 1
}

# ── Locate Python and helper script ─────────────────────────────────────────
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

# ── Build argument list ──────────────────────────────────────────────────────
$PyArgs = @($PyScript)

if ($PdfOverview) { $PyArgs += "--pdfoverview" }
if ($FullImage)   { $PyArgs += "--fullimage"   }

$PyArgs += $ImagePath
$PyArgs += "--tile-size", $TileSize
$PyArgs += "--overlap",   $Overlap
$PyArgs += "--strength",  $Strength
if ($Float16) { $PyArgs += "--float16" }

& $Python @PyArgs
exit $LASTEXITCODE
