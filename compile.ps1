<#
.SYNOPSIS
    Build StyleTransfer.exe with PyInstaller.

.DESCRIPTION
    1. Installs / upgrades PyInstaller into the project venv.
    2. Cleans any previous build/ and dist/ artefacts.
    3. Invokes PyInstaller with style_transfer.spec to produce a single
       self-contained portable exe (no installer needed on the target machine).

    Output: dist\StyleTransfer.exe

    Copy that single file to any Windows 10/11 x64 laptop and double-click.
    The app extracts itself to %TEMP% on first run (normal PyInstaller behaviour).

.NOTES
    * Run from the project root, or just call the script from anywhere — it
      uses $PSScriptRoot to locate the venv and spec file automatically.
    * The venv must already exist (.venv\).  Run `pip install -r requirements.txt`
      first if you are setting up a fresh clone.
    * torch / torchvision are excluded from the bundle (inference uses ONNX
      Runtime only).  The "Train new style" feature will not work in the
      compiled exe — it requires the full dev environment.
    * UPX is used for compression if it is on PATH; if not, PyInstaller
      silently skips it and the exe will be slightly larger.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root      = $PSScriptRoot                              # project root
$VenvPy    = "$Root\.venv\Scripts\python.exe"
$VenvPip   = "$Root\.venv\Scripts\pip.exe"
$SpecFile  = "$Root\style_transfer.spec"
$OutputExe = "$Root\dist\StyleTransfer.exe"

# Verify the venv exists before doing anything else
if (-not (Test-Path $VenvPy)) {
    throw "Python venv not found at $VenvPy.  Run: python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt"
}

# ── 1. Install / upgrade PyInstaller ─────────────────────────────────────
Write-Host "`n=== Installing / upgrading PyInstaller ===" -ForegroundColor Cyan
& $VenvPip install --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed (exit $LASTEXITCODE)" }

# ── 2. Clean previous artefacts ──────────────────────────────────────────
Write-Host "`n=== Cleaning previous build artefacts ===" -ForegroundColor Cyan
foreach ($dir in @("$Root\build", "$Root\dist")) {
    if (Test-Path $dir) {
        Remove-Item -Recurse -Force $dir
        Write-Host "  Removed: $dir"
    }
}

# ── 3. Run PyInstaller ───────────────────────────────────────────────────
Write-Host "`n=== Building StyleTransfer.exe (this takes a few minutes) ===" -ForegroundColor Cyan
& $VenvPy -m PyInstaller $SpecFile --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

# ── 4. Report result ─────────────────────────────────────────────────────
if (Test-Path $OutputExe) {
    $SizeMB = [math]::Round((Get-Item $OutputExe).Length / 1MB, 1)
    Write-Host "`n=== Build successful ===" -ForegroundColor Green
    Write-Host "    $OutputExe  ($SizeMB MB)"
    Write-Host ""
    Write-Host "Copy dist\StyleTransfer.exe to any Windows 10/11 x64 machine and run it." -ForegroundColor Yellow
} else {
    throw "Expected output not found: $OutputExe"
}
