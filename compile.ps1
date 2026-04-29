<#
.SYNOPSIS
    Build PetersPictureStyler app directory with PyInstaller.

.DESCRIPTION
    1. Installs / upgrades PyInstaller into the project venv.
    2. Cleans any previous build/ and dist/ artefacts.
    3. Invokes PyInstaller with style_transfer.spec to produce a one-directory
       bundle: dist\PetersPictureStyler\
    4. Copies styles\ into the output directory so styles can be added later
       without recompiling.

    Output: dist\PetersPictureStyler\
              PetersPictureStylist.exe   ← double-click to run GUI
              BatchStyler.exe            ← headless CLI for batch style transfer
              styles\                    ← drop new style folders here
              app.log                    ← written at runtime

    Copy the entire dist\PetersPictureStyler\ folder to any Windows 10/11 x64
    machine.  To add a new style later, just drop its folder into styles\ and
    append the entry to styles\catalog.json — no recompile needed.

.NOTES
    * Only the Stylist UI app is compiled — the Trainer is a developer-only
      workflow that requires the full dev environment (torch, torchvision).
    * Run from the project root, or just call the script from anywhere — it
      uses $PSScriptRoot to locate the venv and spec file automatically.
    * The venv must already exist (.venv\).  Run `pip install -r requirements.txt`
      first if you are setting up a fresh clone.
    * torch / torchvision are excluded from the bundle (inference uses ONNX
      Runtime only).
    * UPX is used for compression if it is on PATH; if not, PyInstaller
      silently skips it and the DLLs will be slightly larger.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root       = $PSScriptRoot                              # project root
$VenvPy     = "$Root\.venv\Scripts\python.exe"
$VenvPip    = "$Root\.venv\Scripts\pip.exe"
$SpecFile   = "$Root\style_transfer.spec"
$OutputDir  = "$Root\dist\PetersPictureStyler"
$OutputExe  = "$OutputDir\PetersPictureStylist.exe"
$BatchExe   = "$OutputDir\BatchStyler.exe"

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
Write-Host "`n=== Building PetersPictureStyler\ (this takes a few minutes) ===" -ForegroundColor Cyan
& $VenvPy -m PyInstaller $SpecFile --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

# ── 4. Remove stray EXE stubs that PyInstaller drops directly in dist\ ───
#    PyInstaller creates intermediate single-file stubs in dist\ as part of
#    the onedir COLLECT step.  They are not the final deliverable.
Write-Host "`n=== Removing intermediate EXE stubs from dist\ ===" -ForegroundColor Cyan
foreach ($stub in @("$Root\dist\PetersPictureStylist.exe", "$Root\dist\BatchStyler.exe")) {
    if (Test-Path $stub) {
        Remove-Item -Force $stub
        Write-Host "  Removed: $stub"
    }
}

# ── 5. Copy styles\ into the output directory ────────────────────────────
Write-Host "`n=== Copying styles\ into output directory ===" -ForegroundColor Cyan
$SrcStyles = "$Root\styles"
$DstStyles = "$OutputDir\styles"
if (Test-Path $DstStyles) { Remove-Item -Recurse -Force $DstStyles }
Copy-Item -Recurse $SrcStyles $DstStyles
$StyleCount = (Get-ChildItem $DstStyles -Directory).Count
Write-Host "  Copied $StyleCount style folder(s) to $DstStyles"

# ── 6. Report result ─────────────────────────────────────────────────────
if (Test-Path $OutputExe) {
    $ExeMB   = [math]::Round((Get-Item $OutputExe).Length / 1MB, 1)
    $BatchMB = if (Test-Path $BatchExe) { [math]::Round((Get-Item $BatchExe).Length / 1MB, 1) } else { "?" }
    $DirMB   = [math]::Round((Get-ChildItem $OutputDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 0)
    Write-Host "`n=== Build successful ===" -ForegroundColor Green
    Write-Host "    $OutputDir\  ($DirMB MB total)"
    Write-Host "    $OutputExe  ($ExeMB MB)"
    Write-Host "    $BatchExe  ($BatchMB MB)"
    Write-Host ""
    Write-Host "Copy the entire dist\PetersPictureStyler\ folder to any Windows 10/11 x64 machine." -ForegroundColor Yellow
    Write-Host "To add a new style: drop its folder into styles\ and update styles\catalog.json." -ForegroundColor Yellow
} else {
    throw "Expected output not found: $OutputExe"
}
