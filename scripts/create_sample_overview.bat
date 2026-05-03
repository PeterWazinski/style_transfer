@echo off
setlocal enabledelayedexpansion

:: Resolve project root (one level above this scripts\ folder)
set "ROOT=%~dp0.."
set "BATCHSTYLER=%ROOT%\dist\PetersPictureStyler\BatchStyler.exe"
set "SAMPLE_PICS=%ROOT%\sample_images\sample_pics"
set "STYLE_OVERVIEWS=%ROOT%\sample_images\style-overviews"
set "CHAIN_OVERVIEWS=%ROOT%\sample_images\style-chain-overviews"
set "CHAIN_DIR=%ROOT%\sample_images\style-chains"

if not exist "%STYLE_OVERVIEWS%" mkdir "%STYLE_OVERVIEWS%"
if not exist "%CHAIN_OVERVIEWS%" mkdir "%CHAIN_OVERVIEWS%"

echo === Style overviews ===
for %%f in ("%SAMPLE_PICS%\*.jpg") do (
    echo Processing %%~nxf ...
    "%BATCHSTYLER%" --style-overview --outdir "%STYLE_OVERVIEWS%" "%%f"
)

echo === Style-chain overviews ===
for %%f in ("%SAMPLE_PICS%\*.jpg") do (
    echo Processing %%~nxf ...
    "%BATCHSTYLER%" --style-chain-overview "%CHAIN_DIR%" --outdir "%CHAIN_OVERVIEWS%" "%%f"
)

echo Done.
endlocal
