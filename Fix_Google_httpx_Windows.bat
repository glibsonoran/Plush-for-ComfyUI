@echo off
REM Force‑reinstall the incompatible packages
REM Copy this file into the "\scripts" subdirectory of the python installation used for ComfyUI
REM Then run this file
pip install --force-reinstall -v "google.genai==1.5.0"
pip install --force-reinstall -v "httpx==0.27.2"

REM Display installed versions for verification
echo.
echo Installed versions:
for /f "tokens=2 delims=: " %%A in ('pip show google.genai ^| findstr /R /C:"Version"') do set GG_VER=%%A
for /f "tokens=2 delims=: " %%A in ('pip show httpx ^| findstr /R /C:"Version"') do set HTTPX_VER=%%A
echo google.genai version: %GG_VER%
echo httpx version:      %HTTPX_VER%

pause
