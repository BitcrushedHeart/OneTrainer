@echo off
:: Run as Administrator to set MSVC environment variables system-wide.
:: This enables torch.compile/inductor to find cl.exe.
:: Uses PowerShell to avoid setx's 1024-character PATH truncation.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Run this script as Administrator.
    pause
    exit /b 1
)

set "MSVC_VER=14.44.35207"
set "MSVC_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\%MSVC_VER%"
set "SDK_VER=10.0.26100.0"
set "SDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "CL_DIR=%MSVC_ROOT%\bin\Hostx64\x64"

:: Verify paths exist
if not exist "%CL_DIR%\cl.exe" (
    echo ERROR: cl.exe not found at %CL_DIR%\cl.exe
    pause
    exit /b 1
)

echo Setting MSVC environment variables...

:: INCLUDE
setx /M INCLUDE "%MSVC_ROOT%\include;%SDK_ROOT%\Include\%SDK_VER%\ucrt;%SDK_ROOT%\Include\%SDK_VER%\shared;%SDK_ROOT%\Include\%SDK_VER%\um;%SDK_ROOT%\Include\%SDK_VER%\winrt"

:: LIB
setx /M LIB "%MSVC_ROOT%\lib\x64;%SDK_ROOT%\Lib\%SDK_VER%\ucrt\x64;%SDK_ROOT%\Lib\%SDK_VER%\um\x64"

:: Append cl.exe directory to system PATH using PowerShell (no 1024-char limit)
powershell -Command ^
    "$clDir = '%CL_DIR%';" ^
    "$syspath = [Environment]::GetEnvironmentVariable('Path', 'Machine');" ^
    "if ($syspath -notlike \"*Hostx64\x64*\") {" ^
    "    [Environment]::SetEnvironmentVariable('Path', \"$syspath;$clDir\", 'Machine');" ^
    "    Write-Host 'Added cl.exe to system PATH.';" ^
    "} else {" ^
    "    Write-Host 'cl.exe directory already in PATH, skipping.';" ^
    "}"

echo.
echo Done. Restart your terminal for changes to take effect.
pause
