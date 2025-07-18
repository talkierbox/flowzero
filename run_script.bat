@echo off
REM Resolve this script’s directory and change into it
pushd %~dp0

REM Ensure “scripts” is a Python package
if not exist "scripts\" (
    echo Error: scripts\ directory not found.
    exit /b 1
)
if not exist "scripts\__init__.py" (
    echo Error: scripts\__init__.py not found.
    exit /b 1
)

REM Collect all .py files in scripts\ excluding __init__.py
setlocal enabledelayedexpansion
set count=0
for %%F in ("scripts\*.py") do (
    if /I not "%%~nxF"=="__init__.py" (
        set /A count+=1
        set "file!count!=%%~nxF"
    )
)

if %count% EQU 0 (
    echo No Python files found in scripts\.
    exit /b 1
)

REM Display menu
echo Select a script to run:
for /L %%i in (1,1,%count%) do (
    echo %%i^) !file%%i!
)

:prompt
set /p choice=Enter number:
if not defined file%choice% (
    echo Invalid selection. Try again.
    goto prompt
)

REM Strip .py extension and run
set "fname=!file%choice%!"
set "mod=!fname:.py=!"
python -m scripts.%mod%
set exitCode=%ERRORLEVEL%

endlocal
popd
exit /b %exitCode%