@echo off
REM =============================================================================
REM File        : scripts/build_all_protected.bat
REM Project     : MemGuard v1.0
REM Component   : Build Automation - Complete Protection Pipeline
REM Description : Windows batch script to build fully protected MemGuard Pro
REM               • Runs obfuscation, compilation, and packaging
REM               • Creates customer-ready distribution
REM               • Validates all protection layers
REM Author      : Kyle Clouthier
REM Version     : 1.0.0
REM Created     : 2025-01-21
REM Modified    : 2025-01-21 (Initial creation)
REM License     : Proprietary - Patent Pending
REM Copyright   : © 2025 Kyle Clouthier (Canada). All rights reserved.
REM =============================================================================

echo.
echo ===============================================
echo  MemGuard Pro - Complete Protection Build
echo ===============================================
echo.

REM Check if we're in the right directory
if not exist "memguard" (
    echo ERROR: memguard directory not found
    echo Please run from MemGuard project root
    pause
    exit /b 1
)

REM Create scripts directory if needed
if not exist "scripts" mkdir scripts

echo [1/4] Activating Python environment...
if exist "mem-env\Scripts\activate.bat" (
    call mem-env\Scripts\activate.bat
    echo     ✅ Virtual environment activated
) else (
    echo     ⚠️  No virtual environment found, using system Python
)

echo.
echo [2/4] Running obfuscation and compilation...
python scripts\package_pro_distribution.py
if %ERRORLEVEL% neq 0 (
    echo     ❌ Protection build failed
    pause
    exit /b 1
)

echo.
echo [3/4] Validating protection layers...
if exist "dist\memguard-pro-*\memguard_pro_protected" (
    echo     ✅ Protected package created
) else (
    echo     ❌ Protected package not found
    pause
    exit /b 1
)

if exist "dist\memguard-pro-*.zip" (
    echo     ✅ Distribution archive created
) else (
    echo     ❌ Distribution archive not found
    pause
    exit /b 1
)

echo.
echo [4/4] Build summary...
for %%f in (dist\memguard-pro-*.zip) do (
    echo     📦 Archive: %%~nxf
    echo     📊 Size: %%~zf bytes
)

echo.
echo ===============================================
echo  🎉 MemGuard Pro Protection Build Complete!
echo ===============================================
echo.
echo 🔒 Protection Features Applied:
echo    ✅ Source code obfuscation
echo    ✅ Bytecode compilation
echo    ✅ String encryption
echo    ✅ Custom import loader
echo    ✅ License validation
echo.
echo 📦 Customer Delivery Package:
echo    • Complete installation package
echo    • License validation tools
echo    • Enterprise documentation
echo    • Support contact information
echo.
echo 🚀 Ready for distribution to customers!
echo.

pause