@echo off
setlocal enabledelayedexpansion

:: Exit the script if any step fails critically
set "EXIT_ON_ERROR=1"

:: 1. Parse Arguments
set "VENV_PATH=%~1"

:: 2. Virtual Environment Evaluation Logic
if not "%VENV_PATH%"=="" (
    :: The user provided a specific environment path argument
    if exist "%VENV_PATH%\" (
        echo Found existing environment folder at "%VENV_PATH%". Reusing it...
    ) else (
        echo Environment path "%VENV_PATH%" does not exist. Creating it now...
        python -m venv "%VENV_PATH%"
        if %errorlevel% neq 0 (
            echo ERROR: Failed to create virtual environment.
            exit /b %errorlevel%
        )
    )
    echo Activating "%VENV_PATH%"...
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    :: The user did NOT provide an environment path argument
    echo No environment path argument provided.
    if not "%VIRTUAL_ENV%"=="" (
        echo Reusing currently active environment: %VIRTUAL_ENV%
    ) else (
        echo ⚠️ Warning: No active environment detected, and no folder argument was provided.
        echo Installing directly to the local system Python scope.
    )
)

:: 3. Upgrade Core Utilities
echo Upgrading package manager utilities...
python -m pip install --upgrade pip setuptools wheel

:: 4. Clean Cache & Install Requirements Suite
echo Clearing pip install caches...
pip --no-cache-dir cache purge

echo Installing base requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Installation of requirements.txt failed.
    exit /b %errorlevel%
)

:: 5. Inject Liger Kernel with dependency checking disabled
echo Safely installing liger-kernel without conflicting dependencies...
pip install "liger-kernel>=0.5.0" --no-deps
if %errorlevel% neq 0 (
    echo ERROR: Installation of liger-kernel failed.
    exit /b %errorlevel%
)

:: 6. Install flash-attn (optional, for varlen flash attention)
echo Installing flash-attn (optional)...
pip install flash-attn
if %errorlevel% neq 0 (
    echo ERROR: Installation of flash-attn failed.
    exit /b %errorlevel%
)

echo 🎉 Windows Installation Completed Successfully!
endlocal