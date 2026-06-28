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

:: 5. cut-cross-entropy — fused linear+CE loss (Windows-compatible, no Triton needed)
::    --no-deps: prevents it from pulling in a CPU torch from PyPI over our CUDA build.
echo Installing cut-cross-entropy (no-deps)...
pip install --no-deps "cut-cross-entropy>=25.1.1"
if %errorlevel% neq 0 (
    echo ERROR: cut-cross-entropy installation failed.
    exit /b %errorlevel%
)

:: 7. Triton for Windows (required by Liger)
echo Installing triton-windows...
pip install "triton-windows>=3.7.0"
if %errorlevel% neq 0 (
    echo ERROR: triton-windows installation failed.
    exit /b %errorlevel%
)

:: 8. Liger fused kernels (RMSNorm, SwiGLU, fused CE loss)
echo Installing liger-kernel...
pip install "liger-kernel>=0.8.0"
if %errorlevel% neq 0 (
    echo ERROR: liger-kernel installation failed.
    exit /b %errorlevel%
)

:: 9. Flash Attention — pre-built Windows wheel (Python 3.12, CUDA 13.0, torch 2.11)
::    Compiling from source is not supported on Windows.
echo Installing flash-attn (Windows pre-built wheel)...
pip install "https://huggingface.co/Sumitc13/flash-attn-windows-wheels/resolve/main/flash_attn-2.8.3%%2Bcu130torch2.11-cp312-cp312-win_amd64.whl"
if %errorlevel% neq 0 (
    echo ERROR: flash-attn installation failed.
    exit /b %errorlevel%
)

echo Windows Installation Completed Successfully!
endlocal