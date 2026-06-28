#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Detect WSL2
IS_WSL=false
if grep -qi "microsoft" /proc/version 2>/dev/null; then
    IS_WSL=true
    echo "WSL2 detected."
fi

# CUDA version required by the torch build in requirements.txt (cu130 = 13.0)
REQUIRED_CUDA_MAJOR=13
REQUIRED_CUDA_MINOR=0

# Check CUDA toolkit version before doing anything else
check_cuda_version() {
    # Try to put the standard CUDA path on PATH if nvcc isn't found
    if ! command -v nvcc &>/dev/null; then
        export PATH=/usr/local/cuda/bin:$PATH
    fi

    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: nvcc not found. Install CUDA toolkit ${REQUIRED_CUDA_MAJOR}.${REQUIRED_CUDA_MINOR} first:"
        echo "  sudo apt install -y cuda-toolkit-${REQUIRED_CUDA_MAJOR}-${REQUIRED_CUDA_MINOR}"
        echo "  export PATH=/usr/local/cuda-${REQUIRED_CUDA_MAJOR}.${REQUIRED_CUDA_MINOR}/bin:\$PATH"
        exit 1
    fi

    # Parse major.minor from nvcc output, e.g. "Cuda compilation tools, release 13.0, V13.0.90"
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    echo "Detected CUDA toolkit: ${CUDA_VERSION}"
    echo "Required CUDA toolkit: ${REQUIRED_CUDA_MAJOR}.${REQUIRED_CUDA_MINOR}"

    if [ "$CUDA_MAJOR" -ne "$REQUIRED_CUDA_MAJOR" ] || [ "$CUDA_MINOR" -lt "$REQUIRED_CUDA_MINOR" ]; then
        echo "ERROR: CUDA version mismatch."
        echo "  Found   : ${CUDA_VERSION}"
        echo "  Required: >=${REQUIRED_CUDA_MAJOR}.${REQUIRED_CUDA_MINOR} (matches torch+cu130 in requirements.txt)"
        echo ""
        echo "To fix: install the correct toolkit and retry:"
        echo "  sudo apt install -y cuda-toolkit-${REQUIRED_CUDA_MAJOR}-${REQUIRED_CUDA_MINOR}"
        echo "  export PATH=/usr/local/cuda-${REQUIRED_CUDA_MAJOR}.${REQUIRED_CUDA_MINOR}/bin:\$PATH"
        exit 1
    fi

    echo "CUDA version check passed."
}

check_cuda_version

# Default behavior variables
VENV_PATH=""
CREATE_NEW=false

# 1. Parse Arguments
# Check if the user passed a path (e.g., ./setup.sh ./my_env)
if [ -n "$1" ]; then
    VENV_PATH="$1"
fi

# 2. Virtual Environment Evaluation Logic
if [ -n "$VENV_PATH" ]; then
    # The user provided a specific environment path
    if [ -d "$VENV_PATH" ]; then
        echo "Found existing environment at '$VENV_PATH'. Reusing it..."
    else
        echo "Environment path '$VENV_PATH' does not exist. Creating it now..."
        python3.12 -m venv "$VENV_PATH"
    fi
    echo "Activating '$VENV_PATH'..."
    source "$VENV_PATH/bin/activate"
    
else
    # The user did NOT provide an environment path argument
    echo "No environment path provided."
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Reusing currently active environment: $VIRTUAL_ENV"
    else
        echo "⚠️ Warning: No active environment detected, and no directory argument was provided."
        echo "Installing to the system Python scope."
    fi
fi

# 3. Upgrade Core Utilities
echo "Upgrading package manager utilities..."
python -m pip install --upgrade pip setuptools wheel

# 4. Clean Cache & Install Requirements Suite
echo "Clearing pip install caches..."
pip cache purge

echo "Installing base requirements.txt..."
pip install -r requirements.txt

# 5. cut-cross-entropy — fused linear+CE loss (no Triton needed)
# --no-deps: prevents it from pulling in a CPU torch from PyPI over our CUDA build.
echo "Installing cut-cross-entropy (no-deps)..."
pip install --no-deps "cut-cross-entropy>=25.1.1"

# 7. Triton (required by Liger)
echo "Installing triton..."
pip install "triton>=3.7.0"

# 8. Liger fused kernels (RMSNorm, SwiGLU, fused CE loss)
echo "Installing liger-kernel..."
pip install "liger-kernel>=0.8.0"

# 9. Flash Attention
# PyTorch 2.11 SDPA uses FA2 internally via cuDNN on Ampere — no separate
# flash-attn package needed. If you have a compatible pre-built wheel, install
# it here and set use_flash_attn: true in train_config.yaml to use the varlen API.
# pip install "<wheel-url>"

echo "Installation Completed Successfully!"