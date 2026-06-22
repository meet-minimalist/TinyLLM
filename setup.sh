#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

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

# 5. Inject Liger Kernel with dependency checking disabled
echo "Safely installing liger-kernel without conflicting dependencies..."
pip install "liger-kernel>=0.5.0" --no-deps

# 6. Install flash-attn (optional, for varlen flash attention)
echo "Installing flash-attn (optional)..."
pip install flash-attn

echo "🎉 Installation Completed Successfully!"