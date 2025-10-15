#!/bin/bash

# Table QA Project Environment Setup Script
# Usage: bash setup.sh

set -e

echo "======================================"
echo "Table QA Project Environment Setup"
echo "======================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
ENV_NAME="table-qa"
PYTHON_VERSION="3.10"

echo ""
echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch (CUDA 11.8)
echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "======================================"
echo "Verifying installation..."
echo "======================================"

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CUDA not available')"
python -c "from transformers import AutoModelForCausalLM; print('Transformers: OK')"
python -c "import pandas as pd; print(f'Pandas version: {pd.__version__}')"
python -c "from trl import PPOTrainer; print('TRL: OK')"

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To download datasets, run:"
echo "  bash scripts/download_datasets.sh"
echo ""
