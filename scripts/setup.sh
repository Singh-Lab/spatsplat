#!/bin/bash
# scripts/setup.sh - Infrastructure logic

# 1. Path Management
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 2. Load variables from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "Error: .env file not found in $PROJECT_ROOT"
    exit 1
fi

# 3. Create directory structure
mkdir -p "$SPATSPLAT_DATA_DIR/raw" "$SPATSPLAT_DATA_DIR/processed"
mkdir -p "$PROJECT_ROOT/notebooks/checkpoints" "$PROJECT_ROOT/logs"

# 4. Conda Environment
ENV_NAME=${CONDA_ENV_NAME:-spatsplat}
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "--> Updating environment '$ENV_NAME'..."
    conda env update --name "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml" --prune
else
    echo "--> Creating environment '$ENV_NAME'..."
    conda env create --name "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml"
fi

echo "------------------------------------------------"
echo "Infrastructure Setup Complete."
echo "CRITICAL: You must manually install PyTorch for your specific hardware."
echo "Visit https://pytorch.org/get-started/locally/ for instructions."
echo "------------------------------------------------"