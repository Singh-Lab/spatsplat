#!/bin/bash
# scripts/download_tiny_dataset.sh - Data Acquisition Logic

# 1. Path Management
# Get the directory where this script lives (scripts/)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Define the Project Root (one level up)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 2. Load variables from .env (Located in Project Root)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "Error: .env file not found in $PROJECT_ROOT"
    exit 1
fi

# 3. Navigate to the Data Directory
# Ensure the base raw directory exists
mkdir -p "$SPATSPLAT_DATA_DIR/raw/tiny_dataset"
cd "$SPATSPLAT_DATA_DIR/raw/tiny_dataset" || { echo "Error: Could not enter data directory"; exit 1; }

# 4. Dependency Check
for cmd in wget unzip; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed. Please install it to proceed."
        exit 1
    fi
done

# 5. Download the Mouse Brain Tiny Subset from 10x Genomics
# We check if the zip or the extracted folder already exists to avoid redundant downloads
if [ -f "Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs.zip" ] || [ -d "transcripts.csv.gz" ]; then
    echo "Tiny dataset already exists in $PWD. Skipping download."
else
    echo "Downloading Xenium Mouse Brain Tiny Subset..."
    wget https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs.zip
    
    echo "Unzipping data..."
    unzip Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs.zip
    
    echo "Cleaning up zip file..."
    rm Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_outs.zip
fi

echo "Data acquisition complete. Files are located in: $PWD"