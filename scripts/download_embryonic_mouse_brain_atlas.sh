#!/bin/bash
# scripts/download_embryonic_mouse_brain_atlas.sh - Data acquisition for the embryonic mouse brain neurogenesis atlas

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "Error: .env file not found in $PROJECT_ROOT"
    exit 1
fi

if [ -z "${SPATSPLAT_DATA_DIR:-}" ]; then
    echo "Error: SPATSPLAT_DATA_DIR is not set in .env"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed. Please install it to proceed."
    exit 1
fi

DATASET_NAME=${EMBRYONIC_MOUSE_BRAIN_ATLAS_DATASET_NAME:-embryonic_mouse_brain_atlas}
DEST_DIR="$SPATSPLAT_DATA_DIR/raw/$DATASET_NAME"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR" || { echo "Error: Could not enter $DEST_DIR"; exit 1; }

declare -A FILE_URLS=(
    ["Control02.h5ad"]="https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000235/stomics/STSP0007363/Control02.h5ad"
    ["Control05.h5ad"]="https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000235/stomics/STSP0007364/Control05.h5ad"
    ["Control06.h5ad"]="https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000235/stomics/STSP0007365/Control06.h5ad"
    ["Control09.h5ad"]="https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000235/stomics/STSP0007366/Control09.h5ad"
)

for filename in "${!FILE_URLS[@]}"; do
    url="${FILE_URLS[$filename]}"
    if [ -f "$filename" ]; then
        echo "Found existing $filename in $DEST_DIR. Skipping."
        continue
    fi

    echo "Downloading $filename..."
    echo "Source: $url"
    curl -L --retry 3 --retry-delay 2 "$url" -o "$filename"
done

echo "Data acquisition complete. Files are located in: $DEST_DIR"
