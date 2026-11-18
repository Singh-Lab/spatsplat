#!/bin/bash
# scripts/download_cosmx_nsclc_dataset.sh - Data acquisition for public CosMx NSCLC files

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

for cmd in curl unzip; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed. Please install it to proceed."
        exit 1
    fi
done

DATASET_NAME=${COSMX_NSCLC_DATASET_NAME:-cosmx_nsclc}
DOWNLOAD_URL=${COSMX_NSCLC_DOWNLOAD_URL:-https://github.com/FridleyLab/spatialGE_Data/raw/refs/heads/main/nanostring_nsclc.zip?download=}
ARCHIVE_NAME=${COSMX_NSCLC_ARCHIVE_NAME:-nanostring_nsclc.zip}

DEST_DIR="$SPATSPLAT_DATA_DIR/raw/$DATASET_NAME"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR" || { echo "Error: Could not enter $DEST_DIR"; exit 1; }

if [ -d "nanostring_nsclc" ] || [ -f "$ARCHIVE_NAME" ]; then
    echo "CosMx NSCLC dataset files already exist in $DEST_DIR."
    echo "Remove the archive/folder manually if you want to force a fresh download."
    exit 0
fi

echo "Downloading CosMx NSCLC dataset..."
echo "Source: $DOWNLOAD_URL"
curl -L --retry 3 --retry-delay 2 "$DOWNLOAD_URL" -o "$ARCHIVE_NAME"

echo "Unzipping data..."
unzip -q "$ARCHIVE_NAME"

echo "Cleaning up archive..."
rm -f "$ARCHIVE_NAME"

echo "Data acquisition complete. Files are located in: $DEST_DIR"
echo
echo "Note:"
echo "- The default source is the public spatialGE_Data NSCLC bundle."
echo "- If you obtain an official Bruker/NanoString direct download URL, set"
echo "  COSMX_NSCLC_DOWNLOAD_URL in .env and rerun this script."
