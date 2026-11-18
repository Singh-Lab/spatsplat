#!/bin/bash
# scripts/download_pdac_mets_visium_zenodo_rds.sh - Download the large
# Zenodo RDS object for GSE272362 / PDAC_Mets

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

DATASET_NAME=${PDAC_METS_VISIUM_DATASET_NAME:-pdac_mets_visium_gse272362}
ZENODO_RECORD=${PDAC_METS_VISIUM_ZENODO_RECORD:-10712047}
RDS_NAME=${PDAC_METS_VISIUM_RDS_NAME:-PDAC_Updated.rds}
RDS_URL=${PDAC_METS_VISIUM_RDS_URL:-https://zenodo.org/records/${ZENODO_RECORD}/files/${RDS_NAME}?download=1}
RDS_MD5=${PDAC_METS_VISIUM_RDS_MD5:-d0f0b12e0fb013f3def1a62d0f925cbf}

DEST_DIR="$SPATSPLAT_DATA_DIR/raw/$DATASET_NAME"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR" || { echo "Error: Could not enter $DEST_DIR"; exit 1; }

if [ -f "$RDS_NAME" ]; then
    echo "Found existing $RDS_NAME in $DEST_DIR. Skipping download."
    exit 0
fi

echo "Downloading $RDS_NAME..."
echo "Source: $RDS_URL"
echo "Expected md5: $RDS_MD5"
curl -fL --retry 3 --retry-delay 2 "$RDS_URL" -o "$RDS_NAME"

echo
echo "Download complete: $DEST_DIR/$RDS_NAME"
echo "Suggested next step: inspect the R object structure and confirm that it contains spot-level expression and sample/barcode metadata."
