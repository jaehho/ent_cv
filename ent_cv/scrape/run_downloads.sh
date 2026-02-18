#!/bin/bash
# scripts/onedrive/run_downloads.sh

set -euo pipefail

# Resolve project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
RAW_DATA_ROOT="$PROJECT_ROOT/data/raw"

cd "$SCRIPT_DIR"

if [ ! -f "urls.txt" ]; then
    echo "Error: urls.txt not found."
    exit 1
fi

# 1. Extract unique case IDs (e.g., 20251113_02, 20251124_01) from the URLs
# This looks for the pattern YYYYMMDD_NN in the URL path
CASE_IDS=$(grep -oE '[0-9]{8}_[0-9]{2}' urls.txt | sort -u)

echo "[*] Found $(echo "$CASE_IDS" | wc -l) unique cases. starting partitioned download..."

for CASE in $CASE_IDS; do
    OUTPUT_DIR="$RAW_DATA_ROOT/$CASE"
    echo "--- Processing Case: $CASE ---"
    echo "[*] Destination: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # 2. Filter urls.txt for only the current case and pipe to aria2c
    grep "$CASE" urls.txt | aria2c -i - \
        --load-cookies=cookies.txt \
        --save-cookies=cookies.txt \
        --continue=true \
        --max-concurrent-downloads=2 \
        --split=8 \
        --min-split-size=64M \
        --file-allocation=none \
        -d "$OUTPUT_DIR"
done

echo "[*] All case downloads finished."