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

# 1. Extract unique case IDs
CASE_IDS=$(grep -oE '[0-9]{8}_[0-9]{2}' urls.txt | sort -u)
NUM_CASES=$(echo "$CASE_IDS" | wc -l)

echo "[*] Found $NUM_CASES unique cases. Starting parallel optimized downloads..."

# 2. Use xargs to process cases in parallel. 
# -P 4: Processes 4 cases at a time (adjust based on your bandwidth).
# -I {}: Placeholder for the CASE_ID.
echo "$CASE_IDS" | xargs -P 4 -I {} bash -c '
    CASE="{}"
    OUTPUT_DIR="'"$RAW_DATA_ROOT"'/$CASE"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Filter URLs for this case and pipe to aria2c
    grep "$CASE" urls.txt | aria2c -i - \
        --load-cookies=cookies.txt \
        --save-cookies=cookies.txt \
        --continue=true \
        --max-concurrent-downloads=5 \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --file-allocation=falloc \
        --summary-interval=0 \
        --console-log-level=error \
        -d "$OUTPUT_DIR"
    
    echo "[+] Finished Case: $CASE"
'

echo "[*] All case downloads finished."