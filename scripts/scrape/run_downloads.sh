#!/bin/bash
# scripts/onedrive/run_downloads.sh
#
# Pipeline:
#   1. Generate urls.txt from the Part10 OneDrive link (generate_urls.py)
#   2. Use aria2c + cookies.txt to download all parts into data/raw/...

set -euo pipefail

# Resolve project root (directory containing this script, then go up two levels if needed)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Where to put raw videos inside CCDS structure
OUTPUT_DIR="$PROJECT_ROOT/data/raw/20251124_01"

cd "$SCRIPT_DIR"

# echo "[*] Generating URLs..."
# python3 generate_urls.py

echo "[*] Starting aria2c downloads into $OUTPUT_DIR..."
mkdir -p "$OUTPUT_DIR"

aria2c -i urls.txt \
  --load-cookies=cookies.txt \
  --save-cookies=cookies.txt \
  --continue=true \
  --max-concurrent-downloads=2 \
  --split=8 \
  --min-split-size=64M \
  --file-allocation=none \
  -d "$OUTPUT_DIR"

echo "[*] Downloads finished. Files saved in: $OUTPUT_DIR"
