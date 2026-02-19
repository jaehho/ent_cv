#!/bin/bash
# Deploy the nuclio function
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNC_ROOT="$SCRIPT_DIR/nuclio"

echo "Creating project 'cvat' (if not exists)..."
nuctl create project cvat --platform local 2>/dev/null || true

echo "Deploying function..."
nuctl deploy --project-name cvat --path "$FUNC_ROOT" \
    --file "$FUNC_ROOT/function.yaml" --platform local \
    --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
    --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

echo ""
nuctl get function --platform local
