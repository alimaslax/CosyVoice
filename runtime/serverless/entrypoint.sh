#!/usr/bin/env bash
set -euo pipefail

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/data/cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/cache/matplotlib}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

python /app/runtime/serverless/bootstrap_model.py
exec uvicorn runtime.serverless.somali_api:app --host 0.0.0.0 --port "${PORT:-8000}"
