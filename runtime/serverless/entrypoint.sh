#!/usr/bin/env bash
set -euo pipefail

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/data/cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/cache/matplotlib}"
# Runpod's worker launcher may provide its own PYTHONPATH.  Preserve it, but
# prepend the vendored CosyVoice sources so `cosyvoice.flow` can import the
# sibling Matcha-TTS package during model construction.
export PYTHONPATH="/app:/app/third_party/Matcha-TTS:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

echo "[somali-entrypoint] starting model-cache bootstrap" >&2
python /app/runtime/serverless/bootstrap_model.py
echo "[somali-entrypoint] model cache bootstrap complete; starting FastAPI on port ${PORT:-8000}" >&2
exec uvicorn runtime.serverless.somali_api:app --host 0.0.0.0 --port "${PORT:-8000}"
