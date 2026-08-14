#!/usr/bin/env bash
# One-shot Somali TTS: loads the model, writes one WAV, then exits.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${TMPDIR:-/tmp}/cosyvoice-one-shot-cache"

mkdir -p "${CACHE_ROOT}/numba" "${CACHE_ROOT}/matplotlib"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CACHE_ROOT}/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec conda run --no-capture-output -n cosy-voice \
  python "${ROOT_DIR}/scripts/speak_somali_pace.py" "$@"
