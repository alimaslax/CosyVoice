#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/pretrained_models/somali-resume-20260801-epoch-11-whole"
INSTRUCT_MODEL_DIR="${ROOT_DIR}/pretrained_models/somali-epoch11-instruct-hybrid"
PORT="${PORT:-8000}"
CACHE_ROOT="${TMPDIR:-/tmp}/cosyvoice-epoch11-cache"

source "${HOME}/python/somali/bin/activate"
export PYTHONPATH="${ROOT_DIR}/third_party/Matcha-TTS:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CACHE_ROOT}/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export COSYVOICE_PROMPT_WAV="${COSYVOICE_PROMPT_WAV:-/Users/mali/ai/sod-code/cosyvoice3/mac-inference/0007.flac}"
export COSYVOICE_PROMPT_TEXT_FILE="${COSYVOICE_PROMPT_TEXT_FILE:-/Users/mali/ai/sod-code/cosyvoice3/mac-inference/tts.text}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
cd "${ROOT_DIR}"
exec python webui.py --model_dir "${MODEL_DIR}" --instruct_model_dir "${INSTRUCT_MODEL_DIR}" --port "${PORT}"
