#!/bin/zsh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
runtime_cache="$script_dir/runtime-cache"

mkdir -p "$runtime_cache/numba" "$runtime_cache/matplotlib" "$runtime_cache/huggingface" "$runtime_cache/pycache"
chmod 700 "$runtime_cache" "$runtime_cache/numba" "$runtime_cache/matplotlib" "$runtime_cache/huggingface" "$runtime_cache/pycache"

export NUMBA_CACHE_DIR="$runtime_cache/numba"
export MPLCONFIGDIR="$runtime_cache/matplotlib"
export HF_HOME="$runtime_cache/huggingface"
export PYTHONPYCACHEPREFIX="$runtime_cache/pycache"
export TOKENIZERS_PARALLELISM=false
export COSYVOICE_DIR="${COSYVOICE_DIR:-/Users/mali/ai/CosyVoice}"

source ~/python/somali/bin/activate
cd "$script_dir"
exec python "$script_dir/run_zero_shot.py" "$@"
