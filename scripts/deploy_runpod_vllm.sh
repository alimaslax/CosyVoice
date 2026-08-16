#!/usr/bin/env bash
# Build and deploy the code-defined RTX 4090 vLLM endpoint through Runpod Flash.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example and set HF_TOKEN." >&2
  exit 78
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
: "${HF_TOKEN:?HF_TOKEN is required in .env}"

cd "${ROOT_DIR}/runtime/runpod_flash"
flash build
flash deploy
