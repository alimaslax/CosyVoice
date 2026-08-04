#!/usr/bin/env bash
# Launch the Omar punctuation + pace CosyVoice3 run on a prepared GPU host.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work}"
DATASET_REPO="${DATASET_REPO:-lewenberg/omar-cosyvoice3-punctuated-paced-v1}"
DATA_DIR="${WORK_DIR}/${DATASET_REPO##*/}"

export RUN_NAME="${RUN_NAME:-somali-punctuated-paced-20260802}"
export DATASET_REPO
export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-${DATA_DIR}}"
export LLM_CHECKPOINT="${LLM_CHECKPOINT:-${DATA_DIR}/training-runs/somali-punctuated-paced-20260802/checkpoints/llm/init.pt}"
export FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-${WORK_DIR}/models/Fun-CosyVoice3-0.5B-2512/flow.pt}"
export ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-6}"
# Keep generated checkpoints local unless upload is explicitly requested.
export ARCHIVE_TO_HF="${ARCHIVE_TO_HF:-false}"

exec "${ROOT_DIR}/scripts/start_somali_a100_training.sh"
