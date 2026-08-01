#!/usr/bin/env bash
# Run on the GPU host from the CosyVoice checkout after creating .env with
# HF_TOKEN=<a read token scoped to the private dataset>.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work}"
RUN_NAME="${RUN_NAME:-somali-resume-$(date +%Y%m%d)}"
DATASET_REPO="${DATASET_REPO:-lewenberg/omar-somali-asr}"
DATA_DIR="${WORK_DIR}/${DATASET_REPO##*/}"
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-${WORK_DIR}/${DATASET_REPO##*/}-cosyvoice}"

LLM_CHECKPOINT="${LLM_CHECKPOINT:-${DATA_DIR}/training-runs/somali-scratch-lr-decay-20260728/checkpoints/llm/epoch_5_whole.pt}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-${DATA_DIR}/training-runs/flow-checkpoints-20260728/epoch_5_whole.pt}"

[[ -f "${ROOT_DIR}/.env" ]] || { echo "Missing ${ROOT_DIR}/.env with HF_TOKEN" >&2; exit 1; }
mkdir -p "${WORK_DIR}"

to_container_path() {
  local path="$1"
  if [[ "${path}" != "${WORK_DIR}/"* ]]; then
    echo "Checkpoint must be inside WORK_DIR (${WORK_DIR}): ${path}" >&2
    exit 1
  fi
  printf '/workspace/work/%s\n' "${path#"${WORK_DIR}/"}"
}

LLM_CHECKPOINT_CONTAINER="$(to_container_path "${LLM_CHECKPOINT}")"
FLOW_CHECKPOINT_CONTAINER="$(to_container_path "${FLOW_CHECKPOINT}")"
TRAINING_DATA_DIR_CONTAINER="$(to_container_path "${TRAINING_DATA_DIR}")"

exec docker run --rm --name "cosyvoice-${RUN_NAME}" --gpus all \
  --env-file "${ROOT_DIR}/.env" \
  -e RUN_NAME \
  -e DATASET_REPO \
  -e TRAINING_DATA_DIR="${TRAINING_DATA_DIR_CONTAINER}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-cosyvoice-somali}" \
  -e WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}" \
  -e ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-6}" \
  -e MAX_EPOCH="${MAX_EPOCH:-}" \
  -e WORK_DIR=/workspace/work \
  -e LLM_CHECKPOINT="${LLM_CHECKPOINT_CONTAINER}" \
  -e FLOW_CHECKPOINT="${FLOW_CHECKPOINT_CONTAINER}" \
  -v "${ROOT_DIR}:/workspace/CosyVoice" \
  -v "${WORK_DIR}:/workspace/work" \
  cosyvoice:a100 scripts/run_somali_full_finetune.sh
