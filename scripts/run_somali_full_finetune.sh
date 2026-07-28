#!/usr/bin/env bash
# Run inside the cosyvoice:a100 container. It performs full fine-tuning of the
# LLM and Flow modules; HiFT/HiFi-GAN is deliberately never trained.
set -Eeuo pipefail

DATASET_REPO="lewenberg/so-audio-cosyvoice3"
BASE_MODEL_REPO="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
WORK_DIR="${WORK_DIR:-/workspace/work}"
DATA_DIR="${WORK_DIR}/so-audio-cosyvoice3"
MODEL_DIR="${WORK_DIR}/models/Fun-CosyVoice3-0.5B-2512"
RUN_DIR="${WORK_DIR}/somali-full-finetune"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250}"
LLM_CHECKPOINT="${LLM_CHECKPOINT:-${MODEL_DIR}/llm.pt}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required at runtime." >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${RUN_DIR}"

hf download "${DATASET_REPO}" --repo-type dataset --local-dir "${DATA_DIR}"
hf download "${BASE_MODEL_REPO}" --local-dir "${MODEL_DIR}"

# data.list in the uploaded artifact contains build-machine paths. Recreate
# portable lists from the downloaded shards.
find "${DATA_DIR}/train/parquet" -type f -name 'parquet_*.tar' | sort > "${RUN_DIR}/train.data.list"
find "${DATA_DIR}/dev/parquet" -type f -name 'parquet_*.tar' | sort > "${RUN_DIR}/dev.data.list"

CONFIG="${RUN_DIR}/cosyvoice3-somali-full-finetune.yaml"
cp examples/libritts/cosyvoice3/conf/cosyvoice3.yaml "${CONFIG}"
python3 - "${CONFIG}" "${CHECKPOINT_STEPS}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
checkpoint_steps = sys.argv[2]
contents = path.read_text(encoding="utf-8")
contents = contents.replace("    save_per_step: -1", f"    save_per_step: {checkpoint_steps}", 1)
path.write_text(contents, encoding="utf-8")
PY

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}"

run_full_finetune() {
  local model="$1"
  local checkpoint="$2"
  local model_dir="${RUN_DIR}/${model}"
  local tensorboard_dir="${RUN_DIR}/tensorboard/${model}"
  if [[ "${model}" == "llm" && -n "${LLM_TENSORBOARD_DIR:-}" ]]; then
    tensorboard_dir="${LLM_TENSORBOARD_DIR}"
  fi

  # Passing --model selects only this module. train.py nulls llm/flow/HiFT/
  # HiFi-GAN except the selected model, so no vocoder parameters are updated.
  torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config "${CONFIG}" \
    --train_data "${RUN_DIR}/train.data.list" \
    --cv_data "${RUN_DIR}/dev.data.list" \
    --qwen_pretrain_path "${MODEL_DIR}/CosyVoice-BlankEN" \
    --onnx_path "${MODEL_DIR}" \
    --model "${model}" \
    --checkpoint "${checkpoint}" \
    --model_dir "${model_dir}" \
    --tensorboard_dir "${tensorboard_dir}" \
    --ddp.dist_backend nccl \
    --num_workers 4 \
    --prefetch 8 \
    --pin_memory \
    --use_amp
}

echo "Starting full LLM fine-tune; checkpoints every ${CHECKPOINT_STEPS} optimizer steps."
run_full_finetune llm "${LLM_CHECKPOINT}"

echo "Starting full Flow fine-tune; checkpoints every ${CHECKPOINT_STEPS} optimizer steps."
run_full_finetune flow "${MODEL_DIR}/flow.pt"

echo "LLM and Flow full fine-tunes completed. HiFT/HiFi-GAN remained frozen."
