#!/usr/bin/env bash
# Run inside the cosyvoice:a100 container. It performs full fine-tuning of the
# LLM and Flow modules; HiFT/HiFi-GAN is deliberately never trained.
set -Eeuo pipefail

DATASET_REPO="${DATASET_REPO:-lewenberg/omar-somali-asr}"
BASE_MODEL_REPO="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
WORK_DIR="${WORK_DIR:-/workspace/work}"
DATA_DIR="${WORK_DIR}/${DATASET_REPO##*/}"
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-${DATA_DIR}-cosyvoice}"
MODEL_DIR="${WORK_DIR}/models/Fun-CosyVoice3-0.5B-2512"
RUN_NAME="${RUN_NAME:-somali-scratch-lr-decay}"
RUN_DIR="${WORK_DIR}/${RUN_NAME}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-3000}"
CV_STEPS="${CV_STEPS:-3000}"
ARCHIVE_KEEP_LOCAL="${ARCHIVE_KEEP_LOCAL:-16}"
# Archive settled checkpoints to Hugging Face by default. Set false only when
# a local-only run is explicitly wanted.
ARCHIVE_TO_HF="${ARCHIVE_TO_HF:-true}"
# WarmupLR reaches LEARNING_RATE at WARMUP_STEPS, then decays proportionally
# to 1/sqrt(step). This is deliberately lower and non-constant after the
# previous constant-1e-5 run overfit its validation split.
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
# Number of epochs to run after the epoch stored in the supplied checkpoints.
# CosyVoice treats max_epoch as an exclusive upper bound, so an epoch_5
# checkpoint plus six additional epochs needs max_epoch=12 (epochs 6..11).
ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-6}"
# Optional absolute upper bound. If set too low, it is raised to preserve the
# requested additional-epoch count.
MAX_EPOCH="${MAX_EPOCH:-}"
LLM_CHECKPOINT="${LLM_CHECKPOINT:-${MODEL_DIR}/llm.pt}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-${MODEL_DIR}/flow.pt}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required at runtime." >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${RUN_DIR}"

# The raw Omar snapshot is prepared separately and already mounted at DATA_DIR.
# Do not re-download the whole dataset during a resumed training launch: aside
# from wasting time, fine-grained HF tokens can reject the repository snapshot
# API even though individual authorized files are readable.
if [[ ! -f "${DATA_DIR}/train/metadata.jsonl" || ! -f "${DATA_DIR}/dev/metadata.jsonl" ]]; then
  echo "Raw dataset is missing under ${DATA_DIR}; download/prepare it before training." >&2
  exit 1
fi

# snapshot_download receives HF_TOKEN explicitly. This is reliable with
# fine-grained tokens in containers, where the hf CLI may not forward it.
python3 scripts/download_hf_snapshot.py "${BASE_MODEL_REPO}" \
  --local-dir "${MODEL_DIR}"

# data.list in the uploaded artifact contains build-machine paths. Recreate
# portable lists from the downloaded shards.
find "${TRAINING_DATA_DIR}/train/parquet" -type f -name 'parquet_*.tar' | sort > "${RUN_DIR}/train.data.list"
find "${TRAINING_DATA_DIR}/dev/parquet" -type f -name 'parquet_*.tar' | sort > "${RUN_DIR}/dev.data.list"
[[ -s "${RUN_DIR}/train.data.list" && -s "${RUN_DIR}/dev.data.list" ]] || {
  echo "Missing CosyVoice Parquet shards under ${TRAINING_DATA_DIR}; run scripts/prepare_omar_asr_dataset.py first." >&2
  exit 1
}

resume_epoch() {
  # CosyVoice checkpoints store the completed epoch in their state dict. Load
  # tensor payloads onto the meta device so this metadata check is cheap.
  python3 - "$1" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="meta", weights_only=False)
epoch = state.get("epoch", -1) if isinstance(state, dict) else -1
if not isinstance(epoch, int):
    raise SystemExit(f"Checkpoint epoch is not an integer: {epoch!r}")
print(epoch)
PY
}

LLM_RESUME_EPOCH="$(resume_epoch "${LLM_CHECKPOINT}")"
FLOW_RESUME_EPOCH="$(resume_epoch "${FLOW_CHECKPOINT}")"
LATEST_RESUME_EPOCH="${LLM_RESUME_EPOCH}"
if (( FLOW_RESUME_EPOCH > LATEST_RESUME_EPOCH )); then
  LATEST_RESUME_EPOCH="${FLOW_RESUME_EPOCH}"
fi
# CosyVoice uses range(start_epoch + 1, max_epoch). Preserve the requested
# number of *additional* epochs for every resumed component.
MIN_MAX_EPOCH=$((LATEST_RESUME_EPOCH + 1 + ADDITIONAL_EPOCHS))
if [[ -z "${MAX_EPOCH}" ]]; then
  MAX_EPOCH="${MIN_MAX_EPOCH}"
elif (( MAX_EPOCH < MIN_MAX_EPOCH )); then
  echo "MAX_EPOCH=${MAX_EPOCH} is too low for ${ADDITIONAL_EPOCHS} additional epochs after checkpoint epoch ${LATEST_RESUME_EPOCH}; raising it to ${MIN_MAX_EPOCH}."
  MAX_EPOCH="${MIN_MAX_EPOCH}"
fi

# Archive only when explicitly enabled. The default is local checkpoints,
# which are pulled to the Mac through SSH by the caller.
ARCHIVER_PID=""
if [[ "${ARCHIVE_TO_HF}" == "true" ]]; then
  python3 scripts/archive_cosyvoice_checkpoints.py \
    --watch "${RUN_DIR}" --repo-id "${DATASET_REPO}" --run-name "${RUN_NAME}" \
    --keep-local "${ARCHIVE_KEEP_LOCAL}" &
  ARCHIVER_PID=$!
fi
cleanup() {
  if [[ -n "${ARCHIVER_PID}" ]]; then
    kill "${ARCHIVER_PID}" 2>/dev/null || true
    wait "${ARCHIVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CONFIG="${RUN_DIR}/cosyvoice3-somali-full-finetune.yaml"
cp examples/libritts/cosyvoice3/conf/cosyvoice3.yaml "${CONFIG}"
python3 - "${CONFIG}" "${CHECKPOINT_STEPS}" "${CV_STEPS}" "${LEARNING_RATE}" "${WARMUP_STEPS}" "${MAX_EPOCH}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
checkpoint_steps = sys.argv[2]
cv_steps = sys.argv[3]
learning_rate = sys.argv[4]
warmup_steps = sys.argv[5]
max_epoch = sys.argv[6]
contents = path.read_text(encoding="utf-8")
contents = contents.replace("    save_per_step: -1", f"    save_per_step: {checkpoint_steps}\n    cv_per_step: {cv_steps}", 1)
contents = contents.replace("        lr: 1e-5 # change to 1e-5 during sft", f"        lr: {learning_rate} # scratch Somali fine-tune peak lr", 1)
contents = contents.replace("    scheduler: constantlr # change to constantlr during sft", "    scheduler: warmuplr # warm up then inverse-square-root decay", 1)
contents = contents.replace("        warmup_steps: 2500", f"        warmup_steps: {warmup_steps}", 1)
contents = contents.replace("    max_epoch: 200", f"    max_epoch: {max_epoch}", 1)
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

echo "Starting LLM fine-tune '${RUN_NAME}' from ${LLM_CHECKPOINT} (checkpoint epoch ${LLM_RESUME_EPOCH}); ${ADDITIONAL_EPOCHS} additional epochs (exclusive max epoch ${MAX_EPOCH}), peak lr ${LEARNING_RATE}, warmup ${WARMUP_STEPS}, validation/checkpoints every ${CHECKPOINT_STEPS} optimizer steps; keep ${ARCHIVE_KEEP_LOCAL} locally after archival."
run_full_finetune llm "${LLM_CHECKPOINT}"

echo "Starting full Flow fine-tune; checkpoints every ${CHECKPOINT_STEPS} optimizer steps."
run_full_finetune flow "${FLOW_CHECKPOINT}"

echo "LLM and Flow full fine-tunes completed. HiFT/HiFi-GAN remained frozen."
