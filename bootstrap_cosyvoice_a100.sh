#!/usr/bin/env bash
# Bootstrap an Ubuntu GPU server for the CosyVoice Somali trainer.
#
# Run with a token in the environment so it never enters this tracked file:
#   HF_TOKEN=... sudo -E bash bootstrap_cosyvoice_a100.sh
# Optional: WANDB_API_KEY=... sudo -E bash bootstrap_cosyvoice_a100.sh
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/alimaslax/CosyVoice.git}"
APP_DIR="${APP_DIR:-/opt/CosyVoice}"
IMAGE_NAME="${IMAGE_NAME:-cosyvoice:a100}"
DATASET_REPO="${DATASET_REPO:-lewenberg/omar-somali-asr}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo --preserve-env=HF_TOKEN,WANDB_API_KEY,WANDB_PROJECT \
    bash "$0" "$@"
fi

if [[ -z "${HF_TOKEN:-}" && -t 0 ]]; then
  read -r -s -p 'Hugging Face token: ' HF_TOKEN
  echo
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo 'HF_TOKEN is required.' >&2; exit 1; }

source /etc/os-release
[[ "${ID}" == "ubuntu" ]] || { echo 'This script supports Ubuntu only.' >&2; exit 1; }

apt_updated=0
apt_update() {
  if [[ "${apt_updated}" -eq 0 ]]; then
    apt-get update
    apt_updated=1
  fi
}

# Docker cannot install host GPU drivers. Install Ubuntu's recommended driver
# only when one is absent, then reboot and rerun this script.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'No NVIDIA driver detected; installing Ubuntu recommended driver.'
  apt_update
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers install
  echo 'Driver installed. Reboot the server, then rerun this script.'
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  apt_update
  apt-get install -y git
fi

# Install Docker only if missing; do not upgrade unrelated host packages.
if ! command -v docker >/dev/null 2>&1; then
  apt_update
  apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  cat >/etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt_updated=0
  apt_update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  apt_update
  apt-get install -y curl gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt_updated=0
  apt_update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

if [[ ! -d "${APP_DIR}/.git" ]]; then
  git clone --recurse-submodules "${REPO_URL}" "${APP_DIR}"
else
  git -C "${APP_DIR}" pull --ff-only
  git -C "${APP_DIR}" submodule update --init --recursive
fi

umask 077
printf 'HF_TOKEN=%s\n' "${HF_TOKEN}" >"${APP_DIR}/.env"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY=%s\n' "${WANDB_API_KEY}" >>"${APP_DIR}/.env"
fi
printf 'WANDB_PROJECT=%s\n' "${WANDB_PROJECT:-cosyvoice-somali}" >>"${APP_DIR}/.env"
chmod 600 "${APP_DIR}/.env"

# Keep BuildKit enabled: it caches the large CUDA/PyTorch dependency layer and
# avoids the legacy builder's high-memory Python installation path.
docker build --progress=plain -t "${IMAGE_NAME}" -f "${APP_DIR}/Dockerfile.a100" "${APP_DIR}"

mkdir -p "${APP_DIR}/work"
# Use the repository's token-aware Python downloader. The HF CLI can return a
# misleading 404 for fine-grained tokens inside this container.
docker run --rm --gpus all \
  --env-file "${APP_DIR}/.env" \
  -v "${APP_DIR}/work:/workspace/work" \
  "${IMAGE_NAME}" \
  python3 scripts/download_hf_snapshot.py "${DATASET_REPO}" \
    --repo-type dataset --local-dir "/workspace/work/${DATASET_REPO##*/}"

echo "Bootstrap complete. Start training with:"
echo "  cd ${APP_DIR} && RUN_NAME=somali-resume-\$(date +%Y%m%d) scripts/start_somali_a100_training.sh"
