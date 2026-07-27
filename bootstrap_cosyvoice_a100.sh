#!/usr/bin/env bash
# Replace this value before copying the script to the Ubuntu server.
HF_TOKEN="hf_PASTE_YOUR_TOKEN_HERE"

set -Eeuo pipefail

REPO_URL="https://github.com/alimaslax/CosyVoice.git"
APP_DIR="/opt/CosyVoice"
IMAGE_NAME="cosyvoice:a100"
DATASET_REPO="lewenberg/so-audio-cosyvoice3"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo bash "$0" "$@"
fi

source /etc/os-release
if [[ "${ID}" != "ubuntu" ]]; then
  echo "This script supports Ubuntu only." >&2
  exit 1
fi

apt_updated=0
apt_update() {
  if [[ "${apt_updated}" -eq 0 ]]; then
    apt-get update
    apt_updated=1
  fi
}

# Docker cannot install host GPU drivers. Install Ubuntu's recommended driver
# only when one is absent, then rerun this script after the required reboot.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver detected; installing Ubuntu's recommended driver."
  apt_update
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers install
  echo "Driver installed. Reboot the server, then rerun this script."
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  apt_update
  apt-get install -y git
fi

# Install Docker only if it is not already available; never run apt upgrade.
if ! command -v docker >/dev/null 2>&1; then
  apt_update
  apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
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
  apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
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

if [[ "${HF_TOKEN}" == "hf_PASTE_YOUR_TOKEN_HERE" ]]; then
  echo "Set HF_TOKEN at the top of this script before running it." >&2
  exit 1
fi
printf 'HF_TOKEN=%s\n' "${HF_TOKEN}" >"${APP_DIR}/.env"
chmod 600 "${APP_DIR}/.env"

docker build -t "${IMAGE_NAME}" -f "${APP_DIR}/Dockerfile.a100" "${APP_DIR}"

mkdir -p "${APP_DIR}/work"
docker run --rm -it --gpus all \
  --env-file "${APP_DIR}/.env" \
  -v "${APP_DIR}/work:/workspace/work" \
  "${IMAGE_NAME}" bash -lc "
    huggingface-cli download ${DATASET_REPO} \\
      --repo-type dataset \\
      --local-dir /workspace/work/so-audio-cosyvoice3
    cd /workspace/work/so-audio-cosyvoice3
    find \"\$PWD/train/parquet\" -name 'parquet_*.tar' -type f | sort > train.data.list
    find \"\$PWD/dev/parquet\" -name 'parquet_*.tar' -type f | sort > dev.data.list
    exec bash
  "
