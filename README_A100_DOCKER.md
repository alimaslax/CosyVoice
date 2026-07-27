# CosyVoice on an Ubuntu A100 server

This image uses the host's NVIDIA driver through the NVIDIA Container Toolkit.
GPU drivers are kernel modules, so a Dockerfile cannot install them correctly.
The host needs a working `nvidia-smi` and Docker's NVIDIA runtime; when they
are already present, do not install or upgrade anything.

If `nvidia-smi` is missing, install a current NVIDIA **host** driver suitable
for the server's Ubuntu release, reboot, install the NVIDIA Container Toolkit,
then verify `nvidia-smi` before building the image. Do not install a CUDA
toolkit separately: this image already supplies CUDA user-space libraries.

Build from the CosyVoice checkout:

```bash
docker build -f Dockerfile.a100 -t cosyvoice:a100 .
cp .env.example .env  # set HF_TOKEN to a write/read Hugging Face token
```

Download the private Parquet dataset and start an interactive GPU container:

```bash
set -a; source .env; set +a
docker run --rm -it --gpus all \
  --env HF_TOKEN \
  --volume "$PWD/work:/workspace/work" \
  cosyvoice:a100 bash -lc '
    huggingface-cli download levenberg/so-audio-cosyvoice3 \
      --repo-type dataset --local-dir /workspace/work/so-audio-cosyvoice3
    cd /workspace/work/so-audio-cosyvoice3
    find "$PWD/train/parquet" -name "parquet_*.tar" -type f | sort > train.data.list
    find "$PWD/dev/parquet" -name "parquet_*.tar" -type f | sort > dev.data.list
    exec bash
  '
```

Check GPU visibility inside the container with `nvidia-smi`. The Hugging Face
token is runtime-only and is not baked into the image.
