# CosyVoice on an Ubuntu A100 server

This image uses the host's NVIDIA driver through the NVIDIA Container Toolkit.
GPU drivers are kernel modules, so a Dockerfile cannot install them correctly.
The host needs a working `nvidia-smi` and Docker's NVIDIA runtime; when they
are already present, do not install or upgrade anything.

If `nvidia-smi` is missing, install a current NVIDIA **host** driver suitable
for the server's Ubuntu release, reboot, install the NVIDIA Container Toolkit,
then verify `nvidia-smi` before building the image. Do not install a CUDA
toolkit separately: this image already supplies CUDA user-space libraries.

> # BLACKWELL NOT SUPPORTED — USE ADA

For a new Ubuntu GPU server, [bootstrap_cosyvoice_a100.sh](bootstrap_cosyvoice_a100.sh)
installs only missing prerequisites, configures Docker's NVIDIA runtime,
builds the image, writes the protected `.env`, and downloads the dataset. Do
not paste a token into the script. Supply it through the environment instead:

```bash
HF_TOKEN=hf_your_read_token sudo -E bash bootstrap_cosyvoice_a100.sh
```

Add `WANDB_API_KEY=...` to that invocation when W&B logging is required.

## Resuming the Somali fine-tune

Create `.env` in the checkout with a Hugging Face **read** token that is
scoped to the private `lewenberg/omar-somali-asr` dataset. Do not commit
this file.

```bash
printf 'HF_TOKEN=hf_your_read_token\n' > .env
chmod 600 .env
docker build -f Dockerfile.a100 -t cosyvoice:a100 .
```

To mirror TensorBoard metrics to Weights & Biases, add these optional values
to the same `.env` file before launching. `WANDB_ENTITY` is optional for a
personal workspace.

```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=cosyvoice-somali
WANDB_ENTITY=your_team_or_username
```

The training runner records train loss, accuracy, learning rate, gradient norm,
and aggregated validation metrics directly to W&B on rank 0. It deliberately
does not use W&B TensorBoard-sync mode, which conflicts with explicit global
step values. The A100 launcher bind-mounts the checkout into the training
container, so changes to the runner or metric logging code take effect on the
next launch without an image rebuild.

The rank-zero process initializes W&B with `sync_tensorboard=True`; no W&B
credentials are stored in the image or repository.

The launcher downloads the dataset through the Hugging Face Python API with
`HF_TOKEN` passed explicitly. This avoids a CLI issue that can cause a 404 for
a valid fine-grained token inside a container. It resumes the LLM and Flow
modules from the default epoch-5 checkpoints below, while leaving HiFT and
HiFi-GAN frozen. Checkpoints and validation run every 3,000 optimizer steps.
Each completed checkpoint is uploaded to the dataset under
`training-runs/<run-name>/checkpoints/`; after a confirmed upload, only the
latest 16 local checkpoints are retained per model directory.

```bash
nohup scripts/start_somali_a100_training.sh > work/somali-resume.log 2>&1 &
tail -f work/somali-resume.log
```

Defaults:

- `training-runs/somali-scratch-lr-decay-20260728/checkpoints/llm/epoch_5_whole.pt`
- `training-runs/flow-checkpoints-20260728/epoch_5_whole.pt`

Override either path or the run name without editing files:

```bash
RUN_NAME=somali-resume-2 \
LLM_CHECKPOINT=/absolute/path/to/llm.pt \
FLOW_CHECKPOINT=/absolute/path/to/flow.pt \
scripts/start_somali_a100_training.sh
```

Set `CHECKPOINT_STEPS`, `CV_STEPS`, or `ARCHIVE_KEEP_LOCAL` to override the
3,000-step / 16-local-checkpoint defaults.

`ADDITIONAL_EPOCHS` defaults to `6` and is the number of epochs to run after
the supplied checkpoint for both LLM and Flow. An `epoch_5` checkpoint therefore
runs epochs 6–11, using CosyVoice's exclusive `max_epoch=12` limit. `MAX_EPOCH`
is an optional absolute upper bound; if it is too low, the runner raises it to
preserve the requested additional-epoch count.

## Operational runbook

### One-time server setup

1. Verify the host GPU and Docker GPU runtime:

   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

2. Clone or update this repository at `/opt/CosyVoice`, create the protected
   `.env` file, and build the image:

   ```bash
   cd /opt/CosyVoice
   chmod 600 .env
   docker build -f Dockerfile.a100 -t cosyvoice:a100 .
   ```

   > # BLACKWELL NOT SUPPORTED — USE ADA

### Start and monitor a run

```bash
cd /opt/CosyVoice
RUN_NAME=somali-resume-20260731 \
nohup scripts/start_somali_a100_training.sh \
  > work/somali-resume-20260731.log 2>&1 &

tail -n 100 -F work/somali-resume-20260731.log
```

From a local terminal:

```bash
ssh -i ~/.ssh/id_ed25519_verda_a100 -o IdentitiesOnly=yes root@SERVER_IP \
  'tail -n 100 -F /opt/CosyVoice/work/somali-resume-20260731.log'
```

Check whether training has reached the GPU (rather than relying only on the
log banner):

```bash
ssh -i ~/.ssh/id_ed25519_verda_a100 -o IdentitiesOnly=yes root@SERVER_IP \
  'docker ps --format "table {{.Names}}\\t{{.Status}}"; nvidia-smi'
```

### Hugging Face download correction

For a private repository with a fine-grained token, the `hf download` CLI can
return a misleading 404 from inside the container even when the token is
valid. `scripts/download_hf_snapshot.py` avoids that failure by calling
`huggingface_hub.snapshot_download(..., token=HF_TOKEN)` directly. The runner
uses it for both the private dataset and the base model. Keep `HF_TOKEN` only
in `.env`; do not place it in a command line, source file, Docker layer, or
Git history.

### W&B and checkpoint uploads

The rank-zero training process uses W&B's TensorBoard synchronization, so the
same scalar stream appears in TensorBoard and W&B. W&B does not receive data
until the training process has initialized after downloads complete.

The archive watcher begins before model training. It waits for a checkpoint to
settle, uploads it to Hugging Face, and only then prunes it locally. It keeps
the newest 16 `.pt` files in each model directory; remote checkpoint history
is retained under `training-runs/<run-name>/checkpoints/`.

### If the log shows only the CUDA banner

The banner is emitted by the CUDA base image before Python starts. A short
quiet period can be normal while the dataset and base model are downloaded.
If it remains unchanged, use `docker ps` and `nvidia-smi` above. If there is
no `cosyvoice-<run-name>` container and no GPU process, the run has stopped;
inspect the full log rather than assuming training is active:

```bash
cat /opt/CosyVoice/work/<run-name>.log
```

Stop a run cleanly with:

```bash
docker stop cosyvoice-<run-name>
```

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
    huggingface-cli download levenberg/omar-somali-asr \
      --repo-type dataset --local-dir /workspace/work/omar-somali-asr
    cd /workspace/work/omar-somali-asr
    find "$PWD/train/parquet" -name "parquet_*.tar" -type f | sort > train.data.list
    find "$PWD/dev/parquet" -name "parquet_*.tar" -type f | sort > dev.data.list
    exec bash
  '
```

Check GPU visibility inside the container with `nvidia-smi`. The Hugging Face
token is runtime-only and is not baked into the image.
