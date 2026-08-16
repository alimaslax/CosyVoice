# Somali CosyVoice on Runpod Flash

This Flash project is the deployment configuration for the public GHCR Somali
CosyVoice image. It does not rebuild the image.

It creates a separate queue-based endpoint named `cosyvoice-somali-4090-vllm` that
is pinned to an RTX 4090. It runs the same Somali checkpoint through vLLM,
allows one concurrent request, starts from zero, and stops five seconds after
the last job. The configuration lives in `somali_endpoint.py`; the Runpod UI
is not the source of truth.

## One-time prerequisite

Put a Hugging Face read token in the ignored repository-root `.env`:

```bash
HF_TOKEN=hf_your_read_token
```

`HF_TOKEN` is passed to the running worker only; it is not committed or
included in the image. The GHCR image is public, so no registry credential is
needed.

Attach a Runpod network volume and use `/runpod-volume/models/somali` as
`MODEL_DIR`. The worker atomically writes the Hugging Face snapshot and its
completion marker there, so later scale-from-zero requests reuse it. If a
volume cannot be attached in the selected region, the endpoint still works,
but each fresh worker must populate its ephemeral model cache.

## Deploy

```bash
./scripts/deploy_runpod_vllm.sh
```

`flash deploy` applies the Python configuration above. It is the command to
use for future changes rather than manually editing the Runpod UI.

The vLLM build is tagged `0.3.2-vllm`; it validates and atomically commits
`MODEL_DIR/vllm` before accepting jobs. An old, partial, or checkpoint-mismatched
export is rebuilt in a staging directory and replaced only after validation.
The vLLM image rejects `COSYVOICE_BACKEND=torch`; keep `0.2.3` deployed
separately as an audio-quality control until the end-to-end synthesis benchmark
is better.

For the queue endpoint, submit a job from the repository root:

```bash
./scripts/runpod_tts.sh medium "Qoraalka Soomaaliga halkan geli." output.wav
```

Treat completion only as transport success. Check the WAV duration with
`ffprobe`, listen to it, and compare it with a PyTorch-control WAV generated
from the same text and pace. Record synthesis seconds divided by generated
audio seconds (RTF); queue/cold-start time is reported separately.
