# Somali CosyVoice on Runpod Flash

This Flash project is the deployment configuration for the public GHCR Somali
CosyVoice image. It does not rebuild the image.

It creates a separate queue-based endpoint named `cosyvoice-somali-4090` that
is pinned to an RTX 4090. It allows one concurrent request, starts from zero,
and stops five seconds after the last job. The configuration lives in
`somali_endpoint.py`; the Runpod UI is not the source of truth.

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
cd runtime/runpod_flash
flash build
flash deploy
```

`flash deploy` applies the Python configuration above. It is the command to
use for future changes rather than manually editing the Runpod UI.

For the queue endpoint, submit a job from the repository root:

```bash
./scripts/runpod_tts.sh medium "Qoraalka Soomaaliga halkan geli." output.wav
```
