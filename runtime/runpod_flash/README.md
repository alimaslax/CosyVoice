# Somali CosyVoice on Runpod Flash

This Flash project deploys the public GHCR Somali CosyVoice image as a Runpod
Serverless HTTP service. It does not rebuild the image.

The endpoint selects Runpod's 24 GB Ampere pool (L4, A5000, or 3090), allows
one concurrent request, and scales from zero to one worker.

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
completion marker there, so later scale-from-zero requests reuse it.

## Deploy

```bash
cd runtime/runpod_flash
flash build
flash deploy
```

After the first worker downloads the private model, use the endpoint's native
routes: `GET /health` and `POST /synthesize` with
`{"text":"...","pace":"medium"}`.
