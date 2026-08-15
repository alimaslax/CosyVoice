# Somali CosyVoice on Runpod Flash

This Flash project deploys the supplied pre-built Somali CosyVoice image as a
Runpod Serverless HTTP service. It does not rebuild the image.

The endpoint selects Runpod's 24 GB Ampere pool (L4, A5000, or 3090), allows
one concurrent request, and scales from zero to one worker.

## One-time prerequisite

Flash supports using a private image but does not create container-registry
credentials. In Runpod, save the supplied VCCR Docker username and password as
a container-registry credential, then put the resulting credential ID in the
ignored repository-root `.env`:

```bash
HF_TOKEN=hf_your_read_token
RUNPOD_CONTAINER_REGISTRY_AUTH_ID=your_runpod_registry_credential_id
```

`HF_TOKEN` is passed to the running worker only; neither it nor the VCCR
password is committed or included in the image.

## Deploy

```bash
cd runtime/runpod_flash
flash build
flash deploy
```

After the first worker downloads the private model, use the endpoint's native
routes: `GET /health` and `POST /synthesize` with
`{"text":"...","pace":"medium"}`.
