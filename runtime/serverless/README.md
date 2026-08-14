# Somali Pace TTS on Verda Serverless Containers

This is an L40S-focused, Linux/AMD64 HTTP image. It contains application code,
CUDA runtime dependencies, and three 24 kHz Omar reference clips only. It does
not contain model weights, the training dataset, Gradio, or credentials.

## API

`GET /health` returns `200` and `{"status":"ready"}` after the model loads.

`POST /synthesize` accepts JSON and returns `audio/wav`:

```json
{"text":"Qoraalka Soomaaliga halkan geli.", "pace":"medium"}
```

`pace` must be `slow`, `medium`, or `fast`.

## First deployment preparation

The complete model is published privately at
`lewenberg/somali-punctuated-paced-20260802`. If a fresh upload is needed, use:

```bash
hf upload lewenberg/somali-punctuated-paced-20260802 \
  pretrained_models/somali-punctuated-paced-20260802 . --repo-type model --private
```

The container downloads only its roughly 5 GB inference subset from this full
repository into Verda's persistent `/data/models/somali` scratch disk. It uses
a completion marker, so partial Spot downloads are never loaded. Later
scale-to-zero starts reuse it.

## Build and push to Verda Container Registry

Use Verda's private Container Registry (`vccr.io`), not Docker Hub. First create
**Verda** registry credentials in the Verda dashboard (Project → Credentials →
Create credentials → Verda). Then configure the Verda CLI and Docker once:

```bash
verda registry configure
verda registry configure-docker
```

With Docker Desktop (or another Buildx builder) running, build the AMD64 image
locally and push it to the private registry:

```bash
docker buildx build --platform linux/amd64 \
  --file runtime/serverless/Dockerfile.l40s \
  --tag cosyvoice-somali:0.1.0 \
  --load .

verda registry push cosyvoice-somali:0.1.0 \
  --repo cosyvoice-somali --tag 0.1.0
```

`verda registry ls` prints the exact private `vccr.io/...` image reference for
the deployment form. Do not put Verda registry or Hugging Face credentials in
the Dockerfile, image, or source tree.

## Verda deployment

Create a Serverless Container deployment with:

- Image: the `vccr.io/.../cosyvoice-somali:0.1.0` reference from
  `verda registry ls`
- Compute: `1x L40S`; Spot is suitable for this retryable service
- Exposed port: `8000`; health check path: `/health`
- Scaling: one concurrent request per replica; minimum replicas `0`, maximum
  replicas `1` initially
- Included Container disk mounted at `/data`
- Secret environment variable: `HF_TOKEN` with read access to the private model
- Plain environment variable: `HF_MODEL_REPO=lewenberg/somali-punctuated-paced-20260802`

The first request after a brand-new deployment downloads the model. Later cold
starts reload it from `/data`, avoiding another multi-gigabyte transfer.
