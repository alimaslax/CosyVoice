"""Runpod Flash deployment for the dedicated RTX 4090 Somali TTS worker."""

import os

from runpod_flash import Endpoint, GpuType
from runpod_flash.core.resources.template import PodTemplate


IMAGE = "ghcr.io/alimaslax/cosyvoice-somali:0.2.3"
MODEL_REPO = "lewenberg/somali-punctuated-paced-20260802"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError("{} must be set in the repository .env file".format(name))
    return value


# This is Flash's pre-built queue-worker pattern. The container owns the
# Runpod queue handler; Flash owns the deployment, GPU selection, and scaling.
somali_tts = Endpoint(
    name="cosyvoice-somali-4090",
    image=IMAGE,
    # Do not use a mixed 24 GB group: that can assign the slower L4, A5000,
    # or 3090.  This endpoint is pinned to the 24 GB RTX 4090.
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 1),
    idle_timeout=5,
    max_concurrency=1,
    env={
        "HF_TOKEN": required_env("HF_TOKEN"),
        "HF_MODEL_REPO": MODEL_REPO,
        # Runpod mounts the endpoint's network volume here.  This keeps the
        # downloaded Hugging Face model across serverless scale-to-zero.
        "MODEL_DIR": "/runpod-volume/models/somali",
        "PORT": "8000",
        "PORT_HEALTH": "8000",
        # Runpod queue-mode handler used by scripts/runpod_tts.sh.
        "RUNPOD_QUEUE_MODE": "1",
    },
    template=PodTemplate(
        containerDiskInGb=20,
        ports="8000/http",
    ),
)
