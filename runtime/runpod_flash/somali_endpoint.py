"""Runpod Flash deployment for the pre-built Somali CosyVoice HTTP image."""

import os

from runpod_flash import Endpoint, GpuGroup
from runpod_flash.core.resources.template import PodTemplate


IMAGE = "vccr.io/086368d3-195a-40fb-ac58-ef40f674c8b3/cosyvoice-somali:0.1.6"
MODEL_REPO = "lewenberg/somali-punctuated-paced-20260802"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError("{} must be set in the repository .env file".format(name))
    return value


# A pre-built image is Flash client mode: Flash provisions it as a Serverless
# endpoint and preserves its native FastAPI routes (/health and /synthesize).
somali_tts = Endpoint(
    name="cosyvoice-somali",
    image=IMAGE,
    gpu=GpuGroup.AMPERE_24,
    workers=(0, 1),
    idle_timeout=60,
    max_concurrency=1,
    env={
        "HF_TOKEN": required_env("HF_TOKEN"),
        "HF_MODEL_REPO": MODEL_REPO,
    },
    template=PodTemplate(
        # Runpod stores the VCCR username/password as a registry credential;
        # Flash needs its Runpod credential ID, never the password itself.
        containerRegistryAuthId=required_env("RUNPOD_CONTAINER_REGISTRY_AUTH_ID"),
        containerDiskInGb=20,
    ),
)
