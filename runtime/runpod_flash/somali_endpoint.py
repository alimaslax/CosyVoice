"""Runpod Flash deployment for the pre-built Somali CosyVoice HTTP image."""

import os

from runpod_flash import Endpoint, GpuGroup
from runpod_flash.core.resources.template import PodTemplate


IMAGE = "ghcr.io/alimaslax/cosyvoice-somali:0.1.7"
MODEL_REPO = "lewenberg/somali-punctuated-paced-20260802"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError("{} must be set in the repository .env file".format(name))
    return value


class PrebuiltHttpEndpoint(Endpoint):
    """Expose a pre-built HTTP image to Flash's deployment manifest builder."""

    @property
    def is_client(self) -> bool:
        # ``Endpoint(image=...)`` is normally a direct client.  This service
        # is a deployed load-balanced worker, declared by the route below.
        return False


# Declaring one route makes Flash emit a load-balanced resource.  The worker
# itself is the supplied image, which serves its native FastAPI routes.
somali_tts = PrebuiltHttpEndpoint(
    name="cosyvoice-somali",
    image=IMAGE,
    gpu=GpuGroup.AMPERE_24,
    workers=(0, 1),
    idle_timeout=60,
    max_concurrency=1,
    env={
        "HF_TOKEN": required_env("HF_TOKEN"),
        "HF_MODEL_REPO": MODEL_REPO,
        # Runpod mounts the endpoint's network volume here.  This keeps the
        # downloaded Hugging Face model across serverless scale-to-zero.
        "MODEL_DIR": "/runpod-volume/models/somali",
        "PORT": "8000",
        "PORT_HEALTH": "8000",
    },
    template=PodTemplate(
        containerDiskInGb=20,
        ports="8000/http",
    ),
)


@somali_tts.get('/_flash-deployment-marker')
async def flash_deployment_marker():
    """Declare the load-balanced Flash resource; the custom image owns HTTP."""
    return {'status': 'configured'}
