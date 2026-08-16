"""Select the CosyVoice inference engine without changing model weights."""

import os

from cosyvoice.cli.cosyvoice import AutoModel


MODEL_DIR = os.environ.get('MODEL_DIR', '/runpod-volume/models/somali')
BACKEND = os.environ.get('COSYVOICE_BACKEND', 'torch').strip().lower()
VLLM_RUNTIME_MARKER = '/etc/cosyvoice-vllm-runtime'


def validate_runtime_backend() -> None:
    """Refuse the invalid torch-on-vLLM-container combination.

    ``COSYVOICE_BACKEND`` is exposed as an endpoint environment variable and
    can be overridden accidentally in the Runpod UI.  The vLLM image carries
    an immutable marker so it can never silently fall back to the PyTorch
    loader, whose CUDA runtime is not a quality-control configuration.
    """
    if os.path.exists(VLLM_RUNTIME_MARKER) and BACKEND != 'vllm':
        raise RuntimeError(
            'The vLLM container requires COSYVOICE_BACKEND=vllm; '
            'use the separate PyTorch control image for COSYVOICE_BACKEND=torch.')


def load_model():
    """Load the Somali checkpoint with the selected execution backend."""
    validate_runtime_backend()
    if BACKEND == 'torch':
        return AutoModel(model_dir=MODEL_DIR, fp16=True)
    if BACKEND == 'vllm':
        # The exported vLLM config identifies this custom CosyVoice model by
        # name, so register the implementation before AutoModel creates the
        # vLLM engine. The checkpoint itself is unchanged.
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)
        model = AutoModel(model_dir=MODEL_DIR, load_vllm=True, fp16=False)
        # ``load_vllm`` creates this export as a side effect.  Assert the
        # durable completion marker before the worker can accept a job.
        from cosyvoice.utils.file_utils import vllm_export_is_ready
        export_path = os.path.join(MODEL_DIR, 'vllm')
        if not vllm_export_is_ready(export_path):
            raise RuntimeError('vLLM export cache was not committed: {}'.format(export_path))
        return model
    raise ValueError('Unsupported COSYVOICE_BACKEND={!r}; use torch or vllm'.format(BACKEND))
