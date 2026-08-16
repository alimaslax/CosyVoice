"""Select the CosyVoice inference engine without changing model weights."""

import os

from cosyvoice.cli.cosyvoice import AutoModel


MODEL_DIR = os.environ.get('MODEL_DIR', '/runpod-volume/models/somali')
BACKEND = os.environ.get('COSYVOICE_BACKEND', 'torch').strip().lower()


def load_model():
    """Load the Somali checkpoint with the selected execution backend."""
    if BACKEND == 'torch':
        return AutoModel(model_dir=MODEL_DIR, fp16=True)
    if BACKEND == 'vllm':
        # The exported vLLM config identifies this custom CosyVoice model by
        # name, so register the implementation before AutoModel creates the
        # vLLM engine. The checkpoint itself is unchanged.
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)
        return AutoModel(model_dir=MODEL_DIR, load_vllm=True, fp16=False)
    raise ValueError('Unsupported COSYVOICE_BACKEND={!r}; use torch or vllm'.format(BACKEND))
