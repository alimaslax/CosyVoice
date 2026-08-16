"""Inference-safe placeholder for pyworld.

CosyVoice imports its training dataset processor while resolving the model
configuration, even though serverless synthesis never calls F0 extraction.
Avoid compiling pyworld in the vLLM runtime; fail clearly if a training-only
function is ever invoked.
"""

def _training_only(*_args, **_kwargs):
    raise RuntimeError('pyworld is available only in the CosyVoice training runtime')


harvest = _training_only
dio = _training_only
stonemask = _training_only
