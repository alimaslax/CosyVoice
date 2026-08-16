"""Runpod queue worker for Somali pace-controlled synthesis."""

import base64
import time

import runpod
import torch

from runtime.serverless.somali_api import synthesize
from runtime.serverless.model_loader import BACKEND, MODEL_DIR, load_model


_model = None


def initialize_model():
    """Prebuild/load the selected backend before registering for queue work."""
    global _model
    if _model is None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is unavailable')
        print('[somali-queue] prebuilding and loading {} model from {}'.format(BACKEND, MODEL_DIR), flush=True)
        _model = load_model()
        print('[somali-queue] {} model and export cache ready'.format(BACKEND), flush=True)
    return _model


def handler(job):
    global _model
    payload = job.get('input', {})
    text = str(payload.get('text', '')).strip()
    pace = str(payload.get('pace', 'medium')).lower()
    if not text:
        return {'error': 'text is required'}
    if pace not in {'slow', 'medium', 'fast'}:
        return {'error': 'pace must be slow, medium, or fast'}
    initialize_model()
    started = time.monotonic()
    wav = synthesize(_model, text, pace)
    return {
        'audio_base64': base64.b64encode(wav).decode('ascii'),
        'content_type': 'audio/wav',
        'backend': BACKEND,
        'seconds': round(time.monotonic() - started, 3),
    }


if __name__ == '__main__':
    # vLLM cache creation is deliberately part of readiness, not the first
    # request.  A partial network-volume export is rebuilt before Runpod can
    # route work to this replica.
    initialize_model()
    runpod.serverless.start({'handler': handler})
