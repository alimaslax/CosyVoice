"""Runpod queue worker for Somali pace-controlled synthesis."""

import base64
import os
import time

import runpod
import torch

from cosyvoice.cli.cosyvoice import AutoModel
from runtime.serverless.somali_api import synthesize


MODEL_DIR = os.environ.get('MODEL_DIR', '/runpod-volume/models/somali')
_model = None


def handler(job):
    global _model
    payload = job.get('input', {})
    text = str(payload.get('text', '')).strip()
    pace = str(payload.get('pace', 'medium')).lower()
    if not text:
        return {'error': 'text is required'}
    if pace not in {'slow', 'medium', 'fast'}:
        return {'error': 'pace must be slow, medium, or fast'}
    if _model is None:
        if not torch.cuda.is_available():
            return {'error': 'CUDA GPU is unavailable'}
        print('[somali-queue] loading model from {}'.format(MODEL_DIR), flush=True)
        _model = AutoModel(model_dir=MODEL_DIR, fp16=True)
        print('[somali-queue] model ready', flush=True)
    started = time.monotonic()
    wav = synthesize(_model, text, pace)
    return {
        'audio_base64': base64.b64encode(wav).decode('ascii'),
        'content_type': 'audio/wav',
        'seconds': round(time.monotonic() - started, 3),
    }


runpod.serverless.start({'handler': handler})
