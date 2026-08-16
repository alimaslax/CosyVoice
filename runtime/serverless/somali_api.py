"""L40S-optimised HTTP service for the Somali pace-controlled model."""

import asyncio
import io
import time
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from cosyvoice.cli.somali_pace import PACE_PRESETS, get_prompt_path
from runtime.serverless.model_loader import BACKEND, MODEL_DIR, load_model


state = {'model': None, 'lock': asyncio.Lock()}


def log(message: str) -> None:
    print('[somali-api {}] {}'.format(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message), flush=True)


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    pace: str = Field(default='medium')


def synthesize(model, text: str, pace: str) -> bytes:
    preset = PACE_PRESETS[pace]
    chunks = [item['tts_speech'] for item in model.inference_instruct2(
        text.strip(), preset['instruction'], get_prompt_path(pace), stream=False)]
    if not chunks:
        raise RuntimeError('CosyVoice did not produce audio')
    buffer = io.BytesIO()
    torchaudio.save(buffer, torch.cat(chunks, dim=1), model.sample_rate, format='wav')
    return buffer.getvalue()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not torch.cuda.is_available():
        log('fatal: CUDA GPU is unavailable')
        raise RuntimeError('A CUDA GPU is required for this service')
    log('CUDA available: {}; loading {} model from {}'.format(torch.cuda.get_device_name(0), BACKEND, MODEL_DIR))
    started = time.monotonic()
    try:
        state['model'] = load_model()
    except Exception as error:
        log('fatal model-load error after {:.1f}s: {!r}'.format(time.monotonic() - started, error))
        raise
    log('model loaded in {:.1f}s; service is ready'.format(time.monotonic() - started))
    yield
    log('service shutdown requested')
    state['model'] = None


app = FastAPI(title='Somali Pace TTS', lifespan=lifespan)


@app.get('/health')
def health():
    if state['model'] is None:
        raise HTTPException(status_code=503, detail='model is loading')
    return {'status': 'ready', 'paces': list(PACE_PRESETS)}


@app.get('/ping', include_in_schema=False)
def ping():
    """Runpod load-balancer readiness probe.

    A 204 keeps a newly started worker out of traffic while the model downloads
    and loads; Runpod begins routing only after this endpoint returns 200.
    """
    if state['model'] is None:
        return Response(status_code=204)
    return Response(status_code=200)


@app.post('/synthesize', responses={200: {'content': {'audio/wav': {}}}})
async def synthesize_endpoint(request: SynthesisRequest):
    pace = request.pace.lower()
    if pace not in PACE_PRESETS:
        raise HTTPException(status_code=422, detail='pace must be slow, medium, or fast')
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail='text cannot be blank')
    if state['model'] is None:
        raise HTTPException(status_code=503, detail='model is loading')
    async with state['lock']:
        wav = await asyncio.to_thread(synthesize, state['model'], text, pace)
    return Response(wav, media_type='audio/wav', headers={
        'Content-Disposition': 'inline; filename="somali-{}.wav"'.format(pace),
    })
