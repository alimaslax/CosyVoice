"""L40S-optimised HTTP service for the Somali pace-controlled model."""

import asyncio
import io
import os
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.cli.somali_pace import PACE_PRESETS, get_prompt_path


MODEL_DIR = os.environ.get('MODEL_DIR', '/data/models/somali')
state = {'model': None, 'lock': asyncio.Lock()}


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
        raise RuntimeError('A CUDA GPU is required for this service')
    state['model'] = AutoModel(model_dir=MODEL_DIR, fp16=True)
    yield
    state['model'] = None


app = FastAPI(title='Somali Pace TTS', lifespan=lifespan)


@app.get('/health')
def health():
    if state['model'] is None:
        raise HTTPException(status_code=503, detail='model is loading')
    return {'status': 'ready', 'paces': list(PACE_PRESETS)}


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
