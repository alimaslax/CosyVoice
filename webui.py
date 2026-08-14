# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""A small, pace-focused WebUI for the locally fine-tuned Somali model."""

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import gradio as gr
import pyarrow.parquet as pq
import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging


PACE_PRESETS = {
    'Slow · < 120.9 WPM': {
        'utt': 'omar_5d75f8fa2d39a60c',
        'parquet': 'train/parquet/parquet_000000000.tar',
        'instruction': 'You are a helpful assistant. Speak slowly and deliberately.<|endofprompt|>',
    },
    'Medium · 120.9–154.5 WPM': {
        'utt': 'omar_0c1a18d5f8711884',
        'parquet': 'train/parquet/parquet_000000000.tar',
        'instruction': 'You are a helpful assistant. Speak at a natural, moderate pace.<|endofprompt|>',
    },
    'Fast · > 154.5 WPM': {
        'utt': 'omar_9044bab1effeadc7',
        'parquet': 'train/parquet/parquet_000000001.tar',
        'instruction': 'You are a helpful assistant. Speak at a fast pace.<|endofprompt|>',
    },
}

DEFAULT_DATASET_DIR = ROOT_DIR / 'data' / 'omar-cosyvoice3-punctuated-paced-v1'
DEFAULT_PROMPT_CACHE_DIR = Path(tempfile.gettempdir()) / 'cosyvoice-preset-prompts'

instruct_cosyvoice = None
preset_prompt_wavs = {}


def _is_readable_prompt(path: Path) -> bool:
    """Return whether a cached prompt is a usable 24 kHz audio file."""
    try:
        return path.is_file() and torchaudio.info(str(path)).sample_rate == 24000
    except RuntimeError:
        return False


def _write_prompt(cache_path: Path, audio_data: bytes) -> None:
    """Atomically write one embedded FLAC to the runtime cache."""
    fd, temporary_path = tempfile.mkstemp(dir=cache_path.parent, suffix='.flac')
    try:
        with os.fdopen(fd, 'wb') as file:
            file.write(audio_data)
        os.replace(temporary_path, cache_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def load_preset_prompts(dataset_dir: Path = DEFAULT_DATASET_DIR,
                        cache_dir: Path = DEFAULT_PROMPT_CACHE_DIR) -> dict[str, str]:
    """Extract the three embedded pace references and return their cached paths.

    The dataset's ``wav`` values point at the machine that built the dataset, so
    the only portable source of the prompt audio is its ``audio_data`` Parquet
    column. Cached files are validated on every startup and regenerated when
    needed.
    """
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = {
        label: cache_dir / '{}.flac'.format(preset['utt'])
        for label, preset in PACE_PRESETS.items()
    }
    missing_labels = {
        label for label, path in prompt_paths.items()
        if not _is_readable_prompt(path)
    }
    if not missing_labels:
        return {label: str(path) for label, path in prompt_paths.items()}

    wanted_by_parquet = defaultdict(dict)
    for label in missing_labels:
        preset = PACE_PRESETS[label]
        wanted_by_parquet[preset['parquet']][preset['utt']] = label

    for relative_parquet, wanted_utts in wanted_by_parquet.items():
        parquet_path = dataset_dir / relative_parquet
        if not parquet_path.is_file():
            raise RuntimeError('Pace dataset Parquet file is missing: {}'.format(parquet_path))
        found_utts = set()
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=['utt', 'audio_data'], batch_size=64):
            utterances = batch.column('utt').to_pylist()
            audio_items = batch.column('audio_data').to_pylist()
            for utterance, audio_data in zip(utterances, audio_items):
                label = wanted_utts.get(utterance)
                if label is None:
                    continue
                _write_prompt(prompt_paths[label], bytes(audio_data))
                if not _is_readable_prompt(prompt_paths[label]):
                    raise RuntimeError('Extracted prompt is not valid 24 kHz audio: {}'.format(utterance))
                found_utts.add(utterance)
            if found_utts == set(wanted_utts):
                break
        unresolved = set(wanted_utts) - found_utts
        if unresolved:
            raise RuntimeError('Pace prompt utterance(s) not found in {}: {}'.format(
                parquet_path, ', '.join(sorted(unresolved))))

    return {label: str(path) for label, path in prompt_paths.items()}


def generate_audio(tts_text: str, pace_label: str):
    if not tts_text or not tts_text.strip():
        raise gr.Error('Enter text to synthesize.')
    if pace_label not in PACE_PRESETS:
        raise gr.Error('Select a speaking pace.')
    if instruct_cosyvoice is None or pace_label not in preset_prompt_wavs:
        raise gr.Error('The pace presets are not ready. Restart the WebUI.')

    preset = PACE_PRESETS[pace_label]
    logging.info('get pace-controlled inference request: %s', pace_label)
    for output in instruct_cosyvoice.inference_instruct2(
            tts_text.strip(),
            preset['instruction'],
            preset_prompt_wavs[pace_label],
            stream=False):
        yield (instruct_cosyvoice.sample_rate, output['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks(title='Somali Pace TTS') as demo:
        gr.Markdown('## Somali text-to-speech')
        gr.Markdown('Choose a pace. The Omar voice reference is selected automatically.')
        tts_text = gr.Textbox(
            label='Text to Synthesize',
            lines=4,
            placeholder='Enter the Somali text you want the model to speak...',
        )
        pace_label = gr.Radio(
            choices=list(PACE_PRESETS),
            value='Medium · 120.9–154.5 WPM',
            label='Speaking Pace',
        )
        generate_button = gr.Button('Generate Audio', variant='primary')
        audio_output = gr.Audio(label='Generated Audio', autoplay=True)
        generate_button.click(generate_audio, inputs=[tts_text, pace_label], outputs=audio_output)
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name=args.server_name, server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--server_name', type=str, default='127.0.0.1')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--instruct_model_dir', type=str, default=None,
                        help='optional instruction-capable model used for pace control')
    args = parser.parse_args()

    dataset_dir = Path(os.environ.get('COSYVOICE_PACE_DATASET_DIR', DEFAULT_DATASET_DIR))
    cache_dir = Path(os.environ.get('COSYVOICE_PRESET_CACHE_DIR', DEFAULT_PROMPT_CACHE_DIR))
    preset_prompt_wavs = load_preset_prompts(dataset_dir, cache_dir)

    cosyvoice = AutoModel(model_dir=args.model_dir)
    instruct_cosyvoice = cosyvoice
    if args.instruct_model_dir:
        logging.info('loading pace-control model from %s', args.instruct_model_dir)
        instruct_cosyvoice = AutoModel(model_dir=args.instruct_model_dir)
        if instruct_cosyvoice.sample_rate != cosyvoice.sample_rate:
            raise ValueError('main and instruction models must use the same sample rate')
    main()
