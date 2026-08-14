#!/usr/bin/env python3
"""Generate one Somali WAV at a selected trained speaking pace, then exit."""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.cli.somali_pace import PACE_PRESETS, get_prompt_path


DEFAULT_MODEL_DIR = ROOT_DIR / 'pretrained_models' / 'somali-punctuated-paced-20260802'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate one Somali WAV using a hidden Omar pace reference, then exit.')
    parser.add_argument('--pace', choices=PACE_PRESETS, default='medium',
                        help='trained speaking pace (default: medium)')
    parser.add_argument('--text', required=True, help='Somali text to synthesize')
    parser.add_argument('--output', type=Path, default=None,
                        help='output WAV path (default: outputs/somali-<pace>.wav)')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                        help='local CosyVoice model directory')
    parser.add_argument('--instruct-model-dir', type=Path, default=None,
                        help='optional separate instruction-capable model directory')
    return parser.parse_args()


def main():
    args = parse_args()
    text = args.text.strip()
    if not text:
        raise SystemExit('--text cannot be empty')

    output_path = args.output or ROOT_DIR / 'outputs' / 'somali-{}.wav'.format(args.pace)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModel(model_dir=str(args.model_dir))
    instruct_model = model
    if args.instruct_model_dir:
        instruct_model = AutoModel(model_dir=str(args.instruct_model_dir))
        if instruct_model.sample_rate != model.sample_rate:
            raise RuntimeError('main and instruction models must use the same sample rate')

    preset = PACE_PRESETS[args.pace]
    chunks = [output['tts_speech'] for output in instruct_model.inference_instruct2(
        text,
        preset['instruction'],
        get_prompt_path(args.pace),
        stream=False,
    )]
    if not chunks:
        raise RuntimeError('CosyVoice did not return any audio')
    torchaudio.save(str(output_path), torch.cat(chunks, dim=1), instruct_model.sample_rate)
    print('Wrote {} ({})'.format(output_path, preset['label']))


if __name__ == '__main__':
    main()
