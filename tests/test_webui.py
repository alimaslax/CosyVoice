import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

import webui


def flac_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2400, dtype=np.float32), 24000, format='FLAC')
    return buffer.getvalue()


class FakeModel:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def inference_instruct2(self, text, instruction, prompt_wav, stream=False):
        self.calls.append((text, instruction, prompt_wav, stream))
        yield {'tts_speech': torch.zeros((1, 4))}


class PaceWebUiTests(unittest.TestCase):
    def test_preset_metadata_matches_training_tiers(self):
        self.assertEqual(
            webui.PACE_PRESETS['Slow · < 120.9 WPM']['instruction'],
            'You are a helpful assistant. Speak slowly and deliberately.<|endofprompt|>')
        self.assertEqual(
            webui.PACE_PRESETS['Medium · 120.9–154.5 WPM']['instruction'],
            'You are a helpful assistant. Speak at a natural, moderate pace.<|endofprompt|>')
        self.assertEqual(
            webui.PACE_PRESETS['Fast · > 154.5 WPM']['instruction'],
            'You are a helpful assistant. Speak at a fast pace.<|endofprompt|>')

    def test_load_preset_prompts_extracts_readable_audio(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / 'dataset'
            cache_dir = Path(temporary_directory) / 'cache'
            presets = {
                'Slow': {'utt': 'slow', 'parquet': 'train/parquet/first.parquet', 'instruction': 'slow'},
                'Medium': {'utt': 'medium', 'parquet': 'train/parquet/first.parquet', 'instruction': 'medium'},
                'Fast': {'utt': 'fast', 'parquet': 'train/parquet/second.parquet', 'instruction': 'fast'},
            }
            for relative_path, rows in {
                'train/parquet/first.parquet': [('slow', flac_bytes()), ('medium', flac_bytes())],
                'train/parquet/second.parquet': [('fast', flac_bytes())],
            }.items():
                parquet_path = dataset_dir / relative_path
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({'utt': [row[0] for row in rows],
                                         'audio_data': [row[1] for row in rows]}), parquet_path)

            with patch.dict(webui.PACE_PRESETS, presets, clear=True):
                prompt_paths = webui.load_preset_prompts(dataset_dir, cache_dir)

            self.assertEqual(set(prompt_paths), {'Slow', 'Medium', 'Fast'})
            for path in prompt_paths.values():
                self.assertEqual(sf.info(path).samplerate, 24000)

    def test_generation_uses_hidden_preset_prompt_and_instruction(self):
        model = FakeModel()
        label = 'Fast · > 154.5 WPM'
        with patch.object(webui, 'instruct_cosyvoice', model), \
                patch.object(webui, 'preset_prompt_wavs', {label: '/tmp/fast.flac'}):
            sample_rate, audio = next(webui.generate_audio('  Tijaabo  ', label))

        self.assertEqual(sample_rate, 24000)
        self.assertEqual(audio.shape, (4,))
        self.assertEqual(model.calls, [(
            'Tijaabo',
            webui.PACE_PRESETS[label]['instruction'],
            '/tmp/fast.flac',
            False,
        )])


if __name__ == '__main__':
    unittest.main()
