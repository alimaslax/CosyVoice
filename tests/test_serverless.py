import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cosyvoice.cli.somali_pace import PACE_PRESETS, get_prompt_path
from runtime.serverless import somali_api
from runtime.serverless import bootstrap_model


class FakeModel:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def inference_instruct2(self, text, instruction, prompt_wav, stream=False):
        self.calls.append((text, instruction, prompt_wav, stream))
        yield {'tts_speech': torch.zeros((1, 16))}


class ServerlessTests(unittest.TestCase):
    def test_packaged_prompts_are_available(self):
        for pace in PACE_PRESETS:
            self.assertTrue(get_prompt_path(pace).endswith('{}.flac'.format(pace)))

    def test_synthesis_uses_selected_hidden_prompt(self):
        model = FakeModel()
        wav = somali_api.synthesize(model, 'Tijaabo', 'fast')
        self.assertTrue(wav.startswith(b'RIFF'))
        self.assertEqual(model.calls, [(
            'Tijaabo', PACE_PRESETS['fast']['instruction'], get_prompt_path('fast'), False)])

    def test_endpoint_rejects_invalid_pace(self):
        async def call():
            return await somali_api.synthesize_endpoint(somali_api.SynthesisRequest(text='Tijaabo', pace='invalid'))
        with self.assertRaises(somali_api.HTTPException) as error:
            asyncio.run(call())
        self.assertEqual(error.exception.status_code, 422)

    def test_model_bootstrap_downloads_once_then_reuses_persistent_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / 'data' / 'models' / 'somali'
            lock_dir = model_dir.parent / '.somali-model-download.lock'

            def fake_download(**kwargs):
                destination = Path(kwargs['local_dir'])
                for required in bootstrap_model.REQUIRED_FILES:
                    path = destination / required
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'model')

            with patch.object(bootstrap_model, 'MODEL_DIR', model_dir), \
                    patch.object(bootstrap_model, 'LOCK_DIR', lock_dir), \
                    patch.object(bootstrap_model, 'snapshot_download', side_effect=fake_download) as download, \
                    patch.dict('os.environ', {'HF_TOKEN': 'test-token'}, clear=False):
                bootstrap_model.main()
                self.assertTrue(bootstrap_model.model_is_ready(model_dir))
                self.assertEqual(download.call_count, 1)
                bootstrap_model.main()
                self.assertEqual(download.call_count, 1)
