"""Download the inference model to Verda persistent storage exactly once."""

import os
import shutil
import time
import uuid
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_DIR = Path(os.environ.get('MODEL_DIR', '/data/models/somali'))
MODEL_REPO = os.environ.get('HF_MODEL_REPO', 'lewenberg/somali-punctuated-paced-20260802')
LOCK_DIR = MODEL_DIR.parent / '.somali-model-download.lock'
MARKER = '.cosyvoice-ready'
REQUIRED_FILES = ('cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt', 'campplus.onnx',
                  'speech_tokenizer_v3.onnx', 'CosyVoice-BlankEN/model.safetensors')
ALLOW_PATTERNS = list(REQUIRED_FILES) + [
    'config.json', 'configuration.json',
    'CosyVoice-BlankEN/config.json', 'CosyVoice-BlankEN/generation_config.json',
    'CosyVoice-BlankEN/tokenizer_config.json', 'CosyVoice-BlankEN/merges.txt',
    'CosyVoice-BlankEN/vocab.json',
]


def model_is_ready(path: Path) -> bool:
    return (path / MARKER).is_file() and all((path / required).is_file() for required in REQUIRED_FILES)


def acquire_lock() -> None:
    while True:
        try:
            LOCK_DIR.mkdir()
            return
        except FileExistsError:
            if model_is_ready(MODEL_DIR):
                return
            if time.time() - LOCK_DIR.stat().st_mtime > 7200:
                shutil.rmtree(LOCK_DIR, ignore_errors=True)
                continue
            time.sleep(5)


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if model_is_ready(MODEL_DIR):
        return
    if not os.environ.get('HF_TOKEN'):
        raise RuntimeError('HF_TOKEN must be configured as a Verda secret on first startup')
    acquire_lock()
    if model_is_ready(MODEL_DIR):
        return
    staging_dir = MODEL_DIR.parent / '.somali-model-{}'.format(uuid.uuid4().hex)
    try:
        snapshot_download(repo_id=MODEL_REPO, repo_type='model', local_dir=staging_dir,
                          token=os.environ['HF_TOKEN'], allow_patterns=ALLOW_PATTERNS)
        if not all((staging_dir / required).is_file() for required in REQUIRED_FILES):
            raise RuntimeError('Downloaded model is incomplete: {}'.format(MODEL_REPO))
        (staging_dir / MARKER).write_text('ready\n', encoding='utf-8')
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        staging_dir.replace(MODEL_DIR)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(LOCK_DIR, ignore_errors=True)


if __name__ == '__main__':
    main()
