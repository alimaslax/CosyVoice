"""Download the inference model to Verda persistent storage exactly once."""

import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_DIR = Path(os.environ.get('MODEL_DIR', '/data/models/somali'))
MODEL_REPO = os.environ.get('HF_MODEL_REPO', 'lewenberg/somali-punctuated-paced-20260802')
LOCK_DIR = MODEL_DIR.parent / '.somali-model-download.lock'
LOCK_STALE_SECONDS = int(os.environ.get('MODEL_DOWNLOAD_LOCK_STALE_SECONDS', '120'))
MARKER = '.cosyvoice-ready'
REQUIRED_FILES = ('cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt', 'campplus.onnx',
                  'speech_tokenizer_v3.onnx', 'CosyVoice-BlankEN/model.safetensors')
ALLOW_PATTERNS = list(REQUIRED_FILES) + [
    'config.json', 'configuration.json',
    'CosyVoice-BlankEN/config.json', 'CosyVoice-BlankEN/generation_config.json',
    'CosyVoice-BlankEN/tokenizer_config.json', 'CosyVoice-BlankEN/merges.txt',
    'CosyVoice-BlankEN/vocab.json',
]


def log(message: str) -> None:
    """Emit a timestamped line immediately for Verda Replica logs."""
    print('[somali-bootstrap {}] {}'.format(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message),
          flush=True)


def model_is_ready(path: Path) -> bool:
    return (path / MARKER).is_file() and all((path / required).is_file() for required in REQUIRED_FILES)


def acquire_lock() -> None:
    reported_wait = False
    while True:
        try:
            LOCK_DIR.mkdir()
            log('download lock acquired')
            return
        except FileExistsError:
            if model_is_ready(MODEL_DIR):
                log('model became ready while waiting for download lock')
                return
            if not reported_wait:
                log('another replica holds the download lock; waiting for its persistent /data cache')
                reported_wait = True
            lock_age = time.time() - LOCK_DIR.stat().st_mtime
            if lock_age > LOCK_STALE_SECONDS:
                log('removing stale download lock {:.0f}s old (threshold {}s)'.format(
                    lock_age, LOCK_STALE_SECONDS))
                shutil.rmtree(LOCK_DIR, ignore_errors=True)
                continue
            time.sleep(5)


def staging_summary(staging_dir: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for path in staging_dir.rglob('*'):
        if path.is_file():
            files += 1
            total_bytes += path.stat().st_size
    return files, total_bytes


def download_model(staging_dir: Path) -> None:
    """Download with an independent heartbeat because HF can be quiet for large files."""
    started = time.monotonic()
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(10):
            files, total_bytes = staging_summary(staging_dir)
            log('download progress: {} files, {:.2f} GiB staged, {:.0f}s elapsed'.format(
                files, total_bytes / (1024 ** 3), time.monotonic() - started))

    reporter = threading.Thread(target=heartbeat, name='model-download-progress', daemon=True)
    reporter.start()
    try:
        snapshot_download(repo_id=MODEL_REPO, repo_type='model', local_dir=staging_dir,
                          token=os.environ['HF_TOKEN'], allow_patterns=ALLOW_PATTERNS)
    finally:
        finished.set()
        reporter.join(timeout=1)
    files, total_bytes = staging_summary(staging_dir)
    log('download finished: {} files, {:.2f} GiB staged, {:.0f}s elapsed'.format(
        files, total_bytes / (1024 ** 3), time.monotonic() - started))


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    log('startup: model repo={} target={}'.format(MODEL_REPO, MODEL_DIR))
    if model_is_ready(MODEL_DIR):
        log('persistent model cache is ready; skipping Hugging Face download')
        return
    if not os.environ.get('HF_TOKEN'):
        log('fatal: HF_TOKEN is missing')
        raise RuntimeError('HF_TOKEN must be configured as a Verda secret on first startup')
    log('persistent model cache is absent; beginning first-time download')
    acquire_lock()
    if model_is_ready(MODEL_DIR):
        log('persistent model cache is ready after lock acquisition; skipping download')
        return
    staging_dir = MODEL_DIR.parent / '.somali-model-{}'.format(uuid.uuid4().hex)
    try:
        log('downloading {} selected inference files into {}'.format(len(ALLOW_PATTERNS), staging_dir))
        download_model(staging_dir)
        if not all((staging_dir / required).is_file() for required in REQUIRED_FILES):
            raise RuntimeError('Downloaded model is incomplete: {}'.format(MODEL_REPO))
        log('download validation passed; writing completion marker')
        (staging_dir / MARKER).write_text('ready\n', encoding='utf-8')
        if MODEL_DIR.exists():
            log('replacing incomplete previous model cache')
            shutil.rmtree(MODEL_DIR)
        staging_dir.replace(MODEL_DIR)
        log('persistent model cache committed atomically at {}'.format(MODEL_DIR))
    except Exception:
        log('fatal bootstrap error:\n{}'.format(traceback.format_exc()))
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(LOCK_DIR, ignore_errors=True)


if __name__ == '__main__':
    main()
