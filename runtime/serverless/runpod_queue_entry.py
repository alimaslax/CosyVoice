"""Python-only Runpod queue-worker startup.

Runpod reports shell entrypoint failures only as exit code 127.  Starting with
Python removes the image-shell dependency, performs the persistent-model
bootstrap, and then replaces itself with the queue handler.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault('NUMBA_CACHE_DIR', '/data/cache/numba')
    os.environ.setdefault('MPLCONFIGDIR', '/data/cache/matplotlib')
    Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

    from runtime.serverless.bootstrap_model import main as bootstrap_model

    print('[somali-entrypoint] starting model-cache bootstrap', flush=True)
    bootstrap_model()
    print('[somali-entrypoint] model cache bootstrap complete; starting Runpod queue handler', flush=True)
    os.execv(sys.executable, [sys.executable, '/app/runtime/serverless/queue_handler.py'])


if __name__ == '__main__':
    main()
