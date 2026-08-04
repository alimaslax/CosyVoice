#!/usr/bin/env python3
"""Download one large Hugging Face file over authenticated parallel HTTPS ranges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def open_with_retries(request: Request, timeout: int = 120):
    for attempt in range(8):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as error:
            if error.code not in {404, 429, 500, 502, 503, 504} or attempt == 7:
                raise
            delay = min(30, 2 ** attempt)
            print(f"HTTP {error.code}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("path")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--size", type=int)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        parser.error("token file is empty")
    headers = {"Authorization": f"Bearer {token}"}
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    url = f"https://huggingface.co/{prefix}{args.repo_id}/resolve/main/{quote(args.path, safe='/')}"

    if args.size is not None:
        total = args.size
    else:
        head = Request(url, headers=headers, method="HEAD")
        with open_with_retries(head) as response:
            total = int(response.headers["Content-Length"])
    workers = max(1, min(args.workers, total))
    chunk_size = (total + workers - 1) // workers
    ranges = [(start, min(total - 1, start + chunk_size - 1)) for start in range(0, total, chunk_size)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".parallel")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    os.ftruncate(fd, total)

    def fetch(byte_range: tuple[int, int]) -> int:
        start, end = byte_range
        request = Request(url, headers={**headers, "Range": f"bytes={start}-{end}"})
        offset = start
        with open_with_retries(request) as response:
            if response.status != 206:
                raise RuntimeError(f"expected HTTP 206 for {start}-{end}, received {response.status}")
            while block := response.read(1024 * 1024):
                os.pwrite(fd, block, offset)
                offset += len(block)
        if offset != end + 1:
            raise RuntimeError(f"short range {start}-{end}: stopped at {offset}")
        return end - start + 1

    try:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, byte_range) for byte_range in ranges]
            for future in as_completed(futures):
                completed += future.result()
                print(f"Downloaded {completed}/{total} bytes", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)

    if temporary.stat().st_size != total:
        raise RuntimeError(f"size mismatch: {temporary.stat().st_size} != {total}")
    temporary.replace(args.output)
    print(f"Download complete: {args.output} ({total} bytes)", flush=True)


if __name__ == "__main__":
    main()
