#!/usr/bin/env python3
"""Download the private Omar ASR dataset without the Hub snapshot API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="lewenberg/omar-somali-asr")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN must be set")
    headers = {"Authorization": f"Bearer {token}"}
    base = f"https://huggingface.co/datasets/{args.repo}/resolve/main"
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(relative: str) -> None:
        target = args.output / relative
        if target.is_file() and target.stat().st_size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        response = requests.get(f"{base}/{relative}", headers=headers, stream=True, timeout=120)
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        temporary.replace(target)

    files: list[str] = ["README.md", "dataset_summary.json"]
    for split in ("train", "dev"):
        metadata = f"{split}/metadata.jsonl"
        fetch(metadata)
        files.extend([metadata, f"{split}/manifest.jsonl"])
        for line in (args.output / metadata).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            files.append(f"{split}/audio/{row['utt']}.flac")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, path) for path in sorted(set(files))]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index % 100 == 0 or index == len(futures):
                print(f"Downloaded {index}/{len(futures)} files", flush=True)


if __name__ == "__main__":
    main()
