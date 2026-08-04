#!/usr/bin/env python3
"""Download a Hugging Face snapshot with an explicit runtime token.

This avoids relying on the Hugging Face CLI's implicit-token behaviour inside
containers.  Set HF_TOKEN in the environment; never pass it on the command
line or commit it to the repository.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import json


def download_direct_https(args: argparse.Namespace, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    collection = {"model": "models", "dataset": "datasets", "space": "spaces"}[args.repo_type]
    request = Request(
        f"https://huggingface.co/api/{collection}/{args.repo_id}",
        headers=headers,
    )
    print(f"Fetching repository manifest: {request.full_url}", flush=True)
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    files = [item["rfilename"] for item in payload.get("siblings", [])]
    if args.allow_pattern:
        files = [
            path for path in files
            if any(fnmatch(path, pattern) for pattern in args.allow_pattern)
        ]
    print(f"Downloading {len(files)} files through direct authenticated HTTPS", flush=True)

    def download(path: str) -> None:
        target = Path(args.local_dir) / path
        if target.is_file() and target.stat().st_size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        repo_prefix = "datasets/" if args.repo_type == "dataset" else ""
        url = f"https://huggingface.co/{repo_prefix}{args.repo_id}/resolve/main/{quote(path, safe='/')}"
        request = Request(url, headers=headers)
        with urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(files)))) as pool:
        list(pool.map(download, files))
    print(f"Download complete: {args.repo_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default="model")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--allow-pattern", action="append", default=None)
    parser.add_argument("--direct-https", action="store_true")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip() if args.token_file else os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN must be set in the environment or --token-file must be provided")

    if args.direct_https:
        download_direct_https(args, token)
        return

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    print(f"Downloading {args.repo_type} snapshot: {args.repo_id}", flush=True)
    try:
        snapshot_download(
            args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            allow_patterns=args.allow_pattern,
            token=token,
        )
    except RepositoryNotFoundError:
        download_direct_https(args, token)
        return
    print(f"Download complete: {args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
