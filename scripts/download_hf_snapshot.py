#!/usr/bin/env python3
"""Download a Hugging Face snapshot with an explicit runtime token.

This avoids relying on the Hugging Face CLI's implicit-token behaviour inside
containers.  Set HF_TOKEN in the environment; never pass it on the command
line or commit it to the repository.
"""

from __future__ import annotations

import argparse
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default="model")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--allow-pattern", action="append", default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN must be set in the environment")

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo_type} snapshot: {args.repo_id}", flush=True)
    snapshot_download(
        args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        allow_patterns=args.allow_pattern,
        token=token,
    )
    print(f"Download complete: {args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
