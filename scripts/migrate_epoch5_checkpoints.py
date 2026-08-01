#!/usr/bin/env python3
"""Copy the required epoch-5 checkpoints into the active training dataset."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


SOURCE_REPO = "lewenberg/so-audio-cosyvoice3"
DESTINATION_REPO = "lewenberg/omar-somali-asr"
FILES = (
    "training-runs/somali-scratch-lr-decay-20260728/checkpoints/llm/epoch_5_whole.pt",
    "training-runs/flow-checkpoints-20260728/epoch_5_whole.pt",
)


def main() -> None:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    for path in FILES:
        local = hf_hub_download(SOURCE_REPO, path, repo_type="dataset", token=token)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=path,
            repo_id=DESTINATION_REPO,
            repo_type="dataset",
            commit_message=f"Migrate required resume checkpoint: {Path(path).name}",
        )
        print(f"Migrated {path}", flush=True)


if __name__ == "__main__":
    main()
