#!/usr/bin/env python3
"""Archive completed CosyVoice checkpoints and retain a rolling local window."""

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_state(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prune_uploaded_checkpoints(watch: Path, state: dict[str, dict[str, object]], keep_local: int) -> None:
    """Keep the newest successfully archived checkpoints in each model directory."""
    if keep_local < 1:
        return

    for directory in {checkpoint.parent for checkpoint in watch.rglob("*.pt")}:
        checkpoints = sorted(
            directory.glob("*.pt"), key=lambda checkpoint: checkpoint.stat().st_mtime, reverse=True
        )
        for checkpoint in checkpoints[keep_local:]:
            relative = checkpoint.relative_to(watch).as_posix()
            archived = state.get(relative)
            if not archived or archived.get("size") != checkpoint.stat().st_size:
                continue
            logging.info("pruning archived local checkpoint %s", checkpoint)
            checkpoint.unlink()
            state.pop(relative, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--settle-seconds", type=int, default=45)
    parser.add_argument("--keep-local", type=int, default=16)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.watch.mkdir(parents=True, exist_ok=True)
    state_path = args.watch / "archive-upload-state.json"
    state = load_state(state_path)
    api = HfApi(token=token)

    while True:
        for checkpoint in sorted(args.watch.rglob("*.pt")):
            if time.time() - checkpoint.stat().st_mtime < args.settle_seconds:
                continue
            relative = checkpoint.relative_to(args.watch).as_posix()
            size = checkpoint.stat().st_size
            prior = state.get(relative)
            if prior and prior.get("size") == size:
                continue
            checksum = digest(checkpoint)
            if prior and prior.get("sha256") == checksum:
                prior["size"] = size
                save_state(state_path, state)
                continue
            remote_path = f"training-runs/{args.run_name}/checkpoints/{relative}"
            logging.info("uploading %s (%d bytes) to %s", checkpoint, size, remote_path)
            try:
                api.upload_file(
                    path_or_fileobj=str(checkpoint),
                    path_in_repo=remote_path,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message=f"Archive {args.run_name}: {relative}",
                )
            except Exception:
                # A network or Hub-side failure must never stop the watcher.
                # Leave the local checkpoint intact and retry on the next poll.
                logging.exception("archive failed for %s; retaining it for retry", checkpoint)
                continue
            state[relative] = {"sha256": checksum, "size": size, "remote_path": remote_path}
            save_state(state_path, state)
            logging.info("archived %s", checkpoint)
        prune_uploaded_checkpoints(args.watch, state, args.keep_local)
        save_state(state_path, state)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
