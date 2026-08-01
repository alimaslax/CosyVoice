#!/usr/bin/env python3
"""Convert the Omar ASR Hub snapshot into feature-enriched CosyVoice shards."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_manifest(source: Path, output: Path, split: str) -> None:
    rows = []
    metadata = source / split / "metadata.jsonl"
    for line in metadata.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        utt = row["utt"]
        text = row["text"].strip()
        audio = source / split / "audio" / f"{utt}.flac"
        if text and audio.is_file():
            rows.append((utt, row.get("speaker", "omar"), text, audio))
    if not rows:
        raise RuntimeError(f"No valid records in {metadata}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "wav.scp").write_text("".join(f"{utt} {audio}\n" for utt, _, _, audio in rows), encoding="utf-8")
    (output / "text").write_text("".join(f"{utt} {text}\n" for utt, _, text, _ in rows), encoding="utf-8")
    (output / "utt2spk").write_text("".join(f"{utt} {speaker}\n" for utt, speaker, _, _ in rows), encoding="utf-8")
    (output / "instruct").write_text("".join(f"{utt} {INSTRUCT}\n" for utt, _, _, _ in rows), encoding="utf-8")
    speakers: dict[str, list[str]] = {}
    for utt, speaker, _, _ in rows:
        speakers.setdefault(speaker, []).append(utt)
    (output / "spk2utt").write_text("".join(f"{speaker} {' '.join(utts)}\n" for speaker, utts in speakers.items()), encoding="utf-8")
    print(f"Prepared {split}: {len(rows)} utterances", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.replace:
            raise SystemExit(f"Output exists: {args.output}; use --replace")
        shutil.rmtree(args.output)
    for split in ("train", "dev"):
        destination = args.output / split
        write_manifest(args.source, destination, split)
        run([sys.executable, "tools/extract_embedding.py", "--dir", str(destination),
             "--onnx_path", str(args.model_dir / "campplus.onnx")])
        run([sys.executable, "tools/extract_speech_token.py", "--dir", str(destination),
             "--onnx_path", str(args.model_dir / "speech_tokenizer_v3.onnx")])
        parquet = destination / "parquet"
        parquet.mkdir()
        run([sys.executable, "tools/make_parquet_list.py", "--src_dir", str(destination),
             "--des_dir", str(parquet), "--num_utts_per_parquet", "1000", "--num_processes", "4"])


if __name__ == "__main__":
    main()
