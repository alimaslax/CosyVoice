#!/usr/bin/env python3
"""Generate zero-shot speech with a local CosyVoice 3 model on macOS."""

import argparse
import os
from pathlib import Path
import sys

import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--prompt-text",
        type=Path,
        default=Path("tts.text"),
        help="Plain-text transcript matching the speaker reference (default: tts.text).",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/Users/mali/ai/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cosyvoice_dir = Path(os.environ.get("COSYVOICE_DIR", "/Users/mali/ai/CosyVoice"))
    sys.path.insert(0, str(cosyvoice_dir / "third_party/Matcha-TTS"))
    sys.path.insert(0, str(cosyvoice_dir))

    from cosyvoice.cli.cosyvoice import AutoModel

    prompt_text = args.prompt_text.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt text file is empty: {args.prompt_text}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModel(model_dir=str(args.model_dir))

    for result in model.inference_zero_shot(
        args.text,
        "You are a helpful assistant.<|endofprompt|>" + prompt_text,
        str(args.reference),
        stream=False,
    ):
        torchaudio.save(str(args.output), result["tts_speech"], model.sample_rate)
        print(f"Saved {args.output} at {model.sample_rate} Hz")


if __name__ == "__main__":
    main()
