"""Shared fixed-reference pace controls for the Somali CosyVoice model."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / 'asset' / 'somali_pace_prompts'

PACE_PRESETS = {
    'slow': {
        'label': 'Slow · < 120.9 WPM',
        'prompt_filename': 'slow.flac',
        'instruction': 'You are a helpful assistant. Speak slowly and deliberately.<|endofprompt|>',
    },
    'medium': {
        'label': 'Medium · 120.9–154.5 WPM',
        'prompt_filename': 'medium.flac',
        'instruction': 'You are a helpful assistant. Speak at a natural, moderate pace.<|endofprompt|>',
    },
    'fast': {
        'label': 'Fast · > 154.5 WPM',
        'prompt_filename': 'fast.flac',
        'instruction': 'You are a helpful assistant. Speak at a fast pace.<|endofprompt|>',
    },
}


def get_prompt_path(pace: str, prompt_dir: Path = PROMPT_DIR) -> str:
    """Return the packaged Omar reference clip for a pace preset."""
    try:
        preset = PACE_PRESETS[pace]
    except KeyError as error:
        raise ValueError('Unknown pace: {}'.format(pace)) from error
    path = Path(prompt_dir) / preset['prompt_filename']
    if not path.is_file():
        raise FileNotFoundError('Missing packaged pace prompt: {}'.format(path))
    return str(path)
