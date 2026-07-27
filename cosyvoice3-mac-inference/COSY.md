# CosyVoice 3 on macOS

This folder contains a reproducible macOS setup for the local
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` runtime, plus a zero-shot Somali test
using `0007.flac` and its transcript in `tts.text`.

## Files here

- `0007.flac` — 25-second Somali speaker reference, mono 24 kHz FLAC.
- `tts.text` — the plain-text transcript matching `0007.flac`. The runner
  uses this file as its zero-shot prompt transcript.
- `0007.json` — the original word-level transcription metadata retained for
  reference; it is not read by the runner.
- `cosyvoice3_somali_0007.wav` — the resulting Somali test: mono 24 kHz WAV.
- `run_zero_shot_macos.sh` — the reusable launcher. It activates the Somali
  virtual environment, configures writable caches, and invokes the Python
  runner.
- `run_zero_shot.py` — the inference implementation.
- `runtime-cache/` — created on first run; it is deliberately ignored because
  it only holds temporary Numba, Matplotlib, and Hugging Face cache data.

## One-time setup

Run all commands from the repository root, using the project environment
specified in `AGENTS.md`:

```bash
source ~/python/somali/bin/activate
cd /Users/mali/ai/CosyVoice
git submodule update --init --recursive
```

The Matcha-TTS submodule is required. Without it, startup fails with
`ModuleNotFoundError: No module named 'matcha'`.

Download the model once (about 8 GB including the available inference assets):

```bash
source ~/python/somali/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='/Users/mali/ai/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B',
    ignore_patterns=['llm.rl.pt'],  # Optional RL checkpoint; not needed here.
)
PY
```

Install the runtime packages into that environment. Use `python -m pip`, not
the `pip` executable: this virtual environment's standalone `pip` launcher may
point at an old relocated interpreter.

```bash
source ~/python/somali/bin/activate
python -m pip install \
  'conformer==0.3.2' 'diffusers==0.29.0' 'HyperPyYAML==1.2.3' \
  'inflect==7.3.1' 'librosa==0.10.2' 'modelscope==1.20.0' \
  'omegaconf==2.3.0' 'onnx==1.16.0' 'onnxruntime==1.18.0' \
  'transformers==4.51.3' 'x-transformers==2.11.24' 'wetext==0.0.4' \
  'openai-whisper==20231117' 'hydra-core==1.3.2' 'lightning==2.2.4' \
  'gdown==5.1.0' 'wget==3.2' 'matplotlib==3.7.5' 'rich==13.7.1' \
  'pyarrow==18.1.0' 'pyworld==0.3.4'
```

On a clean macOS Python 3.9 install, Whisper may need these first:

```bash
python -m pip install 'setuptools<81' wheel
python -m pip install --no-build-isolation 'openai-whisper==20231117'
```

`pyworld` builds locally on macOS, so Xcode Command Line Tools must be
available (`xcode-select --install`) if no compatible wheel is found.

## Why the runtime caches are local

CosyVoice imports Numba, Matplotlib, and Hugging Face utilities during model
initialization. On this machine, the normal paths under `~/.cache` and
`~/.matplotlib` were not writable from the runtime, leading to errors such as
`RuntimeError: cannot cache function` and Hugging Face cache warnings.

The launcher fixes this by exporting:

```text
NUMBA_CACHE_DIR=<this folder>/runtime-cache/numba
MPLCONFIGDIR=<this folder>/runtime-cache/matplotlib
HF_HOME=<this folder>/runtime-cache/huggingface
PYTHONPYCACHEPREFIX=<this folder>/runtime-cache/pycache
TOKENIZERS_PARALLELISM=false
```

The launcher creates these directories with owner-only permissions (`0700`).
The supplied speaker reference is owner-readable only (`0600`); keep any new
speaker reference audio at that permission level. Plain-text transcripts and
output WAVs can be `0644` when they do not contain sensitive content.

## Run the Somali zero-shot example

```bash
cd /Users/mali/ai/sod-code/cosyvoice3
./run_zero_shot_macos.sh \
  --reference 0007.flac \
  --text 'Subax wanaagsan. Maanta waa maalin fiican. Waxaan rajaynayaa inaad caafimaad qabto, oo aad maalintaada si farxad leh u bilowdo.' \
  --output cosyvoice3_somali_0007.wav
```

Pass `--prompt-text PATH` only when using a different transcript file; the
default is `tts.text`.

The model runs on CPU on this Mac (CosyVoice's CUDA path is unavailable), so
the first load and long reference prompts take longer. The 25-second supplied
reference generated a 12.64-second Somali test at 24 kHz.

## Somali tokenizer findings

### What works

CosyVoice 3 tokenizes Somali text cleanly. Its actual TTS text tokenizer is
the bundled Qwen2 BPE tokenizer in `CosyVoice-BlankEN`, loaded by
`cosyvoice/tokenizer/tokenizer.py`. It does not have an unknown-token ID, so
ordinary Somali Latin-script text is represented losslessly rather than being
replaced with unknown symbols.

The local checks for this folder's text found:

| Text | Characters | Tokens | Unknown tokens | Round trip |
| --- | ---: | ---: | ---: | --- |
| `tts.text` with the required prompt prefix | 283 | 117 | 0 | Exact |
| Somali output test sentence | 127 | 50 | 0 | Exact |

For example, `Subax wanaagsan` is represented as ordinary subword pieces
(`Sub`, `ax`, and pieces for ` wanaagsan`), not as bytes, replacement symbols,
or an unsupported-language fallback.

`<|endofprompt|>` is a real CosyVoice special token and must remain between
the required assistant prefix and the reference transcript. In this installed
model it has token ID `151646`.

### What is *not* Somali support

Do **not** prepend `<|so|>` to Somali text. It is not a recognized CosyVoice
3 language-control token; the tokenizer reads it literally as `<`, `|`, `so`,
`|`, `>`.

The source tree also contains a separate legacy Whisper tokenizer table that
maps `so` to Somali. That table is not used to tokenize TTS text: CosyVoice 3
uses Whisper only to extract acoustic features from the reference audio.

The model card officially lists TTS support for Chinese, English, Japanese,
Korean, German, Spanish, French, Italian, and Russian. Somali is not in that
published support list. Therefore:

- Somali text is valid model input and is losslessly tokenized.
- Somali pronunciation quality is unverified; tokenization alone does not
  mean the model learned Somali phonology.
- A Somali reference clip supplies speaker/acoustic conditioning, which can
  improve voice similarity, but it does not train the model on Somali.
- Fine-grained pronunciation controls are provided for English CMU phonemes
  and Chinese pinyin, not Somali.

### Input guidance

Use normal Somali spelling and punctuation, as in `tts.text`. Avoid language
tags such as `<|so|>`. The fallback text-normalization path treats all
non-Chinese text as English-like, so write out numbers and abbreviations in
words when pronunciation matters.

## Temporary files used during the original setup

The first setup used these transient, safely removable paths:

```text
/private/tmp/cosyvoice3-download.log
/private/tmp/cosyvoice3-deps.log
/private/tmp/cosyvoice3-somali.log
/private/tmp/cosyvoice-numba-cache
/private/tmp/cosyvoice-matplotlib
/private/tmp/cosyvoice-hf-cache
```

They are not needed by the reusable launcher. It keeps future cache data in
this folder's ignored `runtime-cache/` directory instead.
