# Full Somali fine-tuning plan: LLM + Flow, HiFT frozen

## Decision

We will adapt the pretrained `Fun-CosyVoice3-0.5B-2512` model to the clean
Somali corpus with **full-parameter fine-tuning** of these two modules:

| Module | Action | Why |
| --- | --- | --- |
| LLM (`llm.pt`) | Fine-tune every parameter | Learns Somali text-to-speech-token mapping: pronunciation, wording, timing, and prosody. |
| Flow (`flow.pt`) | Fine-tune every parameter | Adapts the speech-token-to-mel-spectrogram mapping to Somali acoustics and the target voices. |
| HiFT / HiFi-GAN (`hift.pt`) | Freeze initially | It is the pretrained waveform generator. Keeping it fixed preserves stable 24 kHz waveform rendering while the upstream modules adapt. |

This is **not LoRA/QLoRA, not an adapter-only run, and not training from
scratch**. Each selected module is initialized from the released checkpoint and
all of its trainable parameters receive optimizer updates. The HiFT checkpoint
is copied unchanged into the inference bundle.

## Why this split is the right first experiment

The current problem is Somali language and voice adaptation, not a change in
audio codec or output format. The LLM generates the discrete speech-token
sequence and the Flow model predicts the mel representation. Those are the
components with the most leverage over Somali pronunciation, pacing, prosody,
and speaker characteristics.

HiFT turns that mel representation into a waveform at the existing 24 kHz
sample rate. It should remain stable when the corpus uses the same clean,
single-channel, 24 kHz target format. Updating it immediately creates an
unnecessary failure mode: the upstream models can improve while the vocoder
introduces buzzing, metallic tones, instability, or reduced speaker quality.

We will only consider a separate HiFT/HiFi-GAN fine-tune after LLM + Flow
evaluation, and only if the held-out samples show clear vocoder-specific
artifacts across otherwise correct text and mel output, or if we intentionally
move to a different sample rate or substantially different acoustic domain.

## Scope and prerequisites

- Corpus: approximately 100 hours of clean Somali speech with reliable Somali
  transcripts.
- Audio target: mono, 24 kHz, lossless FLAC/WAV. The existing processed corpus
  already uses mono 24 kHz FLAC, which matches CosyVoice 3.
- Split: make a fixed train/dev/test split before any training. Keep the test
  set isolated; use it only for final comparison. If multiple speakers exist,
  stratify all splits by speaker and include held-out texts for every evaluated
  speaker.
- Speaker rights: include only speakers and recordings that are authorized for
  model training and for the intended deployment.
- Hardware: use a Linux NVIDIA CUDA machine for training. The local Mac setup
  is suitable for inference and data inspection, but the upstream trainer is
  CUDA/DDP-oriented (`nccl`, CUDA AMP, and GPU model wrapping). Do not plan the
  100-hour full fine-tune as a Mac CPU job.
- Storage: retain the original pretrained model untouched. Put every training
  run, checkpoint, TensorBoard log, manifest, and exported inference bundle in
  a new experiment directory.

## Data preparation

CosyVoice training consumes Parquet shards built from Kaldi-style manifests.
For each split (`train`, `dev`, and `test`), prepare a directory containing:

```text
wav.scp                 <utterance-id> <absolute path to audio>
text                    <utterance-id> <verified Somali transcript>
utt2spk                 <utterance-id> <speaker-id>
spk2utt                 <speaker-id> <utterance-id> ...
instruct                optional; use a consistent prompt template if present
```

For CosyVoice 3, generate the cached conditioning artifacts before sharding:

```text
utt2embedding.pt        utterance-level CampPlus speaker embeddings
spk2embedding.pt        speaker-level CampPlus embeddings
utt2speech_token.pt     discrete tokens from speech_tokenizer_v3.onnx
```

The upstream example scripts provide these stages:

```text
/Users/mali/ai/CosyVoice/tools/extract_embedding.py
/Users/mali/ai/CosyVoice/tools/extract_speech_token.py
/Users/mali/ai/CosyVoice/tools/make_parquet_list.py
```

Use the released model files from:

```text
/Users/mali/ai/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/
```

Specifically, preprocessing uses `campplus.onnx` and
`speech_tokenizer_v3.onnx`. The training tokenizer remains the existing Qwen2
tokenizer in `CosyVoice-BlankEN`; do not add an ad-hoc `<|so|>` language tag.
Somali text already tokenizes losslessly, as documented in `COSY.md`.

Before any GPU run, validate:

1. Every manifest audio path exists and is decodable.
2. Every transcript is non-empty, normalized consistently, and matches its
   recording closely enough to train on.
3. No utterance or source recording leaks across splits.
4. Durations fit the training configuration. The upstream CosyVoice 3 config
   uses 24 kHz and 960-frame feature windows; long recordings must be split
   into suitable utterances rather than silently truncated.
5. At least a small, hand-audited Somali evaluation set covers common words,
   proper names, numbers written out as words, questions, and long sentences.

## Training topology

The official trainer selects one named module with `--model` and removes the
other modules from the constructed training graph. Therefore the full run is
two sequential experiments, not one monolithic optimizer pass:

```text
released llm.pt  -> full Somali LLM fine-tune  -> llm_somali.pt
released flow.pt -> full Somali Flow fine-tune -> flow_somali.pt
released hift.pt -------------------------------- unchanged hift.pt
```

This is exactly how we keep HiFT frozen: **do not invoke** the trainer with
`--model hifigan`. The upstream `hifigan` path enables GAN training and uses
`train_conf_gan`; it is deliberately out of scope for the first run.

Run LLM and Flow from the same fixed data manifests and keep their experiment
directories independent. That makes it possible to evaluate:

1. released Flow + fine-tuned LLM,
2. fine-tuned Flow + released LLM, and
3. fine-tuned LLM + fine-tuned Flow.

This ablation tells us which module actually improves Somali before we make any
vocoder decision.

## Full-parameter commands

The following is the intended CUDA/Linux pattern after the Somali manifests and
Parquet lists have been created. Paths are intentionally isolated under a new
training workspace; they must not overwrite the downloaded base model.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export BASE_MODEL=/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
export SOMALI_DATA=/workspace/somali-cv3/data
export EXP_ROOT=/workspace/somali-cv3/exp
export TB_ROOT=/workspace/somali-cv3/tensorboard

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  --rdzv_id=260726 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  /workspace/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config /workspace/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
  --train_data "$SOMALI_DATA/train/parquet/data.list" \
  --cv_data "$SOMALI_DATA/dev/parquet/data.list" \
  --qwen_pretrain_path "$BASE_MODEL/CosyVoice-BlankEN" \
  --onnx_path "$BASE_MODEL" \
  --model llm \
  --checkpoint "$BASE_MODEL/llm.pt" \
  --model_dir "$EXP_ROOT/llm" \
  --tensorboard_dir "$TB_ROOT/llm" \
  --ddp.dist_backend nccl \
  --num_workers 8 --prefetch 100 --pin_memory --use_amp
```

After selecting the desired LLM checkpoint, run the same command for Flow:

```bash
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  --rdzv_id=260727 --rdzv_backend=c10d --rdzv_endpoint=localhost:1235 \
  /workspace/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config /workspace/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
  --train_data "$SOMALI_DATA/train/parquet/data.list" \
  --cv_data "$SOMALI_DATA/dev/parquet/data.list" \
  --qwen_pretrain_path "$BASE_MODEL/CosyVoice-BlankEN" \
  --onnx_path "$BASE_MODEL" \
  --model flow \
  --checkpoint "$BASE_MODEL/flow.pt" \
  --model_dir "$EXP_ROOT/flow" \
  --tensorboard_dir "$TB_ROOT/flow" \
  --ddp.dist_backend nccl \
  --num_workers 8 --prefetch 100 --pin_memory --use_amp
```

These are full-parameter commands. There is no LoRA flag, adapter injection,
or quantized training path. `--use_amp` is mixed-precision arithmetic for GPU
efficiency; it does **not** make the optimization adapter-only or INT8.

### Initial hyperparameters

Start from the released CosyVoice 3 configuration, rather than inventing a new
training recipe:

```text
LLM + Flow optimizer: Adam
Initial learning rate: 1e-5
Scheduler: constant LR
Warm-up: 2,500 steps
Gradient clipping: 5
Gradient accumulation: 2
Configured maximum epochs: 200
```

Treat 200 as a maximum, not a commitment. Save checkpoints frequently, monitor
development loss and synthesized dev prompts, and choose the best validation
checkpoint or a validation-best average. If the validation quality degrades or
pronunciation regresses, stop and select an earlier checkpoint rather than
training longer by default.

If the target GPUs cannot fit the required global batch at this setting, reduce
per-GPU batch capacity and increase accumulation while preserving a sensible
global batch. Record the exact world size, effective batch, GPU type, CUDA,
PyTorch, commit SHA, manifest hashes, and config copy for every run.

## Evaluation gates

Evaluate each checkpoint set before promoting it:

1. **Text fidelity:** Somali ASR/WER or CER where a trustworthy Somali ASR is
   available, plus manual transcription checks on a fixed held-out script.
2. **Pronunciation:** native-speaker review for Somali vowels, consonants,
   word boundaries, proper names, and code-switching if it is in scope.
3. **Naturalness/prosody:** blind listening comparison against the base model
   and the LLM-only ablation.
4. **Speaker similarity:** compare same-speaker and unseen-speaker zero-shot
   references separately.
5. **Waveform quality:** listen for buzz, metallic ringing, clipping, unstable
   pitch, and noise. Inspect peak levels and durations too.

Promote a checkpoint only when it improves Somali intelligibility and does not
materially regress voice similarity or waveform quality. Keep the base model as
the control in every evaluation.

## When to unfreeze HiFT

Do **not** unfreeze it because Somali pronunciation is imperfect. First correct
that through data quality, the LLM, and Flow. Revisit the vocoder only when all
of these are true:

1. LLM + Flow produce textually correct, stable speech tokens and mel output.
2. Artifacts persist across speakers and utterances in the waveform itself:
   metallic texture, buzzing, instability, or systematic high-frequency damage.
3. The corpus or deployment requires an intentionally different sample rate or
   a substantially different acoustic domain.

If that gate is reached, start a separate, low-risk HiFT experiment from the
pretrained `hift.pt`; do not merge it into the first LLM + Flow run. Compare it
against the frozen-vocoder baseline on the same held-out set before adoption.

## Deliverables for the first complete run

```text
somali-cv3/
  manifests/              immutable train/dev/test source manifests
  data/                   generated feature and Parquet artifacts
  exp/llm/                full LLM checkpoints and selected checkpoint
  exp/flow/               full Flow checkpoints and selected checkpoint
  tensorboard/            training logs
  eval/                   fixed prompts, audio, scores, and listening notes
  inference-bundle/       llm_somali.pt + flow_somali.pt + unchanged hift.pt
  run-manifest.md         exact commands, versions, hardware, and data hashes
```

The resulting inference bundle must preserve the base model's tokenizer,
speaker encoder, speech tokenizer, and HiFT checkpoint. Replace only `llm.pt`
and `flow.pt` after the selected checkpoints have passed evaluation.
