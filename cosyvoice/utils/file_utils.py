# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import shutil
import uuid
from pathlib import Path
import torch
import torchaudio
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


# NOTE do not support bistream inference as only speech token embedding/head is kept
_VLLM_EXPORT_MARKER = '.cosyvoice-vllm-export-ready'
_VLLM_EXPORT_FORMAT = 1


def _file_fingerprint(path):
    """Return inexpensive source-checkpoint metadata for the export marker."""
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def vllm_export_is_ready(model_path):
    """Return whether ``model_path`` is a complete CosyVoice vLLM export.

    An interrupted ``save_pretrained`` leaves a directory behind.  vLLM then
    blocks or fails much later while trying to load that partial directory, so
    existence alone must never be used as the cache-validity signal.
    """
    model_path = Path(model_path)
    marker = model_path / _VLLM_EXPORT_MARKER
    config_path = model_path / 'config.json'
    if not marker.is_file() or not config_path.is_file():
        return False
    try:
        export_metadata = json.loads(marker.read_text(encoding='utf-8'))
        config = json.loads(config_path.read_text(encoding='utf-8'))
        source_llm = model_path.parent / 'llm.pt'
        source_matches = (export_metadata.get('format') == _VLLM_EXPORT_FORMAT
                          and export_metadata.get('source_llm') == _file_fingerprint(source_llm))
    except (OSError, json.JSONDecodeError):
        return False
    architectures = config.get('architectures', [])
    has_custom_model = 'CosyVoice2ForCausalLM' in architectures
    has_weights = any(path.is_file() and path.stat().st_size > 0
                      for path in model_path.glob('*.safetensors'))
    return source_matches and has_custom_model and has_weights


def export_cosyvoice2_vllm(model, model_path, device):
    """Export a complete vLLM cache atomically, replacing partial exports."""
    model_path = Path(model_path)
    if vllm_export_is_ready(model_path):
        logging.info('using validated vLLM export cache at %s', model_path)
        return

    # Keep the staging directory beside the final cache so the final rename is
    # atomic on the mounted network volume.
    staging_path = model_path.parent / '.{}.staging-{}'.format(model_path.name, uuid.uuid4().hex)
    shutil.rmtree(staging_path, ignore_errors=True)
    logging.warning('building vLLM export cache at %s (replacing incomplete cache if present)', model_path)

    dtype = torch.bfloat16
    # lm_head
    use_bias = True if model.llm_decoder.bias is not None else False
    model.llm.model.lm_head = model.llm_decoder
    # embed_tokens
    embed_tokens = model.llm.model.model.embed_tokens
    model.llm.model.set_input_embeddings(model.speech_embedding)
    model.llm.model.to(device)
    model.llm.model.to(dtype)
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = model.speech_embedding.num_embeddings
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = use_bias
    try:
        model.llm.model.save_pretrained(staging_path)
        config_path = staging_path / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        # Avoid a shell edit: it is brittle on network mounts and obscures an
        # incomplete export when it fails.
        if use_bias is True:
            config['architectures'] = ['CosyVoice2ForCausalLM']
            config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        marker_metadata = {
            'format': _VLLM_EXPORT_FORMAT,
            'source_llm': _file_fingerprint(model_path.parent / 'llm.pt'),
        }
        (staging_path / _VLLM_EXPORT_MARKER).write_text(
            json.dumps(marker_metadata, sort_keys=True) + '\n', encoding='utf-8')
        if not vllm_export_is_ready(staging_path):
            raise RuntimeError('vLLM export validation failed for {}'.format(staging_path))
        if model_path.exists():
            shutil.rmtree(model_path)
        staging_path.replace(model_path)
        logging.info('vLLM export cache committed at %s', model_path)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise
    finally:
        model.llm.model.config.vocab_size = tmp_vocab_size
        model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
        model.llm.model.set_input_embeddings(embed_tokens)
