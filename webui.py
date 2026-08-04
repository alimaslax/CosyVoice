# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Pretrained Voice', '3-Second Voice Clone', 'Cross-Lingual Clone', 'Natural-Language Control']
instruct_dict = {
    'Pretrained Voice': '1. Select a pretrained voice\n2. Click Generate Audio',
    '3-Second Voice Clone': '1. Upload or record a prompt (30 seconds maximum; an uploaded file takes priority)\n2. Enter the matching prompt transcript\n3. Click Generate Audio',
    'Cross-Lingual Clone': '1. Upload or record a prompt (30 seconds maximum; an uploaded file takes priority)\n2. Click Generate Audio',
    'Natural-Language Control': '1. Upload or record prompt audio\n2. Enter a speaking instruction\n3. Click Generate Audio',
}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['Natural-Language Control']:
        if instruct_text == '':
            gr.Warning('Natural-Language Control requires an instruction.')
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_wav is None:
            gr.Warning('Natural-Language Control requires prompt audio.')
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_text != '':
            gr.Info('The prompt transcript is ignored in Natural-Language Control mode.')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['Cross-Lingual Clone']:
        if instruct_text != '':
            gr.Info('Instructions are ignored in Cross-Lingual Clone mode.')
        if prompt_wav is None:
            gr.Warning('Cross-Lingual Clone requires prompt audio.')
            yield (cosyvoice.sample_rate, default_data)
            return
        gr.Info('Make sure the synthesis text and prompt audio use different languages.')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3-Second Voice Clone', 'Cross-Lingual Clone']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is required.')
            yield (cosyvoice.sample_rate, default_data)
            return
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt sample rate {} Hz is below the required {} Hz.'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
            return
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['Pretrained Voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('Prompt text, prompt audio, and instructions are ignored in Pretrained Voice mode.')
        if sft_dropdown == '':
            gr.Warning('No pretrained voices are available for this model.')
            yield (cosyvoice.sample_rate, default_data)
            return
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3-Second Voice Clone']:
        if prompt_text == '':
            gr.Warning('Enter the transcript that matches the prompt audio.')
            yield (cosyvoice.sample_rate, default_data)
            return
        if instruct_text != '':
            gr.Info('The pretrained voice and instruction are ignored in 3-Second Voice Clone mode.')

    if mode_checkbox_group == 'Pretrained Voice':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3-Second Voice Clone':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        if cosyvoice.__class__.__name__ == 'CosyVoice3' and '<|endofprompt|>' not in prompt_text:
            prompt_text = 'You are a helpful assistant.<|endofprompt|>{}'.format(prompt_text.strip())
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-Lingual Clone':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        if '<|endofprompt|>' not in instruct_text:
            instruct_text = 'You are a helpful assistant. {}<|endofprompt|>'.format(instruct_text.strip())
        for i in instruct_cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Repository: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    Pretrained models: [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Enter the text to synthesize, choose an inference mode, and follow the displayed steps.")

        tts_text = gr.Textbox(label="Text to Synthesize", lines=1, placeholder="Enter the text you want the model to speak...", value="")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Inference Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Pretrained Voice', value=sft_spk[0], scale=0.25,
                                       visible=has_pretrained_voices)
            stream = gr.Radio(choices=stream_mode_list, label='Stream Audio', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed (non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload Prompt Audio (16 kHz or higher)',
                                         value=default_prompt_wav)
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record Prompt Audio')
        prompt_text = gr.Textbox(label="Prompt Transcript", lines=1, placeholder="Enter the exact words spoken in the prompt audio...",
                                 value=default_prompt_text)
        instruct_text = gr.Textbox(label="Instruction", lines=1, placeholder="Enter a natural-language speaking instruction...", value='')

        generate_button = gr.Button("Generate Audio")

        audio_output = gr.Audio(label="Generated Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name=args.server_name, server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--server_name',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--instruct_model_dir',
                        type=str,
                        default=None,
                        help='optional instruction-capable model used only for Natural-Language Control')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    instruct_cosyvoice = cosyvoice
    if args.instruct_model_dir:
        logging.info('loading Natural-Language Control model from %s', args.instruct_model_dir)
        instruct_cosyvoice = AutoModel(model_dir=args.instruct_model_dir)
        if instruct_cosyvoice.sample_rate != cosyvoice.sample_rate:
            raise ValueError('main and instruction models must use the same sample rate')

    sft_spk = cosyvoice.list_available_spks()
    has_pretrained_voices = len(sft_spk) > 0
    if not has_pretrained_voices:
        inference_mode_list.remove('Pretrained Voice')
        sft_spk = ['']
    default_prompt_wav = os.environ.get('COSYVOICE_PROMPT_WAV')
    if default_prompt_wav and not os.path.isfile(default_prompt_wav):
        logging.warning('default prompt audio not found: {}'.format(default_prompt_wav))
        default_prompt_wav = None
    default_prompt_text = ''
    default_prompt_text_file = os.environ.get('COSYVOICE_PROMPT_TEXT_FILE')
    if default_prompt_text_file and os.path.isfile(default_prompt_text_file):
        with open(default_prompt_text_file, 'r', encoding='utf-8') as file:
            default_prompt_text = file.read().strip()
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
