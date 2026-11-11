"""YuE2 continuation for the isolated service only.

Saves a take's music tokens, and can keep an earlier take's tokens up to a given time and sample on from there,
so a new line inherits the voice, key and tempo of the take it continues. Direct mode (no ABC) only.
ComfyUI's own files are not edited; the patch passes through to the stock code for any other node.
"""
import json
from pathlib import Path

import torch

import comfy.model_management
from comfy.text_encoders import yue2

_stock = yue2.YuE2TEModel.encode_token_weights


def encode_token_weights(self, tokens):
    job = tokens.get('continuation')
    if job is None:
        return _stock(self, tokens)
    device = self.execution_device
    dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    prefix = tokens['prefix'] + [yue2.ABC_END, yue2.MUSIC_START]
    negative = tokens['negative'] + [yue2.MUSIC_START]
    keep = job['keep']
    context = self.config.max_position_embeddings
    budget = min(tokens['max_tokens'], context - max(len(prefix), len(negative))) - len(keep)
    if budget < 1:
        raise ValueError('YuE2 continuation: nothing left to sample after the kept tokens.')
    new, truncated = self._generate(
        prefix + keep, tokens['seed'], budget, 'semantic', dtype,
        negative=negative + keep, cfg_scale=tokens['cfg_scale'], legacy_off=True,
        temperature=tokens['temperature'], top_p=tokens['top_p'], top_k=tokens['top_k'],
        repetition_penalty=tokens['repetition_penalty'], penalty_window=50,
        min_tokens=min(200, budget),
    )
    semantic = keep + new
    if job['save']:
        Path(job['save']).write_text(json.dumps(semantic), encoding='utf-8')
    conditioning, chunks = self._acoustic_conditioning(prefix, semantic, dtype)
    return conditioning, None, {
        'yue2_chunks': chunks, 'yue2_abc_ids': [], 'yue2_frames': len(semantic),
        'yue2_truncated': truncated,
    }


yue2.YuE2TEModel.encode_token_weights = encode_token_weights


class YuE2ContinueMusic:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',),
            'style': ('STRING', {'multiline': True}),
            'lyrics': ('STRING', {'multiline': True}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'max_duration': ('FLOAT', {'default': 105.0, 'min': .04, 'max': 900.0, 'step': .04}),
            'temperature': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 5.0, 'step': .01}),
            'top_p': ('FLOAT', {'default': .95, 'min': .01, 'max': 1.0, 'step': .01}),
            'top_k': ('INT', {'default': 100, 'min': 1, 'max': 32768}),
            'repetition_penalty': ('FLOAT', {'default': 1.2, 'min': .01, 'max': 10.0, 'step': .01}),
            'cfg_scale': ('FLOAT', {'default': 1.01, 'min': 0.0, 'max': 100.0, 'step': .01}),
            'keep_tokens_file': ('STRING', {'default': ''}),
            'keep_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 900.0, 'step': .04}),
            'save_tokens_file': ('STRING', {'default': ''}),
        }}

    RETURN_TYPES = ('CONDITIONING', 'FLOAT')
    RETURN_NAMES = ('conditioning', 'seconds')
    FUNCTION = 'execute'
    CATEGORY = 'model/conditioning/yue2'

    def execute(self, clip, style, lyrics, seed, max_duration, temperature, top_p, top_k, repetition_penalty, cfg_scale,
                keep_tokens_file, keep_seconds, save_tokens_file):
        keep = []
        if keep_tokens_file:
            keep = json.loads(Path(keep_tokens_file).read_text(encoding='utf-8'))[:round(keep_seconds * yue2.FRAMES_PER_SECOND)]
        tokens = clip.tokenize(style, lyrics=lyrics, cot='off', seed=seed, abc='',
                               max_tokens=max(1, round(max_duration * yue2.FRAMES_PER_SECOND)),
                               temperature=temperature, top_p=top_p, top_k=top_k,
                               repetition_penalty=repetition_penalty, cfg_scale=cfg_scale)
        tokens['continuation'] = {'keep': keep, 'save': save_tokens_file}
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (conditioning, conditioning[0][1]['yue2_frames'] / yue2.FRAMES_PER_SECOND)


NODE_CLASS_MAPPINGS = {'YuE2ContinueMusic': YuE2ContinueMusic}
