"""Voice a script line by line with a saved voice profile (voice_profile.py).

Every line, and every alternative wording under "alt", gets <takes> takes with seeds seed0.., so
tts_qc.py can pick the best one. The model is loaded once.

script.json: {"lang": "Chinese", "lines": [{"id": 1, "text": "subtitle text", "tts": "what to read"}, ...]}
  "tts" is optional: write polyphones the model gets wrong as same-sound characters there
  (角色 -> 绝色 jué, 调 tiáo -> 条, 长 zhǎng -> 涨); subtitles keep "text".
  "alt": {"4b": "another wording"} adds variants of line 4.

usage: tts_lines.py <script.json> <profile.pt> <out_dir> [takes=3] [ids=1,4,...|all] [seed0=1]
writes <out_dir>/L<id>_t<k>.wav (id zero-padded: L04, L04b)
Some seeds drift into a different speaker; tts_qc.py catches them. To add takes for a failing
line, run again with a new seed0 instead of overwriting.
env QWEN_TTS_BASE, TTS_DEVICE as in voice_profile.py
"""
import sys, os, json
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

script, prof, out = sys.argv[1], sys.argv[2], sys.argv[3]
takes = int(sys.argv[4]) if len(sys.argv) > 4 else 3
only = set(sys.argv[5].split(',')) if len(sys.argv) > 5 and sys.argv[5] != 'all' else None
seed0 = int(sys.argv[6]) if len(sys.argv) > 6 else 1
os.makedirs(out, exist_ok=True)
model = Qwen3TTSModel.from_pretrained(os.environ.get('QWEN_TTS_BASE', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'),
                                      device_map=os.environ.get('TTS_DEVICE', 'cuda:0'), dtype=torch.bfloat16,
                                      attn_implementation='sdpa')
item = VoiceClonePromptItem(**torch.load(prof, weights_only=False)['item'])

js = json.load(open(script, encoding='utf-8'))
lang = js.get('lang', 'Chinese')
jobs = []
for ln in js['lines']:
    if not only or str(ln['id']) in only:
        jobs.append((f"{ln['id']:02d}", ln.get('tts', ln['text'])))
    for k, t in ln.get('alt', {}).items():
        if not only or k in only:
            jobs.append((f"{int(k[:-1]):02d}{k[-1]}", t))
for lid, text in jobs:
    for s in range(seed0, seed0 + takes):
        torch.manual_seed(s)
        wavs, sr = model.generate_voice_clone(text, lang, voice_clone_prompt=[item])
        o = f'{out}/L{lid}_t{s}.wav'
        sf.write(o, wavs[0], sr)
        print(f'L{lid}_t{s}: {len(wavs[0]) / sr:.1f}s', flush=True)
