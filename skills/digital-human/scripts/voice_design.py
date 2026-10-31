"""Design a voice from a written description with Qwen3-TTS VoiceDesign. The voice belongs to no one.

Each seed gives a somewhat different person. Make a few, let a person pick one, and from then on use
that take as the reference for the clone model (voice_profile.py): only the clone model keeps the
same voice from line to line.

usage: voice_design.py <out_dir> "<description>" "<sample text>" [takes=4] [lang=Chinese]
writes <out_dir>/design_t<k>.wav and <out_dir>/design.json (description, sample text, language)
env QWEN_TTS_DESIGN: model folder or Hugging Face id (default Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
    TTS_DEVICE (default cuda:0)
"""
import sys, os, json
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

out, desc, text = sys.argv[1], sys.argv[2], sys.argv[3]
takes = int(sys.argv[4]) if len(sys.argv) > 4 else 4
lang = sys.argv[5] if len(sys.argv) > 5 else 'Chinese'
os.makedirs(out, exist_ok=True)
model = Qwen3TTSModel.from_pretrained(os.environ.get('QWEN_TTS_DESIGN', 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'),
                                      device_map=os.environ.get('TTS_DEVICE', 'cuda:0'), dtype=torch.bfloat16,
                                      attn_implementation='sdpa')
json.dump({'description': desc, 'text': text, 'lang': lang}, open(os.path.join(out, 'design.json'), 'w',
                                                                   encoding='utf-8'), ensure_ascii=False, indent=1)
for s in range(1, takes + 1):
    torch.manual_seed(s)
    wavs, sr = model.generate_voice_design(text, desc, lang)
    p = os.path.join(out, f'design_t{s}.wav')
    sf.write(p, wavs[0], sr)
    print(f'{p}: {len(wavs[0]) / sr:.1f}s', flush=True)
