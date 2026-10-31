"""Save a voice as a Qwen3-TTS clone profile (.pt), so every line is read in the same voice.

The reference is a designed take (voice_design.py), maybe after voice_knobs.py, plus the text it
reads. --mix blends in the speaker embedding of a second designed voice; --ratio is that voice's
share (0.3 = 70 % reference, 30 % mix). The clone still follows the reference's pitch and melody.

usage: voice_profile.py <ref.wav> "<ref text>" <out.pt> [--mix other.wav --ratio 0.3]
                        [--say "a test line" --lang Chinese]
--say writes <out>_say_t1..3.wav to listen to
env QWEN_TTS_BASE: model folder or Hugging Face id (default Qwen/Qwen3-TTS-12Hz-1.7B-Base), TTS_DEVICE
"""
import argparse, dataclasses, os
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('text')
ap.add_argument('out')
ap.add_argument('--mix')
ap.add_argument('--ratio', type=float, default=0.3)
ap.add_argument('--say')
ap.add_argument('--lang', default='Chinese')
a = ap.parse_args()

model = Qwen3TTSModel.from_pretrained(os.environ.get('QWEN_TTS_BASE', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'),
                                      device_map=os.environ.get('TTS_DEVICE', 'cuda:0'), dtype=torch.bfloat16,
                                      attn_implementation='sdpa')
item = model.create_voice_clone_prompt(a.ref, a.text)[0]
if a.mix:
    e0 = item.ref_spk_embedding
    d = model.create_voice_clone_prompt(a.mix, x_vector_only_mode=True)[0].ref_spk_embedding.to(e0.device)
    e = (1 - a.ratio) * e0 + a.ratio * d
    item = dataclasses.replace(item, ref_spk_embedding=e * (e0.norm() / e.norm()))
torch.save({'item': dataclasses.asdict(item), 'ref': a.ref, 'text': a.text, 'mix': a.mix,
            'ratio': a.ratio if a.mix else 0.0}, a.out)
print('saved', a.out)
if a.say:
    stem = os.path.splitext(a.out)[0]
    for s in (1, 2, 3):
        torch.manual_seed(s)
        wavs, sr = model.generate_voice_clone(a.say, a.lang, voice_clone_prompt=[item])
        sf.write(f'{stem}_say_t{s}.wav', wavs[0], sr)
        print(f'{stem}_say_t{s}.wav: {len(wavs[0]) / sr:.1f}s', flush=True)
