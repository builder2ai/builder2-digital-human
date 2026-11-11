"""EXAMPLE from our song: judge one section where it sits in the whole mix, not on its own. Bar numbers (V = where the
section starts) and file names are that song's. SONG_DIR holds audio/, stems/ and review/.

The interlude and Verse 2 continued in place: level per half bar, spoken phrases against the band, words read from the full mix, and the gap before the third pass."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import os
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from songmap import SR, load, load_map, seg  # noqa: E402
from score_lines import load_audio, load_model, SR as WSR  # noqa: E402

V8 = Path(os.environ.get('SONG_DIR', '.')).resolve()
D_BAR = load_map().bar_seconds

V = 18
db = lambda x: round(float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-6)), 1)
name = sys.argv[1]
mix = load(V8 / 'audio' / f'{name}.mp3')
voice = load(V8 / 'stems' / f'{name}_voice.wav')
print('peak', round(float(np.abs(mix).max()), 3))
print('half bar: t, mix dB, voice dB')
for b in np.arange(V + 40, V + 53, .5):
    t = float(b * D_BAR)
    print(f'  {b:>5} {t:7.2f} {db(seg(mix, t, t + D_BAR / 2)):>6} {db(seg(voice, t, t + D_BAR / 2)):>7}')

# the final gain is applied to the mix only, so bring the voice up to the mix's scale before subtracting
record = __import__('json').loads((V8 / 'review' / f'mix_{name}.json').read_text(encoding='utf-8'))
gain = 10 ** (record['final_gain_db'] / 20)
n = min(len(mix), len(voice))
band = mix[:n] - voice[:n] * gain
for on, length, label in ((V + 47, 1.42, 'phrase 1'), (V + 49, 2.1, 'phrase 2')):
    t = on * D_BAR
    print(label, 'at', round(t, 2), 'voice dB', db(seg(voice, t, t + length) * gain), 'band under it dB', db(seg(band, t, t + length)))

m, _ = load_model()


def heard(a, b, track):
    x = track[int(a * WSR):int(b * WSR)]
    stream, _ = m.transcribe(x, language='zh', beam_size=5, vad_filter=False, condition_on_previous_text=False)
    return re.sub(r'\W', '', ''.join(s.text for s in stream))


full = load_audio(V8 / 'audio' / f'{name}.mp3')
start = (V + 52) * D_BAR
print('call in the mix:', heard((V + 47) * D_BAR - .3, (V + 50.2) * D_BAR, full))
print('verse 2 continued in the mix:')
for a in (-1.2, -.6, -1.6):
    print(' ', heard(start + a, start + 20.4, full))
for a in (41.4, 41.0, 39.5):
    print('last lines of the loop from bar', a, ':', heard((V + a) * D_BAR, (V + 45.1) * D_BAR, full))

# where his voice stops and the next word starts
v = voice.mean(axis=1)
hop = int(.02 * SR)
e = np.array([np.sqrt((v[i:i + hop] ** 2).mean()) for i in range(0, len(v) - hop, hop)])
t_end = (V + 60) * D_BAR
i0, i1 = int((t_end - 3) / .02), int((t_end + 2) / .02)
act = np.where(e[i0:i1] > .02)[0]
runs = []
start_i = act[0]
for p, q in zip(act, act[1:]):
    if q != p + 1:
        runs.append((round(i0 * .02 + start_i * .02 - t_end, 2), round(i0 * .02 + p * .02 - t_end, 2)))
        start_i = q
runs.append((round(i0 * .02 + start_i * .02 - t_end, 2), round(i0 * .02 + act[-1] * .02 - t_end, 2)))
print('voice runs around the third pass, relative to its first bar', runs)
