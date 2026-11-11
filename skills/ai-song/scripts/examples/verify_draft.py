"""EXAMPLE from our song: the delivery check of a whole draft. It reads review/mix_<name>.json (section start times written
by the mixing step), stems/<name>_voice.wav and audio/<name>.mp3 under SONG_DIR; the comparison with 'take D' below checks
that a kept take is still sample-for-sample in the draft, with that song's times.

Check a draft: words per section from its voice bus, voice overlaps, the kept take, level per section, and hard jumps.
Usage: verify_draft.py <name>
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_lines import load_audio, load_model, transcribe, SR  # noqa: E402
from songmap import load_map  # noqa: E402

D = Path(os.environ.get('SONG_DIR', '.')).resolve()
D_BAR_SECONDS = load_map().bar_seconds
name = sys.argv[1]
record = json.loads((D / 'review' / f'mix_{name}.json').read_text(encoding='utf-8'))
voice = load_audio(D / 'stems' / f'{name}_voice.wav')
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(D / 'audio' / f'{name}.mp3'), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
mix = np.frombuffer(raw, np.float32)
# the peak from the stereo file: ffmpeg's downmix to mono can read about 1.4 times too high
stereo = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(D / 'audio' / f'{name}.mp3'), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
print('seconds', round(len(mix) / SR, 2), 'peak', round(float(np.abs(np.frombuffer(stereo, np.float32)).max()), 3))

model, _ = load_model()
names = list(record['sections'])
for i, key in enumerate(names):
    a = record['sections'][key] - 1.3
    b = record['sections'][names[i + 1]] - 1.3 if i + 1 < len(names) else len(voice) / SR
    n = int(.1 * SR)
    frames = [float(np.sqrt((mix[j:j + n] ** 2).mean())) for j in range(max(int(a * SR), 0), int(b * SR) - n, n)]
    print(f"{int(max(a, 0) // 60)}:{max(a, 0) % 60:04.1f}", key, 'mix rms', round(float(np.mean(frames)), 3), '|', transcribe(model, voice, max(a, 0), b), flush=True)

d = load_audio(D / 'stems' / 'D_vocals.flac')
start = record['sections']['verse1'] + (23.0 - (.303 + 9 * D_BAR_SECONDS))  # take D's 23 s, past any overlap with what comes before
x = voice[int(start * SR):int((start + 13) * SR)]
y = d[int(23.0 * SR) - 160:int(23.0 * SR) + len(x) + 160]
print('verse 1 vs take D', round(max(float(np.corrcoef(x, y[k:k + len(x)])[0, 1]) for k in range(0, 321, 4)), 4))
n = int(D_BAR_SECONDS * SR)
bars = np.array([float(np.sqrt((mix[j:j + n] ** 2).mean())) for j in range(0, len(mix) - n, n)])
print('quiet bars (under 40% of the median):', [(round(k * D_BAR_SECONDS, 1), round(float(v), 3)) for k, v in enumerate(bars) if v < .4 * np.median(bars)], 'median', round(float(np.median(bars)), 3))
step = np.abs(np.diff(mix))
print('largest sample jumps at', [round(float(k) / SR, 2) for k in np.argsort(step)[-5:]], 'size', round(float(step.max()), 3))
