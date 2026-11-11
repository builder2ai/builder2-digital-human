"""EXAMPLE from our song: write a spec for ace_step_run.py that has ACE-Step write a song's whole ending in one piece.
The bar numbers, caption and file names are that song's; adapt them. SONG_DIR holds review/ and stems/; the spec and the
context file go to SONG_DIR/ace_beds.

One piece from his second-last line to the song's end: ACE-Step hears the third pass's accompaniment up to bar 12 and writes everything after it (the band dropping out, the erhu under his last two lines, the closing chord), so there are no joins or part levels to make by hand.
Writes src/lastpiece_ctx.wav (draft accompaniment, bars 4-12 of the third pass) and spec_p5.json.
Usage: make_spec_ending.py [count]
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
from songmap import SR, load, load_map, seg  # noqa: E402

V8 = Path(os.environ.get('SONG_DIR', '.')).resolve()
HERE = V8 / 'ace_beds'
D_BAR = load_map().bar_seconds

BASE = 'stage7_v5'
FROM_BAR, WRITE_FROM, WRITE_TO = 4, 12, 19   # bars of the third pass; his last word ends near bar 15.5
CAPTION = ('Chinese folk pop backing track in D minor, 99 BPM, pipa and muted guitar over a relaxed groove. Then the drums and bass stop and the song turns quiet and sparse: '
           'a solo erhu plays a slow, gentle melody with long held notes over soft pipa, then one short closing phrase, and everything settles on a final D minor chord that rings out and dies away. No vocals.')
count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
rec = json.loads((V8 / 'review' / f'mix_{BASE}.json').read_text(encoding='utf-8'))
third_at = round(rec['sections']['third'] / D_BAR) * D_BAR
ctx = seg(load(V8 / 'stems' / f'{BASE}_bed.wav'), third_at + FROM_BAR * D_BAR, third_at + WRITE_FROM * D_BAR)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-c:a', 'pcm_s24le', str(HERE / 'src' / 'lastpiece_ctx.wav')], input=ctx.astype(np.float32).tobytes(), check=True)
start, end = (WRITE_FROM - FROM_BAR) * D_BAR, (WRITE_TO - FROM_BAR) * D_BAR
variants = [{'id': f'p7r_{202619660 + k}', 'seed': 202619660 + k, 'task': 'repaint', 'caption': CAPTION, 'src_audio': 'src/lastpiece_ctx.wav',
             'repainting_start': round(start, 3), 'repainting_end': round(end, 3), 'duration': round(end, 3), 'new_seconds': round(end - start, 3)} for k in range(count)]
spec = {'bpm': 99, 'keyscale': 'D minor', 'duration': round(end, 3), 'lm_negative_prompt': 'vocals, singing', 'from_bar': FROM_BAR,
        'note': 'The whole ending in one piece: bars 12-19 of the third pass written after the accompaniment, the band dropping out by itself.', 'candidates': variants}
(HERE / 'spec_p5.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(variants), 'candidates; context', round(len(ctx) / SR, 2), 's, new', round(start, 3), '-', round(end, 3))
