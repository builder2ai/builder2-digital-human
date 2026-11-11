"""EXAMPLE from our song: screen ACE-Step ending takes before anyone listens. Bar numbers and file names are that song's.
SONG_DIR holds review/, stems/ and ace_beds/.

Takes that write the music under his last two lines (spec_p4.json: bars 11.5-16 between two contexts; spec_p5.json: everything from bar 12 to the end).
For each: is the context kept, are the drums gone (the top band, which the drums fill, against the bars before), do the roots stay those his voice was sung over (above all a D under his last word), is it on the beat, how loud it is against his voice, the lead's notes per eighth against his and how often they rub, and how the end dies away.
On the beat: beat_check.py from bar 12 against the 4 context bars before it, at the eighth note (not the beat: a 3+3+2
accent pulls a beat-period reading by up to half a beat), the accent shift in eighths (not 0 = displaced), and the first
strong onset against the bar line; when the new bars are a held chord, only the onset counts.
Usage: score_ending.py <spec> [ids...]
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import os
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import beat_check  # noqa: E402
from songmap import SR, load, load_map, seg  # noqa: E402

V8 = Path(os.environ.get('SONG_DIR', '.')).resolve()
HERE = V8 / 'ace_beds'
D_BAR = load_map().bar_seconds

NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
db = lambda x: round(float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-7)), 1)
spec = json.loads((HERE / sys.argv[1]).read_text(encoding='utf-8'))
FROM = spec['from_bar']
at = lambda b: (b - FROM) * D_BAR      # a bar of the third pass on the take's clock
rec = json.loads((V8 / 'review' / 'mix_stage7_v5.json').read_text(encoding='utf-8'))
third_at = round(rec['sections']['third'] / D_BAR) * D_BAR
bed = load(V8 / 'stems' / 'stage7_v5_bed.wav')
his = load(V8 / 'stems' / 'stage7_v5_voice.wav')
draft = lambda x, b0, b1: seg(x, third_at + b0 * D_BAR, third_at + b1 * D_BAR)


def band(x, lo, hi):
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(np.fft.rfft(x, axis=0) * ((f >= lo) & (f < hi))[:, None], n=len(x), axis=0)


def root(x):
    s = np.abs(np.fft.rfft(x.mean(axis=1) * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    low = (f >= 40) & (f < 220)
    pc = np.round(69 + 12 * np.log2(f[low] / 440)).astype(int) % 12
    return int(np.array([(s[low][pc == k] ** 2).sum() for k in range(12)]).argmax())


def note(x, lo, hi, floor=-45):
    s = x.mean(axis=1)
    if 20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-7) < floor:
        return None
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s)), 16384))
    f = np.fft.rfftfreq(16384, 1 / SR)
    b = (f > lo) & (f < hi)
    return int(round(69 + 12 * np.log2(f[b][int(sp[b].argmax())] / 440))) % 12


def notes_line(ns):
    return ' | '.join(' '.join('.' if k is None else NAMES[k] for k in ns[i:i + 8]) for i in range(0, len(ns), 8))


src = load(HERE / spec['candidates'][0]['src_audio'])
top_before = db(band(seg(src, at(9), at(11)), 6000, 14000))
halves = np.arange(12, 16, .5)
roots_was = [root(draft(bed, b, b + .5)) for b in halves]
eighths = np.arange(11.75, 15.5, 1 / 8)
his_notes = [note(draft(his, t, t + 1 / 8), 100, 700, -40) for t in eighths]
print(json.dumps({'roots_was': ' '.join(NAMES[r] for r in roots_was), 'his': notes_line(his_notes), 'his_db': [db(draft(his, b, b + 1)) for b in range(12, 16)], 'top_band_before': top_before}, ensure_ascii=False))
for c in spec['candidates']:
    path = HERE / 'raw' / f"{c['id']}.wav"
    if (sys.argv[2:] and c['id'] not in sys.argv[2:]) or not path.exists():
        continue
    x = load(path)
    n = min(len(x), len(src))
    a, z = int(float(c['repainting_start']) * SR), int(float(c['repainting_end']) * SR)
    before = slice(0, a - SR // 10)
    kept = float(np.corrcoef(x[before].ravel(), src[before].ravel())[0, 1])
    last = len(x) / SR / D_BAR + FROM
    roots = [root(seg(x, at(b), at(b + .5))) for b in halves]
    bc = beat_check.check(x, at(12), D_BAR, new='after', ending=True, before_bars=min(4, 12 - FROM), after_bars=4)
    lead_notes = [note(seg(x, at(t), at(t + 1 / 8)), 300, 2500) for t in eighths]
    both = [(u, v) for u, v in zip(lead_notes, his_notes) if u is not None and v is not None]
    row = {'id': c['id'], 'kept': round(kept, 4), 'roots_same': f'{sum(p == q for p, q in zip(roots, roots_was))}/{len(halves)}', 'roots': ' '.join(NAMES[r] for r in roots),
           'under_last_word': NAMES[root(seg(x, at(15), at(15.5)))],
           'top_band': [round(db(band(seg(x, at(b), at(b + 1)), 6000, 14000)) - top_before, 1) for b in range(11, int(last))],
           'bars_db': [db(seg(x, at(b), at(b + 1))) for b in range(11, int(last))],
           'beat': {'verdict': bc['verdict'], 'jump_8th_ms': bc['jump_8th_ms'], 'accent_shift_8ths': bc['accent_shift_8ths'],
                    'first_strong_onset_ms': bc.get('first_strong_onset_ms'), 'keeps_groove': bc['new_keeps_groove']},
           'lead': notes_line(lead_notes), 'rub': f'{sum(1 for u, v in both if min((u - v) % 12, (v - u) % 12) in (1, 6))}/{len(both)}',
           'end_db': db(seg(x, len(x) / SR - .5, len(x) / SR))}
    print(json.dumps(row, ensure_ascii=False), flush=True)
