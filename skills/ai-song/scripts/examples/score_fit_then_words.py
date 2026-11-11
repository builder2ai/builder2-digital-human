"""EXAMPLE from our song: two-stage screening. Beat and chord fit (cheap) for every take first, then Whisper only for the
best of them. The chord table, bar numbers, times and lines are that song's (a rap section re-sung from 官场 onward);
fit_check.py and score_takes.py are the general versions of the two stages. SONG_DIR holds stems/ and review/.

Which re-sung 官场 onward sits best on its accompaniment? First the beat and chord measures for all takes, then the words for the best of them."""
import sys
sys.dont_write_bytecode = True
from difflib import SequenceMatcher
from pathlib import Path
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pypinyin import lazy_pinyin  # noqa: E402
from songmap import SR, load, load_map, seg  # noqa: E402
from score_lines import load_audio, load_model, SR as WSR  # noqa: E402

V8 = Path(os.environ.get('SONG_DIR', '.')).resolve()
_M = load_map()
D_BAR, d_bar = _M.bar_seconds, _M.bar

S = V8 / 'stems'
CLIP, KEEP = 57.2, 64.92
LINES = ['官场容不下我', '山水还容得下我', '月亮又没得罪我', '我跟它摆什么脸色', '人生已经够难了', '难过的事', '明天再说']
SIXTEENTH = D_BAR / 16
CHORDS = {25: 'Dm', 26: 'Am', 27: 'Gm', 28: 'Dm', 29: 'Dm', 30: 'Am', 31: 'Gm', 32: 'Dm'}   # first half of the bar, from the bed
TONES = {'Dm': {2, 5, 9}, 'Am': {9, 0, 4}, 'Gm': {7, 10, 2}, 'C': {0, 4, 7}}
SECOND_HALF_27 = 'C'
ANALYSIS_TO = 80.3


def onsets(m):
    hop = int(.005 * SR)
    win = int(.02 * SR)
    e = np.array([np.sqrt((m[i:i + win] ** 2).mean()) for i in range(0, len(m) - win, hop)])
    e = np.convolve(e, np.ones(4) / 4, 'same')
    floor = .12 * np.percentile(e, 95)
    out, i = [], 8
    while i < len(e) - 8:
        if e[i] > floor and e[i] > 1.6 * e[i - 8] + .004 and e[i] >= e[i - 1]:
            j = i
            while j + 1 < len(e) and e[j + 1] >= e[j]:
                j += 1
            out.append((i - 4) * .005)
            i = j + 16
        else:
            i += 1
    return np.array(out)


def chord_at(t):
    bar = int((t - d_bar(0)) / D_BAR)
    half = ((t - d_bar(0)) % D_BAR) >= D_BAR / 2
    if bar in (27, 31) and half:
        return 'C'
    return CHORDS.get(bar, 'Dm')


def pitches(m):
    n = int(.05 * SR)
    out = []
    for i in range(0, len(m) - n, n):
        w = m[i:i + n] - m[i:i + n].mean()
        if np.sqrt((w ** 2).mean()) < .02:
            out.append(0.)
            continue
        ac = np.correlate(w, w, 'full')[n - 1:]
        lo, hi = int(SR / 420), min(int(SR / 75), n - 1)
        k = lo + ac[lo:hi].argmax()
        out.append(SR / k if ac[k] > .4 * ac[0] else 0.)
    return np.array(out)


TONES['Am'] = {9, 0, 4}
spec = json.loads((V8 / sys.argv[1]).read_text(encoding='utf-8'))
rows = []
for v in spec['variants']:
    take = v['id']
    a = load(S / f'{take}_vocals.flac').mean(axis=1)
    b = load(S / f'{take}_instrumental.flac')
    x = a[int((KEEP - CLIP) * SR):int((ANALYSIS_TO - CLIP) * SR)]
    t = onsets(x) + KEEP
    words_end = t[-1] if len(t) else 0
    sung = t[t < 76]
    g16 = float(np.mean(np.minimum((sung - d_bar(0)) % SIXTEENTH, SIXTEENTH - (sung - d_bar(0)) % SIXTEENTH) < .04)) if len(sung) else 0
    g8 = float(np.mean(np.minimum((sung - d_bar(0)) % (2 * SIXTEENTH), 2 * SIXTEENTH - (sung - d_bar(0)) % (2 * SIXTEENTH)) < .04)) if len(sung) else 0
    hz = pitches(x)
    hit, n = 0, 0
    notes = []
    for i, h in enumerate(hz):
        if h > 0:
            when = KEEP + i * .05
            midi = 69 + 12 * np.log2(h / 440)
            notes.append(midi)
            n += 1
            hit += (round(midi) % 12) in TONES[chord_at(when)]
    e = np.sqrt((x[:len(x) // 2] ** 2).mean()), np.sqrt((x[len(x) // 2:] ** 2).mean())
    tail = a[int((79.0 - CLIP) * SR):int((80.3 - CLIP) * SR)]
    over = a[int((80.5 - CLIP) * SR):int((83. - CLIP) * SR)]
    bars = [float(np.sqrt((seg(b, d_bar(k) - CLIP, d_bar(k + 1) - CLIP) ** 2).mean())) for k in range(27, 33)]
    row = {'id': take, 'rate': round(len(sung) / (76 - KEEP), 2), 'g16': round(g16, 2), 'g8': round(g8, 2), 'tones': round(hit / max(1, n), 2),
           'range': round(float(np.percentile(notes, 95) - np.percentile(notes, 5)), 1) if notes else 0, 'tail_rms': round(float(np.sqrt((tail ** 2).mean())), 3), 'over_rms': round(float(np.sqrt((over ** 2).mean())), 3), 'bed_min': round(min(bars), 3)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
# words for the best by beat and chord fit
rows.sort(key=lambda r: r['g16'] + r['g8'] + r['tones'], reverse=True)


def folded(text):
    return [p.replace('ing', 'in').replace('eng', 'en') for p in lazy_pinyin(text)]


def found(line, heard):
    want, got = folded(line), folded(heard)
    best = 0
    for i in range(max(1, len(got) - len(want) + 1)):
        best = max(best, SequenceMatcher(None, want, got[i:i + len(want)]).ratio())
    return round(best, 2)


m, _ = load_model()
print('--- words')
for r in [r for r in rows if r['bed_min'] > .08][:18]:
    w = load_audio(S / f"{r['id']}_vocals.flac")
    texts = []
    for t0 in (64.6, 63.6):
        stream, _ = m.transcribe(w[int((t0 - CLIP) * WSR):], language='zh', beam_size=5, vad_filter=False, condition_on_previous_text=False)
        texts.append(re.sub(r'\W', '', ''.join(s.text for s in stream)))
    r['lines'] = [round(float(np.mean([found(l, t) for t in texts])), 2) for l in LINES]
    r['words'] = round(float(np.mean(r['lines'])), 2)
    r['heard'] = texts[0]
    print(json.dumps(r, ensure_ascii=False), flush=True)
(V8 / 'review' / f'stage6_{Path(sys.argv[1]).stem.split("_")[-1].lower()}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
