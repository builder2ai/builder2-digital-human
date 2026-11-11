"""Print the lead line of an instrumental (an interlude, a new intro) on a 16th-note grid, one row per bar, to judge whether
it has a motif, rests and long notes. The strongest peak from 500 Hz to 3 kHz in each slot; '.' where nothing stands out.
A chord-match score misreads a lead line, so read the notes instead.
usage: lead_notes.py <file> --bars 0-8 [--bpm 99 --downbeat 0.303 | --map song_map.json]
"""
import sys
sys.dont_write_bytecode = True
import argparse

import numpy as np

from songmap import SR, grid_from_args, load, seg

NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
WIN = 4096
FREQ = np.fft.rfftfreq(WIN, 1 / SR)
BAND = (FREQ >= 500) & (FREQ < 3000)

ap = argparse.ArgumentParser()
ap.add_argument('file')
ap.add_argument('--bars', default='0-8')
ap.add_argument('--bpm', type=float)
ap.add_argument('--beats-per-bar', type=int, default=4)
ap.add_argument('--downbeat', type=float, default=0.)
ap.add_argument('--map')
args = ap.parse_args()
BAR, PHASE = grid_from_args(args)
x = load(args.file)
b0, b1 = (int(v) for v in args.bars.split('-'))
for k in range(b0, b1):
    m = seg(x, PHASE + k * BAR, PHASE + (k + 1) * BAR).mean(axis=1)
    row = []
    for s in range(16):
        c = int((s + .5) * BAR / 16 * SR)
        w = m[max(0, c - WIN // 2):max(0, c - WIN // 2) + WIN]
        if len(w) < WIN:
            break
        sp = np.abs(np.fft.rfft(w * np.hanning(WIN)))[BAND]
        p = sp.argmax()
        if sp[p] / (np.median(sp) + 1e-6) < 25:
            row.append('.')
        else:
            midi = int(round(69 + 12 * np.log2(FREQ[BAND][p] / 440)))
            row.append(f'{NAMES[midi % 12]}{midi // 12 - 1}')
    print(f'{k:>4} ' + ' '.join(f'{n:>4}' for n in row))
