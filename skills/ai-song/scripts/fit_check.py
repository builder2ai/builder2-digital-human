"""How a vocal sits on its accompaniment: syllable onsets against the beat grid, syllables per bar, and the pitch against the
chord of each half bar. Use it before sending a take to a person, next to the word score: a take can transcribe perfectly
and still sound 'off' (怪) because it is off the grid or off the chords.

Onsets come from rises in the voice's energy. Grid fit = share of onsets within 40 ms of a 16th / 8th / beat
(by chance about 53% / 26% / 13%). Chord tones = share of voiced 50 ms frames on the root, third or fifth of the half bar's
chord, the chord taken from the accompaniment's bass peak and the stronger third (by chance about 25%). Rough, but it
separated a rap section a listener accepted from one they called 'off' in our song (see reference/listening.md).
Both files must be on the same clock (stems of one take, or a vocal and the backing it will be mixed on).
usage: fit_check.py <vocals> <accompaniment> --from S --to S [--bpm 99 --downbeat 0.303 | --map song_map.json] [--label name]
"""
import sys
sys.dont_write_bytecode = True
import argparse

import numpy as np

from songmap import SR, grid_from_args, load, seg

NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def onsets(x):
    m = x.mean(axis=1)
    hop, win = int(.005 * SR), int(.02 * SR)
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
            i = j + 16     # no two syllables within about 80 ms
        else:
            i += 1
    return np.array(out)


def pitch_track(x):
    m = x.mean(axis=1)
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


def chord_of(x):
    """Root and minor/major of a stretch of accompaniment: bass peak, then whichever third is stronger."""
    m = x.mean(axis=1)
    s = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    low = (f >= 45) & (f < 220)
    pcs = np.round(69 + 12 * np.log2(f[low] / 440)).astype(int) % 12
    root = int(np.array([(s[low][pcs == k] ** 2).sum() for k in range(12)]).argmax())
    mid = (f >= 200) & (f < 2000)
    pc = np.round(69 + 12 * np.log2(f[mid] / 440)).astype(int) % 12
    c = np.array([(s[mid][pc == k] ** 2).sum() for k in range(12)])
    return root, c[(root + 3) % 12] >= c[(root + 4) % 12]


ap = argparse.ArgumentParser()
ap.add_argument('vocals')
ap.add_argument('accompaniment')
ap.add_argument('--from', dest='a', type=float, required=True)
ap.add_argument('--to', dest='z', type=float, required=True)
ap.add_argument('--bpm', type=float)
ap.add_argument('--beats-per-bar', type=int, default=4)
ap.add_argument('--downbeat', type=float, default=0.)
ap.add_argument('--map')
ap.add_argument('--label', default='section')
args = ap.parse_args()
BAR, PHASE = grid_from_args(args)
voc, ins = load(args.vocals), load(args.accompaniment)
x = seg(voc, args.a, args.z)
t = onsets(x) + args.a
if not len(t):
    raise SystemExit('no onsets found')
sixteenth = BAR / 16
grid = lambda g: float(np.mean(np.minimum((t - PHASE) % g, g - (t - PHASE) % g) < .04))
bars = np.floor((t - PHASE) / BAR).astype(int)
per_bar = {int(b): int((bars == b).sum()) for b in np.unique(bars)}
hit = voiced = 0
notes = []
for i, h in enumerate(pitch_track(x)):
    if h <= 0:
        continue
    tt = args.a + i * .05
    b = np.floor((tt - PHASE) / BAR)
    half = int(((tt - PHASE) % BAR) // (BAR / 2))
    s0 = PHASE + b * BAR + half * BAR / 2
    root, minor = chord_of(seg(ins, s0, s0 + BAR / 2))
    midi = 69 + 12 * np.log2(h / 440)
    notes.append(midi)
    voiced += 1
    hit += (round(midi) - root) % 12 in (0, 3 if minor else 4, 7)
print(f'{args.label}: {len(t)} syllables in {args.z - args.a:.1f} s = {len(t) / (args.z - args.a):.2f}/s')
print(f'  on a 16th {grid(sixteenth):.2f} (chance .53), on an 8th {grid(2 * sixteenth):.2f} (.26), on a beat {grid(4 * sixteenth):.2f} (.13)')
print(f'  chord tones {hit / max(1, voiced):.2f} (chance .25), range {np.percentile(notes, 95) - np.percentile(notes, 5):.1f} semitones' if notes else '  no pitched frames')
print(f'  syllables per bar {per_bar}')
