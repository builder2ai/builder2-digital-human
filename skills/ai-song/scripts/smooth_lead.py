"""Take the cracked, buzzing edge off a lead instrument in bars that ACE-Step newly wrote, leaving the tune, the drums, the
bass and the bars around them as they are.

In our takes each new lead note (an erhu) brought short spikes at its attack and a block of buzz above about 4.5 kHz that
the song's own accompaniment did not have. Lowering CFG or doubling the steps changed the tune and cleaned nothing, so this
works afterwards, on the lead's part only: split the take with `separate.py --model parts` first. Above 800 Hz any moment
that stands out from its own surroundings in time is brought down to them, and the top is turned down; the phase stays.
The change fades in and out at the edges of the region. Listen afterwards: a measure that improves is not proof the crack
is gone.
usage: smooth_lead.py <take.wav> --parts <dir with <take>_other.wav and _voice.wav> --from S --to S --out new.wav [--strength 0-1]
  --from/--to: the new bars, on the take's clock. Demucs files a lead instrument under 'other' or 'voice'; both are treated.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as win

from songmap import SR, load, write

N, HOP = 2048, 256
SPIKE_FROM = 800.        # below this the lead's body stays as it is
SPIKE_FRAMES = 15        # about 90 ms: a moment's surroundings
SPIKE_ROOM = 1.25        # how far a bin may stand above its surroundings before it is brought down
SHELF_FROM, SHELF_TO = 3500., 5500.
TOP_CUT_DB = 14.         # at full strength
EDGE_FADE = .25


def stft(x):
    return np.fft.rfft(win(np.pad(x, (N, N)), N)[::HOP] * np.hanning(N), axis=1)


def istft(S, length):
    w = np.hanning(N)
    out = np.zeros(HOP * (len(S) - 1) + N)
    norm = np.zeros_like(out)
    for k, f in enumerate(np.fft.irfft(S, n=N, axis=1) * w):
        out[k * HOP:k * HOP + N] += f
        norm[k * HOP:k * HOP + N] += w ** 2
    return (out / np.maximum(norm, 1e-6))[N:N + length]


def median_in_time(a, size):
    half = size // 2
    return np.median(win(np.pad(a, ((half, half), (0, 0)), mode='edge'), size, axis=0), axis=-1)


ap = argparse.ArgumentParser()
ap.add_argument('take')
ap.add_argument('--parts', required=True)
ap.add_argument('--from', dest='a', type=float, required=True)
ap.add_argument('--to', dest='z', type=float, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--strength', type=float, default=1.)
args = ap.parse_args()
name = Path(args.take).stem
x = load(args.take).astype(np.float64)
lead = load(Path(args.parts) / f'{name}_other.wav').astype(np.float64)
vp = Path(args.parts) / f'{name}_voice.wav'
if vp.exists():
    v = load(vp).astype(np.float64)
    k = min(len(lead), len(v))
    lead = lead[:k] + v[:k]
n = min(len(x), len(lead))
a, z = int(args.a * SR), min(n, int(args.z * SR))
f = np.fft.rfftfreq(N, 1 / SR)
above = np.clip((f - SPIKE_FROM * .8) / (SPIKE_FROM * .2), 0, 1)
top = 10 ** (-args.strength * TOP_CUT_DB * np.clip((f - SHELF_FROM) / (SHELF_TO - SHELF_FROM), 0, 1) / 20)
removed = np.zeros((n, 2))
reach = min(n, z + N)
for c in range(2):
    S = stft(lead[:reach, c])
    mag = np.abs(S)
    limit = median_in_time(mag, SPIKE_FRAMES) * SPIKE_ROOM
    gain = np.where(mag > limit, limit / (mag + 1e-12), 1.)
    gain = 1 - args.strength * above * (1 - gain)
    removed[:reach, c] = lead[:reach, c] - istft(S * gain * top, reach)
fade = int(EDGE_FADE * SR)
amount = np.zeros(n)
amount[a:z] = 1
amount[max(a, z - fade):z] = np.linspace(1, 0, z - max(a, z - fade))
if a > 0:
    amount[a:a + fade] = np.linspace(0, 1, fade)
result = x.copy()
result[:n] -= removed * amount[:, None]
write(result, args.out)
level = lambda y: 20 * np.log10(np.sqrt((y[a:z] ** 2).mean()) + 1e-12)
print(args.out, 'treated', round(a / SR, 2), '-', round(z / SR, 2), 's; level change', round(level(result) - level(x), 2), 'dB; peak', round(float(np.abs(result).max()), 3))
