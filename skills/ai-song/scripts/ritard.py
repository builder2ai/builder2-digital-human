"""Slow the end of a song down bit by bit (a ritardando), the voice and the accompaniment each in the way that suits it.

Stretching the finished mix with one phase vocoder sounded like a broken tape to our listener: every consonant, slide and
turn between notes slowed by the same factor and the reverb smeared. A singer who slows down holds the vowels and breaths
longer and keeps the consonants and note changes at speed. So:
- voice (dry, without reverb): WSOLA, the voice's own waveform laid out grain by grain, on a time map that gives the extra
  time only to steady frames (held vowels) and quiet ones (breaths). At every beat it meets the common map, so voice and
  accompaniment stay together. Add the reverb afterwards, at its own length.
- accompaniment: ffmpeg's rubberband with the tempo sent every 10 ms (checked on clicks: within about 10 ms of the map),
  transients 'mixed'.
- curve: the final ritard measured in performances (Friberg & Sundberg 1999), tempo = (1 + (w^q - 1) x)^(1/q), floored at
  --hold. Our listener wanted only 6-9% slower on the last two lines: --hold .93. Start it on a bar line where the drums
  have stopped, so they are never stretched.
Library: CommonMap, accompaniment, voice_map, wsola.
usage: ritard.py --voice dry_voice.wav --bed accompaniment.wav --out slowed.wav --from-bar 12 --to-bar 15.5 [--final .6 --hold .93 --q 2]
       [--bpm 99 --downbeat 0.303 | --map song_map.json]   (voice and bed on the same clock; before --from-bar nothing changes)
"""
import sys
sys.dont_write_bytecode = True
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 48000
STEP = .01            # seconds between tempo commands, and the grid of the common map
FRAME = .005          # seconds per analysis frame of the voice
WSOLA_N, WSOLA_HOP, WSOLA_TOL = 1536, 384, 480     # 32 ms grains, 8 ms apart, placed within 10 ms


class CommonMap:
    """Input seconds -> output seconds for a tempo curve tempo(t) (1 = as is, .6 = 60% of the speed)."""

    def __init__(self, seconds, tempo):
        self.steps = np.arange(0, seconds + STEP, STEP)
        self.tempo = np.asarray(tempo(self.steps), float)
        self.out = np.concatenate([[0], np.cumsum(STEP / self.tempo)[:-1]])

    def __call__(self, t):
        return np.interp(t, self.steps, self.out)


def accompaniment(x, cmap, transients='mixed'):
    """x stretched by ffmpeg's rubberband, the tempo following cmap."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'rit.txt').write_text(''.join(f'{t:.3f} rubberband tempo {v:.5f};\n' for t, v in zip(cmap.steps, cmap.tempo)), encoding='ascii')
        y = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-af',
                            f'asetnsamples=n={int(STEP * SR)}:p=0,asendcmd=f=rit.txt,rubberband=tempo=1:transients={transients}:detector=compound:channels=together',
                            '-f', 'f32le', '-'], input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True, check=True, cwd=d).stdout
    return np.frombuffer(y, np.float32).reshape(-1, 2).copy()


def steadiness(x):
    """Per FRAME of x: 1 where the voice holds still (a held vowel) or is quiet, towards 0 where it changes fast."""
    m = x.mean(axis=1)
    hop, n = int(FRAME * SR), 1024
    pad = np.concatenate([np.zeros(n // 2), m, np.zeros(n)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, n)[::hop][:len(m) // hop + 1]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1))
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f > 80) & (f < 6000)
    level = 20 * np.log10(spec[:, band] + 1e-7)
    top = np.percentile(level.max(axis=1), 95)
    level = np.maximum(level, top - 60)
    flux = np.r_[0, np.abs(np.diff(level, axis=0)).mean(axis=1)]
    energy = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    loud = energy > np.percentile(energy, 95) - 30
    lo, hi = np.percentile(flux[loud], 25), np.percentile(flux[loud], 85)
    w = np.clip(1 - (flux - lo) / (hi - lo), 0, 1)
    w[~loud] = 1.
    w = np.convolve(w, np.ones(5) / 5, mode='same')
    return np.maximum(w, .05)


def voice_map(x, cmap, anchors):
    """Input and output seconds at the frame edges of x: between two anchors the extra time goes to the steady frames."""
    w = steadiness(x)
    t_in = np.arange(len(w) + 1) * FRAME
    d_out = np.full(len(w), FRAME)
    for a, b in zip(anchors[:-1], anchors[1:]):
        k = slice(int(round(a / FRAME)), int(round(b / FRAME)))
        extra = float(cmap(b) - cmap(a) - (b - a))
        if extra > 1e-6:
            d_out[k] += extra * w[k] / w[k].sum()
    after = int(round(anchors[-1] / FRAME))
    mid = (t_in[after:-1] + t_in[after + 1:]) / 2
    d_out[after:] = FRAME / np.interp(mid, cmap.steps, cmap.tempo)
    return t_in, np.concatenate([[0], np.cumsum(d_out)])


def wsola(x, t_in, t_out):
    """x laid out on the map t_in -> t_out (seconds, increasing) by WSOLA."""
    n_out = int(t_out[-1] * SR)
    N, H, tol = WSOLA_N, WSOLA_HOP, WSOLA_TOL
    xp = np.concatenate([np.zeros((N + tol, 2), np.float32), x, np.zeros((2 * N + tol, 2), np.float32)])
    off = N + tol
    m = xp.mean(axis=1).astype(np.float64)
    e = np.concatenate([[0], np.cumsum(m ** 2)])
    win = np.hanning(N + 1)[:N]
    out = np.zeros((n_out + N, 2))
    norm = np.zeros(n_out + N)
    prev = None
    for k in range(-N // 2, n_out, H):
        centre = (k + N / 2) / SR
        p = off + int(round(np.interp(centre, t_out, t_in) * SR)) - N // 2
        if prev is None:
            a = p
        else:
            nat = prev + H
            tpl = m[nat:nat + N]
            lo, hi = max(0, p - tol), min(len(m) - N, p + tol)
            c = np.correlate(m[lo:hi + N], tpl, 'valid')
            en = e[lo + N:hi + N + 1] - e[lo:hi + 1]
            a = lo + int(np.argmax(c / np.sqrt(en + 1e-9))) if tpl.dot(tpl) > 1e-10 else p
        o0, o1 = max(k, 0), k + N
        g0 = o0 - k
        out[o0:o1] += xp[a + g0:a + N] * win[g0:, None]
        norm[o0:o1] += win[g0:]
        prev = a
    out = out[:n_out] / np.maximum(norm[:n_out], 1e-3)[:, None]
    return out.astype(np.float32)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from songmap import grid_from_args, load, write
    ap = argparse.ArgumentParser()
    ap.add_argument('--voice', required=True)
    ap.add_argument('--bed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--from-bar', type=float, required=True)
    ap.add_argument('--to-bar', type=float, required=True)
    ap.add_argument('--final', type=float, default=.6)
    ap.add_argument('--hold', type=float, default=.93)
    ap.add_argument('--q', type=float, default=2.)
    ap.add_argument('--bpm', type=float)
    ap.add_argument('--beats-per-bar', type=int, default=4)
    ap.add_argument('--downbeat', type=float, default=0.)
    ap.add_argument('--map')
    a = ap.parse_args()
    BAR, PHASE = grid_from_args(a)
    voice, bed = load(a.voice), load(a.bed)
    n = max(len(voice), len(bed))
    voice = np.pad(voice, ((0, n - len(voice) + SR), (0, 0)))
    bed = np.pad(bed, ((0, n - len(bed) + SR), (0, 0)))
    # only the tail is processed, from a bar before the curve; up to the curve it is unchanged, so it joins half a bar in
    i0 = max(0, int(round((PHASE + (a.from_bar - 1) * BAR) * SR)))
    off = i0 / SR
    bar_of = lambda t: (t + off - PHASE) / BAR

    def tempo(t):
        x = np.clip((bar_of(t) - a.from_bar) / (a.to_bar - a.from_bar), 0, 1)
        return np.maximum((1 + (a.final ** a.q - 1) * x) ** (1 / a.q), a.hold)

    tv, tb = voice[i0:], bed[i0:]
    cmap = CommonMap(len(tb) / SR, tempo)
    beats = PHASE + np.arange(a.from_bar, a.to_bar + 1e-9, 1 / 4) * BAR - off
    slow_bed = accompaniment(tb, cmap)
    t_in, t_out = voice_map(tv, cmap, beats)
    slow_voice = wsola(tv, t_in, t_out)
    k = min(len(slow_bed), len(slow_voice))
    tail = slow_bed[:k] + slow_voice[:k]
    cut, xf = int(.5 * BAR * SR), int(.06 * SR)
    before = (bed + voice)[:i0 + cut + xf]
    ramp = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
    joint = before[i0 + cut:] * (1 - ramp) + tail[cut:cut + xf] * ramp
    write(np.concatenate([before[:i0 + cut], joint, tail[cut + xf:]]), a.out)
    t = lambda b: float(np.interp(PHASE + b * BAR - off, t_in, t_out))
    print(a.out, f'bars {a.from_bar}-{a.to_bar} now last {t(a.to_bar) - t(a.from_bar):.2f} s instead of {(a.to_bar - a.from_bar) * BAR:.2f} s')


if __name__ == '__main__':
    main()
