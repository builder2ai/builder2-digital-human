"""The song map: one small JSON file every script reads for the song's grid and sections, so bar lengths and section times
are written down once instead of in every script.

It has two stages:
- a plan, while writing lyrics: "measured": false (or no downbeat). Tempo from the style card, each section's bars, or its
  lines and bars per line. Use it to check the song's length against the target (run this file on it). Scripts that need
  real times refuse a plan.
- measured, after the takes are joined: "measured": true and a downbeat = the time in seconds of a bar line (bar 0) on the
  song's clock. Measure it; the tempo in a style text is not reliable.

  {"bpm": 108, "beats_per_bar": 4, "downbeat": null, "measured": false, "key": "G minor", "target_seconds": 30,
   "sections": [{"name": "intro", "bars": 4, "who": "", "manner": "instrumental"},
                {"name": "verse", "lines": 8, "bars_per_line": 1, "who": "female", "manner": "sung"},
                {"name": "chorus", "lines": 4, "bars_per_line": 2, "who": "male", "manner": "sung"}]}
  start_bar may be given; otherwise sections follow each other.

Also the audio helpers the other scripts share: load (any file -> 48 kHz float32 through ffmpeg), seg, write, db.
The map file is found from the argument, else the SONG_MAP environment variable, else ./song_map.json.
usage: songmap.py [song_map.json] [--target 30]    prints each section's length in bars and seconds, and the total
"""
import sys
sys.dont_write_bytecode = True
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 48000


def load(path, sr=SR, channels=2):
    """Decode any audio file with ffmpeg to float32, shape (samples, channels)."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', str(channels), '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).copy()


def seg(x, a, b, sr=SR):
    """x from a to b seconds."""
    return x[max(0, int(round(a * sr))):max(0, int(round(b * sr)))]


def write(x, path, sr=SR, codec='pcm_s24le'):
    """Write (samples, channels) float audio with ffmpeg; the codec follows the extension unless given."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    args = ['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(sr), '-ac', str(x.shape[1]), '-i', 'pipe:0']
    if str(path).endswith('.wav'):
        args += ['-c:a', codec]
    subprocess.run(args + [str(path)], input=np.ascontiguousarray(x).tobytes(), check=True)


def db(x):
    return round(float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)), 1)


class SongMap:
    def __init__(self, data):
        self.data = data
        self.bpm = float(data['bpm'])
        self.beats_per_bar = int(data.get('beats_per_bar', 4))
        d = data.get('downbeat')
        self.measured = bool(data.get('measured', d is not None)) and d is not None
        self.downbeat = float(d) if d is not None else None
        self.key = data.get('key', '')
        self.target = data.get('target_seconds')
        self.sections = []
        at = 0.
        for s in data.get('sections', []):
            s = dict(s)
            if 'bars' not in s and 'lines' in s:
                s['bars'] = s['lines'] * s.get('bars_per_line', 1)
            s.setdefault('start_bar', at)
            at = s['start_bar'] + s.get('bars', 0)
            self.sections.append(s)

    @property
    def beat_seconds(self):
        return 60. / self.bpm

    @property
    def bar_seconds(self):
        return self.beat_seconds * self.beats_per_bar

    def _need_measured(self):
        if not self.measured:
            raise SystemExit('song_map.json is a plan ("measured": false or no downbeat). Measure the downbeat on the joined song first.')

    def bar(self, k):
        """Seconds of bar k (may be fractional) on the song's clock."""
        self._need_measured()
        return self.downbeat + k * self.bar_seconds

    def bar_of(self, t):
        self._need_measured()
        return (t - self.downbeat) / self.bar_seconds

    def section(self, name):
        for s in self.sections:
            if s['name'] == name:
                return s
        raise KeyError(name)

    def section_span(self, name):
        s = self.section(name)
        return self.bar(s['start_bar']), self.bar(s['start_bar'] + s['bars'])

    def total_bars(self):
        return max((s['start_bar'] + s.get('bars', 0) for s in self.sections), default=0)


def load_map(path=None):
    path = Path(path or os.environ.get('SONG_MAP', 'song_map.json'))
    return SongMap(json.loads(path.read_text(encoding='utf-8')))


def grid_from_args(args):
    """(bar seconds, downbeat) from --bpm/--downbeat if given, else from a measured song map."""
    if getattr(args, 'bpm', None):
        bar = 60. / args.bpm * getattr(args, 'beats_per_bar', 4)
        return bar, float(getattr(args, 'downbeat', 0.) or 0.)
    m = load_map(getattr(args, 'map', None))
    m._need_measured()
    return m.bar_seconds, m.downbeat


def main():
    args = [a for a in sys.argv[1:]]
    target = None
    if '--target' in args:
        i = args.index('--target')
        target = float(args[i + 1])
        del args[i:i + 2]
    m = load_map(args[0] if args else None)
    target = target or m.target
    bs = m.bar_seconds
    print(f"{m.bpm} BPM, {m.beats_per_bar} beats a bar: a bar is {bs:.3f} s ({'measured' if m.measured else 'plan'})")
    for s in m.sections:
        if 'lines' in s and s['lines']:
            per = s.get('bars_per_line', s['bars'] / s['lines'])
            how = f" ({s['lines']} lines x {per:g} bars)" if 'bars_per_line' in s else f" ({s['lines']} lines, {per:.2g} bars each)"
        else:
            how = ''
        print(f"  {s['name']:<12} bars {s['start_bar']:>5}-{s['start_bar'] + s['bars']:<5} {s['bars']:>5} bars{how:<24} {s['bars'] * bs:6.1f} s")
    total = m.total_bars() * bs
    print(f'  total {m.total_bars()} bars = {total:.1f} s')
    if target:
        diff = total - target
        print(f'  target {target:.0f} s: {"over" if diff > 0 else "under"} by {abs(diff):.1f} s ({abs(diff) / bs:.1f} bars)')
        if diff > .1 * target:
            print('  -> cut lines or sections, sing more lines per bar, or ask the person to change the target; a faster tempo changes the style')


if __name__ == '__main__':
    main()
