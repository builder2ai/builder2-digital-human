"""Cut one take down to the lyric: from a bar line before the first lyric line to a bar line after the last line of a
chosen section, leaving out lead-in hums, ad-libs and a second pass of the song. For form samples (one whole take, no
join) and for the section a continuation starts from.

YuE2 does not stop at the end of the lyric. A bare [Instrumental] after the last section did not stop it singing in one
test: one take sang the whole song again, one ad-libbed for 16 s, others hummed to the end.
- the cut points come from the words (take_sections.py) and sit on the take's own bar grid (take_grid.py --stems --sections:
  the row measured on the backing, over the lyric lines). Start: --lead-bars bar lines before the bar holding the first sung
  character (a pickup up to 0.15 bar before a bar line belongs to that bar). End: the bar line after the last character of
  the --through section, plus --tail-bars bars; pulled back to a bar line if the song starts again before that;
- 0.3 s fade-in when the cut starts after 0, a fade over the last bar, then gain so the true peak (4x oversampled) sits at
  --peak dBTP; no limiter;
- --lufs -16: gain to that integrated loudness instead, for a set of samples to compare (ACE-Step came out at -12.8 LUFS
  and YuE2 at -16.4 at the same peak, and the louder one tends to win an A/B). The true peak still never goes over --peak:
  if it would, the gain stops there and it says how loud the sample got; give the whole set the quietest of those values;
- --grid-from mix uses the grid take_grid.py measured on the mix (grid_mix) when the backing's tempo is wrong (take_grid
  flags tempo_agree false, e.g. 135 vs 103 BPM on a 3+3+2 groove);
- prints and writes <out>.json: where the cut sits in the take, where the singing sits in the sample, gain, loudness, peak.
numpy and ffmpeg only.
usage: cut_take.py takes/<take>.flac --grid takes.json --sections sections.json --out sample.wav [--through chorus]
                   [--lead-bars 1] [--tail-bars 1] [--peak -1.2] [--lufs -16] [--grid-from stem|mix] [--no-fade]
                   [--from-s S --to-s S]
  --through  cut after the last line of this section (default: the last section that was found)
  --from-s / --to-s  the sung span by hand, in seconds of the take (instead of --sections)
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def true_peak(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ar', '192000', '-ac', '2', '-f', 'f32le', '-'],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return float(np.abs(np.frombuffer(raw, np.float32)).max())


def loudness(path):
    """(integrated LUFS, true peak dBFS) from ffmpeg's ebur128."""
    lo = subprocess.run(['ffmpeg', '-v', 'info', '-nostdin', '-i', str(path), '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                        capture_output=True, text=True).stderr
    li, pk = re.findall(r'I:\s+(-?[\d.]+) LUFS', lo), re.findall(r'Peak:\s+(-?[\d.]+) dBFS', lo)
    return (float(li[-1]) if li else None), (float(pk[-1]) if pk else None)


def grid_of(row, which):
    """(bar seconds, a downbeat) from take_grid.py's row: the backing's grid, or the mix's (grid_mix)."""
    if which == 'mix':
        g = row.get('grid_mix')
        if not g:
            raise SystemExit('no grid_mix in this takes.json row: run take_grid.py --stems again (it stores the mix grid too)')
        return g['bar_s'], g['downbeat']
    return row['bar_s'], row['downbeat']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('take')
    ap.add_argument('--grid', required=True)
    ap.add_argument('--sections')
    ap.add_argument('--out', required=True)
    ap.add_argument('--through')
    ap.add_argument('--lead-bars', type=int, default=1)
    ap.add_argument('--tail-bars', type=int, default=1)
    ap.add_argument('--peak', type=float, default=-1.2)
    ap.add_argument('--lufs', type=float, help='integrated loudness to match a comparison set, e.g. -16 (the peak cap still holds)')
    ap.add_argument('--grid-from', choices=('stem', 'mix'), default='stem')
    ap.add_argument('--no-fade', action='store_true')
    ap.add_argument('--from-s', type=float)
    ap.add_argument('--to-s', type=float)
    a = ap.parse_args()
    tid = Path(a.take).stem
    grid = {r['id']: r for r in json.load(open(a.grid, encoding='utf-8'))}.get(tid)
    if not grid:
        raise SystemExit(f'{tid} is not in {a.grid}')
    warn = []
    if grid.get('measured_on') != 'backing':
        warn.append('the grid was measured on the mix only; measure it on the stems (take_grid.py --stems --sections)')
    if grid.get('tempo_agree') is False and a.grid_from == 'stem':
        warn.append(f"mix and backing disagree on the tempo ({grid.get('bpm_mix')} vs {grid['bpm']}): the cut uses the backing's grid; "
                    f"if the mix is right, run again with --grid-from mix")
    bar, d = grid_of(grid, a.grid_from)
    dur = grid['seconds']
    repeat = None
    if a.from_s is not None and a.to_s is not None:
        s0, z = a.from_s, a.to_s
    else:
        if not a.sections:
            raise SystemExit('give --sections (take_sections.py) or --from-s and --to-s')
        sec = {r['id']: r for r in json.load(open(a.sections, encoding='utf-8'))}.get(tid)
        if not sec or sec.get('sung_from') is None:
            raise SystemExit(f'no lyric found for {tid} in {a.sections}')
        found = [s for s in sec['sections'] if s.get('start') is not None]
        last = next((s for s in found if s['name'].lower() == a.through.lower()), None) if a.through else found[-1]
        if last is None:
            raise SystemExit(f"section {a.through} not found in {tid}; found: {', '.join(s['name'] for s in found)}")
        s0, z = sec['sung_from'], last['end']
        if sec.get('cut_off'):
            warn.append('take_sections says the take is cut off: ' + sec['cut_off'])
        repeat = (sec.get('after') or {}).get('repeat_from') if last is found[-1] else None
    first_bar = d + math.floor((s0 + .15 * bar - d) / bar) * bar
    start = max(0., first_bar - a.lead_bars * bar)
    end = d + math.ceil((z - .1 - d) / bar) * bar + a.tail_bars * bar
    if repeat and repeat < end:
        end2 = d + math.floor((repeat - d) / bar) * bar
        if end2 > z:
            end = end2
        else:
            warn.append(f'the song starts again at {repeat:.1f} s, inside the last bar: the tail holds its first notes')
    if end > dur:
        warn.append(f'the take ends at {dur:.2f} s, before the planned end {end:.2f} s: no tail bar')
        end = dur
    tmp = str(Path(a.out).with_suffix('.tmp.wav'))
    af = f'atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS'
    if not a.no_fade:
        if start > 0:
            af += ',afade=t=in:st=0:d=0.3'
        af += f',afade=t=out:st={max(0., end - start - bar):.4f}:d={min(bar, end - start):.4f}:curve=qsin'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.take, '-af', af, '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s24le', tmp],
                   check=True, stdin=subprocess.DEVNULL)
    gain_peak = a.peak - 20 * math.log10(true_peak(tmp) + 1e-12)
    gain = gain_peak
    if a.lufs is not None:
        before = loudness(tmp)[0]
        gain = a.lufs - before
        if gain > gain_peak + .05:
            warn.append(f'reaching {a.lufs:g} LUFS would push the true peak over {a.peak:g} dBTP; stopped at '
                        f'{before + gain_peak:.1f} LUFS: give the whole comparison set --lufs {math.floor((before + gain_peak) * 10) / 10:g} or lower')
            gain = gain_peak
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', f'volume={gain:.2f}dB', '-c:a', 'pcm_s24le', a.out],
                   check=True, stdin=subprocess.DEVNULL)
    Path(tmp).unlink()
    li, pk = loudness(a.out)
    info = {'take': a.take, 'from_take_s': round(start, 3), 'to_take_s': round(end, 3), 'seconds': round(end - start, 2),
            'sung_from_in_sample_s': round(s0 - start, 2), 'sung_to_in_sample_s': round(z - start, 2),
            'bars': round((end - start) / bar, 2), 'sung_bars': round((z - s0) / bar, 1), 'gain_db': round(gain, 2),
            'integrated_lufs': li, 'true_peak_dbtp': pk, 'lufs_target': a.lufs, 'grid_from': a.grid_from,
            'grid': {k: grid.get(k) for k in ('bpm', 'bpm_mix', 'bar_s', 'downbeat', 'beat_fit_sd_ms', 'key', 'key_next', 'key_gap')},
            'grid_used': {'bar_s': bar, 'downbeat': d}, 'warnings': warn}
    Path(a.out).with_suffix('.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
