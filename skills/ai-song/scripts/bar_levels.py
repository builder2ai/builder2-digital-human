"""Level per bar (or half bar) of a mix, and of its voice bus if given, plus the delivery checks that catch most complaints
of 'no accompaniment here' and 'it doesn't join': quiet bars (under 40% of the median), the true peak (read on the stereo
file: ffmpeg's mono downmix can read about 1.4 times too high) and the largest sample jumps (clicks at a splice).
A quiet bar is not always a hole: a take may have a written break there (in one test the take's own pre-verse half bar sat
at -30.4 dB and the new intro's at -27.8 dB). With --ref (the take's own mix) and --ref-offset (ref seconds = mix seconds -
offset), each quiet bar is compared with the take's bar at the same place, both against their own file's median (the mix
may carry another gain): 'the take has it too' when that bar is also under 40% of the take's median over the same
stretch, or the new bar is not more than 3 dB further under; 'hole' otherwise.
Library: levels(...) and quiet(...) for assemble.py.
usage: bar_levels.py <mix> [--voice <voice bus on the same clock>] [--bars 0-120] [--half] [--bpm 99 --downbeat 0.303 | --map song_map.json]
                     [--ref take.flac --ref-offset 2.538]
"""
import sys
sys.dont_write_bytecode = True
import argparse  # noqa: E402

import numpy as np  # noqa: E402

from songmap import SR, db, grid_from_args, load, seg  # noqa: E402


def levels(mix, bar, phase, step, b0, b1):
    """[(bar number, seconds, dB)] from bar b0 to b1 in steps of `step` seconds."""
    return [(float(b), float(phase + b * bar), db(seg(mix, phase + b * bar, phase + b * bar + step))) for b in np.arange(b0, b1, step / bar)]


def quiet(rows, step, ref=None, ref_offset=0.):
    """The quiet ones (under 40% of the median) of levels(); with ref, each compared with ref's own level at the same place
    (ref seconds = seconds - ref_offset). Returns [{bar, s, db, ref_db, verdict}]."""
    rms = np.array([10 ** (lv / 20) for _, _, lv in rows])
    med = float(np.median(rms)) if len(rms) else 0.
    out = []
    ref_rows = None
    if ref is not None:
        span = [(t - ref_offset) for _, t, _ in rows]
        ref_rows = [db(seg(ref, t, t + step)) if 0 <= t and t + step <= len(ref) / SR else None for t in span]
        valid = [10 ** (v / 20) for v in ref_rows if v is not None]
        ref_med = float(np.median(valid)) if valid else 0.
    for i, ((b, t, lv), r) in enumerate(zip(rows, rms)):
        if r >= .4 * med:
            continue
        q = {'bar': round(b, 2), 's': round(t, 2), 'db': lv, 'under_median_db': round(lv - 20 * np.log10(med + 1e-12), 1)}
        if ref_rows is not None:
            rv = ref_rows[i]
            q['ref_db'] = rv
            if rv is None:
                q['verdict'] = 'no take audio here: listen'
                out.append(q)
                continue
            # levels against each file's own median, since the mix may carry a different gain than the take
            q['ref_under_median_db'] = round(rv - 20 * np.log10(ref_med + 1e-12), 1)
            if 10 ** (rv / 20) < .4 * ref_med or q['under_median_db'] >= q['ref_under_median_db'] - 3:
                q['verdict'] = 'the take has it too (written)'
            else:
                q['verdict'] = 'hole: the take is louder here'
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mix')
    ap.add_argument('--voice')
    ap.add_argument('--bars')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--bpm', type=float)
    ap.add_argument('--beats-per-bar', type=int, default=4)
    ap.add_argument('--downbeat', type=float, default=0.)
    ap.add_argument('--map')
    ap.add_argument('--ref', help="the take's own mix, to tell a written break from a hole")
    ap.add_argument('--ref-offset', type=float, default=0., help='ref seconds = mix seconds - this')
    args = ap.parse_args()
    BAR, PHASE = grid_from_args(args)
    mix = load(args.mix)
    voice = load(args.voice) if args.voice else None
    step = BAR / 2 if args.half else BAR
    n_bars = int((len(mix) / SR - PHASE) / BAR)
    b0, b1 = (float(v) for v in args.bars.split('-')) if args.bars else (0, n_bars)
    print('peak (stereo)', round(float(np.abs(mix).max()), 3), '| seconds', round(len(mix) / SR, 2))
    print('bar     time   mix dB' + ('  voice dB' if voice is not None else ''))
    rows = levels(mix, BAR, PHASE, step, b0, b1)
    for b, t, lv in rows:
        print(f'{b:>6.1f} {t:7.2f} {lv:>7}' + (f' {db(seg(voice, t, t + step)):>8}' if voice is not None else ''))
    q = quiet(rows, step, load(args.ref) if args.ref else None, args.ref_offset)
    print('quiet (under 40% of the median):', [(x['bar'], x['s']) for x in q])
    for x in q:
        if 'verdict' in x:
            print(f"  bar {x['bar']} at {x['s']} s: {x['under_median_db']} dB under the median; the take there "
                  f"{x.get('ref_under_median_db')} dB under its own: {x['verdict']}")
    jump = np.abs(np.diff(mix.mean(axis=1)))
    print('largest sample jumps at', sorted(round(float(k) / SR, 2) for k in np.argsort(jump)[-5:]), 'size', round(float(jump.max()), 3))


if __name__ == '__main__':
    main()
