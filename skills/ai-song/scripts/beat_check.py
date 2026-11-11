"""Is new music (a joined take, an ACE-Step intro or ending) on the same beat as the music next to it?

Three readings, all against the grid of the context (bar lines at --join + k bars):
- eighth-note phase: where the onsets sit against the eighth-note grid on each side of the join, two ways: the circular mean
  of onset strength at the eighth period, and the offset that puts the most onset strength within 30 ms of the eighths.
  The jump is new minus context (positive: the new part is late). Measured at the eighth, not the beat: on a 3+3+2 groove
  the accent on the and-of-2 pulls a beat-period reading by up to half a beat. In one test the beat-period readings passed
  an intro at 0 ms that was displaced by 3 eighths and failed the good one at 270 ms; at the eighth they read -8 and 0 ms.
- accent profile: the onset strength at the 8 eighths of each bar (the mean over the bars of each side, each side read on
  its own eighth phase so a small jump does not move it), and the rotation of the new part's profile that best matches
  the context's. A best shift that is not 0 (and clearly better than 0) means
  the new part's accents sit that many eighths off: a displaced downbeat, which an eighth-phase reading cannot see (the
  eighths still line up). Per-bar shifts are listed too. Positive: the new part's accents come late.
- ending: a new part that is one held chord has no pulse to measure (the phase readings then mean nothing; in one test they
  read +52 to +139 ms on endings whose chord landed within 20 ms of the bar line). So for an ending the first strong onset
  of the first new bar (the first at least half as strong as the strongest there) is timed against the bar line. Phase and
  accents count only when the new part keeps a groove (enough onsets).
Verdict: ok, or what failed: jump (eighth phase over --tol-ms), displaced (accent shift), onset (ending's first onset over
--tol-ms from the bar line), weak (no clear pulse on one side: listen).
Library: check(x, join, bar_s, new='after'|'before', ending=False, ...) for join_takes.py, assemble.py and the examples.
usage: beat_check.py <file> [...] --bar-s 2.3065 --join 9.226 [--new before|after] [--ending] [--before-bars 4]
                     [--after-bars 4] [--tol-ms 30] [--out beat.json]
       beat_check.py --spec ace/spec.json --bar-s 2.3065 [id ...]
  --join: a bar line on the file's clock where the context meets the new part. --new before: the new part comes first (an
    intro written before the context); after (default): the context comes first (a join, an ending).
  --spec: ACE-Step repaint candidates (ace_step_run.py's raw/<id>.wav): the join is -repainting_start for an intro (new
    bars before the context) and repainting_start otherwise; a repaint that runs to the end counts as an ending. The
    context audio must start on a bar line (make it that way).
numpy and ffmpeg only.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import take_grid as tg  # noqa: E402
from songmap import SR, load, seg  # noqa: E402

TOL_GRID = .03       # an onset within 30 ms of an eighth counts as on it (best-offset reading)
WEAK = .08           # circular strength under this: no clear eighth pulse
SHIFT_MARGIN = .15   # the best rotation must beat rotation 0 by this much correlation to call it displaced
GROOVE_PEAKS = 6     # a new part with fewer onset peaks than this (over a quarter of the context's strongest) is a held chord


def onsets(x, t0, t1):
    """Onset strength of x (songmap.load: 48 kHz, (samples, channels)) from t0 to t1 seconds, and each frame's time."""
    s = seg(x, max(0., t0), t1).mean(axis=1)[::2].astype(np.float64)        # 24 kHz is close enough for onsets
    o = tg.onset_env(np.interp(np.arange(0, len(s), 24000 / tg.SR), np.arange(len(s)), s))
    return o, max(0., t0) + tg.frame_time(np.arange(len(o)))


def wrap(v, period):
    return (v + period / 2) % period - period / 2


def circ_offset(o, t, anchor, period):
    """Circular mean of onset strength at `period` against a grid through `anchor`: (offset s, strength 0-1)."""
    z = np.sum(o * np.exp(2j * np.pi * (t - anchor) / period))
    return float(np.angle(z)) / (2 * np.pi) * period, float(np.abs(z) / (o.sum() + 1e-12))


def best_offset(o, t, anchor, period, tol=TOL_GRID, step=.005):
    """The grid offset (s) that puts the most onset strength within `tol` of a grid of `period` through `anchor`."""
    offs = np.arange(-period / 2, period / 2, step)
    sc = [o[np.abs(wrap(t - anchor - p, period)) < tol].sum() for p in offs]
    return float(offs[int(np.argmax(sc))])


def accent_profile(o, t, bar_lines, bar_s, n=8, offset=0.):
    """Per bar, the strongest onset within 35 ms of each of n even positions (moved by `offset` s: the side's own eighth
    phase, so a small phase jump does not change the profile); returns (mean profile scaled to max 1, per bar)."""
    rows = []
    for b0 in bar_lines:
        row = []
        for k in range(n):
            near = np.abs(t - (b0 + offset + bar_s * k / n)) <= .035
            row.append(float(o[near].max()) if near.any() else 0.)
        rows.append(row)
    rows = np.array(rows) if rows else np.zeros((0, n))
    mean = rows.mean(axis=0) if len(rows) else np.zeros(n)
    return mean / (mean.max() + 1e-12), rows


def best_shift(p_new, p_ctx):
    """The rotation r (eighths, signed) with np.roll(p_new, -r) closest to p_ctx: (r, corr at 0, corr at r)."""
    p_new, p_ctx = np.asarray(p_new, float), np.asarray(p_ctx, float)
    n = len(p_new)
    if p_new.std() < 1e-9 or p_ctx.std() < 1e-9:
        return 0, 0., 0.
    c = [float(np.corrcoef(np.roll(p_new, -r), p_ctx)[0, 1]) for r in range(n)]
    r = int(np.argmax(c))
    return (r if r <= n // 2 else r - n), c[0], c[r]


def peaks(o, t, a, b, floor):
    """Times of local maxima of o in [a, b) over `floor`."""
    k = np.where((o[1:-1] >= o[:-2]) & (o[1:-1] > o[2:]) & (o[1:-1] > floor))[0] + 1
    return t[k][(t[k] >= a) & (t[k] < b)], o[k][(t[k] >= a) & (t[k] < b)]


def check(x, join, bar_s, new='after', ending=False, before_bars=4., after_bars=4., beats_per_bar=4, tol_ms=30.):
    """Beat check of the new part against its context at bar line `join` (seconds on x's clock). See the module docstring."""
    e = bar_s / (2 * beats_per_bar)
    n8 = 2 * beats_per_bar
    dur = len(x) / SR
    a, b = max(0., join - before_bars * bar_s), min(dur, join + after_bars * bar_s)
    o, t = onsets(x, a - .5, b + .5)
    sides = {'before': (a, join), 'after': (join, b)}
    ctx_side, new_side = ('before', 'after') if new == 'after' else ('after', 'before')
    out = {'join_s': round(join, 3), 'bar_s': bar_s, 'eighth_ms': round(e * 1000, 1), 'new': new, 'ending': ending}
    read = {}
    for side, (s0, s1) in sides.items():
        m = (t >= s0) & (t < s1)
        circ, strength = circ_offset(o[m], t[m], join, e)
        best = best_offset(o[m], t[m], join, e)
        nb = int(np.floor((s1 - s0) / bar_s + 1e-6))
        bars = [join + k * bar_s for k in range(nb)] if side == 'after' else [join - (k + 1) * bar_s for k in range(nb)][::-1]
        prof, rows = accent_profile(o, t, bars, bar_s, n8, circ if strength >= WEAK else 0.)
        read[side] = {'span_s': [round(s0, 2), round(s1, 2)], 'circ_8th_ms': round(circ * 1000), 'best_8th_ms': round(best * 1000),
                      'strength': round(strength, 2), 'accents_8ths': [round(float(v), 2) for v in prof], 'bars': len(bars), '_rows': rows}
    c, nw = read[ctx_side], read[new_side]
    jump_c = wrap((nw['circ_8th_ms'] - c['circ_8th_ms']) / 1000, e) * 1000
    jump_b = wrap((nw['best_8th_ms'] - c['best_8th_ms']) / 1000, e) * 1000
    shift, corr0, corr_best = best_shift(nw['accents_8ths'], c['accents_8ths'])
    per_bar = [best_shift(r, c['accents_8ths'])[0] if np.std(r) > 1e-9 else None for r in nw['_rows']]
    for v in read.values():
        v.pop('_rows')
    out.update({'context': read[ctx_side], 'new_part': read[new_side], 'jump_8th_ms': [round(jump_c), round(jump_b)],
                'accent_shift_8ths': shift, 'accent_corr_at_0': round(corr0, 2), 'accent_corr_best': round(corr_best, 2),
                'accent_shift_per_bar': per_bar})
    # how busy the new part is: onset peaks over a quarter of the context's strongest
    cm = (t >= sides[ctx_side][0]) & (t < sides[ctx_side][1])
    ref = float(o[cm].max()) if cm.any() else 1.
    pt, _ = peaks(o, t, *sides[new_side], .25 * ref)
    out['new_onset_peaks'] = int(len(pt))
    groove = len(pt) >= GROOVE_PEAKS
    out['new_keeps_groove'] = bool(groove)
    fails, notes = [], []
    if ending:
        # the first onset in the first new bar that is at least half as strong as the strongest one there (an ending chord
        # is often much softer than the groove before it: 0.07-0.26 of the context's strongest onset in one test)
        lo = join - min(.1, .25 * e)
        c2 = (t >= join - 2 * bar_s) & (t < join - .05)
        ref2 = float(o[c2].max()) if c2.any() else ref
        pt2, po2 = peaks(o, t, lo, join + bar_s, 0.)
        if len(pt2):
            k = int(np.argmax(po2 >= .5 * po2.max()))
            first = float(pt2[k] - join) * 1000
            out['first_strong_onset_ms'] = round(first)
            out['first_strong_onset_vs_context'] = round(float(po2[k]) / (ref2 + 1e-12), 2)
            if abs(first) > tol_ms:
                fails.append('onset')
        else:
            out['first_strong_onset_ms'] = None
            notes.append('no onset in the first new bar')
            fails.append('onset')
        if not groove:
            notes.append(f'the new part is a held chord ({len(pt)} onset peaks): eighth phase and accents not counted, the first onset decides')
    if not ending or groove:
        if min(c['strength'], nw['strength']) < WEAK:
            fails.append('weak')
            notes.append(f"no clear eighth pulse (strength {c['strength']} context, {nw['strength']} new): listen to the join")
        elif max(abs(jump_c), abs(jump_b)) > tol_ms:
            fails.append('jump')
            notes.append(f'eighth-note phase jumps {jump_c:+.0f} / {jump_b:+.0f} ms')
        if shift != 0 and corr_best - corr0 >= SHIFT_MARGIN:
            fails.append('displaced')
            notes.append(f"accents line up only shifted by {shift:+d} eighths (r {corr0:+.2f} at 0, {corr_best:+.2f} shifted): "
                         f"a displaced downbeat, or a different pattern; per bar {per_bar}")
    out['verdict'] = 'ok' if not fails else ' '.join(dict.fromkeys(fails))
    out['notes'] = notes
    return out


def line(name, r):
    s = (f"{name}: {r['verdict']:<10} jump {r['jump_8th_ms'][0]:+4d} / {r['jump_8th_ms'][1]:+4d} ms (8th {r['eighth_ms']} ms; strength "
         f"{r['context']['strength']:.2f} ctx {r['new_part']['strength']:.2f} new) | accent shift {r['accent_shift_8ths']:+d} "
         f"(r {r['accent_corr_at_0']:+.2f} at 0, {r['accent_corr_best']:+.2f} best; per bar {r['accent_shift_per_bar']})")
    if r['ending']:
        s += f" | first onset {r['first_strong_onset_ms']} ms, {r['new_onset_peaks']} peaks"
    return s + (f" | {'; '.join(r['notes'])}" if r['notes'] else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='audio files, or candidate ids with --spec')
    ap.add_argument('--spec')
    ap.add_argument('--bar-s', type=float, required=True)
    ap.add_argument('--beats-per-bar', type=int, default=4)
    ap.add_argument('--join', type=float)
    ap.add_argument('--new', choices=('before', 'after'), default='after')
    ap.add_argument('--ending', action='store_true')
    ap.add_argument('--before-bars', type=float, default=4.)
    ap.add_argument('--after-bars', type=float, default=4.)
    ap.add_argument('--tol-ms', type=float, default=30.)
    ap.add_argument('--out')
    a = ap.parse_args()
    jobs = []
    if a.spec:
        sp = Path(a.spec)
        for c in json.loads(sp.read_text(encoding='utf-8'))['candidates']:
            if c.get('task') != 'repaint' or (a.files and c['id'] not in a.files):
                continue
            path = sp.parent / 'raw' / f"{c['id']}.wav"
            if not path.exists():
                continue
            rs, re_ = float(c['repainting_start']), float(c['repainting_end'])
            dur = float(c.get('duration', 0) or 0)
            if rs < 0:
                jobs.append((c['id'], path, -rs, 'before', False))
            else:
                jobs.append((c['id'], path, rs, 'after', re_ < 0 or (dur and re_ >= dur - .1)))
    else:
        if a.join is None:
            raise SystemExit('give --join (a bar line where the context meets the new part), or --spec')
        jobs = [(Path(f).stem, f, a.join, a.new, a.ending) for f in a.files]
    rows = []
    for name, path, join, new, ending in jobs:
        x = load(path)
        # an intro's new bars are all before the join; an ending's context is before it
        bb = min(a.before_bars, join / a.bar_s) if new == 'before' else a.before_bars
        r = check(x, join, a.bar_s, new, ending, bb, a.after_bars, a.beats_per_bar, a.tol_ms)
        r['id'] = name
        rows.append(r)
        print(line(name, r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
