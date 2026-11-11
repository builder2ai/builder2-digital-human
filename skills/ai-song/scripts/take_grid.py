"""Measure takes before choosing or joining them: key, an accurate beat period and bar grid, where the singing starts and
ends, how many bars each section's lines took, and the peak.

YuE2 does not follow the key or tempo written in the style text (in one test, 16 takes landed in 8 keys at 98-118 BPM), so
measure every take and filter on key and tempo first; separate and score only the takes that pass.
- beats: tracked on the backing stem if there is one (better), else on the mix; the period is a straight-line fit through
  the tracked beats over the sung span, so the bar grid does not drift (a tempo read off a tempogram is a coarse bin and drifts
  about 0.2 s over 14 bars). Half and double time are folded towards --bpm-hint.
- other pulse errors: on a 3+3+2 groove the tracker can lock onto the 3-sixteenth pulse, 4:3 of the real tempo (2 of 6 takes
  in one test: 137 and 141 BPM against 103 and 105). So the 4:3, 3:4, 3:2 and 2:3 relatives are tried too, and the one whose
  beat, half-bar and bar lags best explain the onsets wins (with a mild pull towards the hint). With stems, the tempo is
  measured on the mix as well (stored as grid_mix); a disagreement is flagged (tempo_agree false): tap along or ask the
  person; cut_take.py and harmony_groove.py take --grid-from mix to use the mix's grid instead.
- downbeat: the beat phase (of 4) with the most low-band (< 150 Hz) onset energy over the sung span; a guess, check it.
- key: Krumhansl fit on the chroma of the backing (or mix); the runner-up is printed, a gap under 0.03 is marked '?'.
- where the words are: with --sections (take_sections.py's output) the sung span runs from the first lyric line to the last,
  and each section gets its own bars and bars per line (the slope of line starts). Without it, the span comes from the
  vocal stem's energy, which also counts hums, vocalise lead-ins (up to 20 s), ad-libs after the chorus and a whole second
  pass of the song: bars per line from --lines then read 1.8-3.1 where the real figure was 0.9-1.3. Use it only as rough.
- on the mix alone (before separating) it can check key, tempo (see the 4:3 note), peak and whether the take reached
  max_duration (hit_cap). It cannot see where the singing starts or ends, so it cannot tell a cut-off take from one that
  kept humming or sang the song again; hit_cap only says "look at this one after separating".
Pass/fail is on key and tempo only.
numpy and ffmpeg only.
usage: take_grid.py take.flac [...] [--stems DIR] [--sections sections.json] [--bpm-hint 108] [--key "G minor" [--relative-ok]]
                    [--bpm 108 --tol 3] [--max-duration 40] [--lines 8] [--out takes.json]
  stems are found as DIR/<take name>_instrumental.flac and _vocals.flac (separate.py's names); without them the mix is used.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SR, NFFT, HOP = 22050, 1024, 256
FR = SR / HOP
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def load_mono(path, sr=SR):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def peak_of(path):
    """Sample peak and the number of samples at full scale, read on the stereo file at 48 kHz."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '2', '-ar', '48000', '-f', 'f32le', '-'],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    x = np.abs(np.frombuffer(raw, np.float32))
    return float(x.max()) if len(x) else 0., int((x >= .999).sum())


def stft_mag(x, n_fft=NFFT, hop=HOP):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))


def frame_time(k):
    return (np.asarray(k) * HOP + NFFT / 2) / SR


def onset_env(x, fmin=0., fmax=None):
    S = stft_mag(x)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (f >= fmin) & (f < (fmax or SR / 2))
    L = np.log1p(100 * S[:, band])
    d = np.r_[0., np.maximum(0, np.diff(L, axis=0)).mean(axis=1)]
    w = int(FR * .5) | 1
    d = np.maximum(0, d - np.convolve(d, np.ones(w) / w, 'same'))
    return d / (d.max() + 1e-12)


def autocorr(o):
    o = o - o.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(o))))
    F = np.fft.rfft(o, n)
    ac = np.fft.irfft(F * np.conj(F), n)[:len(o)]
    return ac / (ac[0] + 1e-12)


def beat_period(o, hint=None):
    """Beat period in frames: the autocorrelation peak, weighted towards the hint (default 110 BPM)."""
    ac = autocorr(o)
    lo, hi = int(FR * 60 / 200), int(FR * 60 / 55)
    lags = np.arange(lo, min(hi, len(ac) - 1))
    bpm = 60 * FR / lags
    weight = np.exp(-.5 * (np.log2(bpm / (hint or 110)) / .9) ** 2)
    k = lags[np.argmax(ac[lags] * weight)]
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    return k + .5 * (a - c) / (a - 2 * b + c + 1e-12)


def track(o, period, alpha=100.):
    """Dynamic-programming beat tracker (Ellis 2007): beats on strong onsets, gaps close to the period."""
    n = len(o)
    score, back = o.copy(), -np.ones(n, int)
    for t in range(n):
        hi = int(t - period / 2)
        if hi <= 0:
            continue
        prev = np.arange(max(0, int(t - 2 * period)), hi)
        cand = score[prev] - alpha * np.log((t - prev) / period) ** 2
        j = int(np.argmax(cand))
        score[t], back[t] = o[t] + cand[j], prev[j]
    t = int(np.argmax(score[max(0, n - int(period)):])) + max(0, n - int(period))
    out = []
    while t >= 0:
        out.append(t)
        t = back[t]
    return frame_time(out[::-1])


def fit_grid(beats, a=None, z=None):
    """Beat period and phase from a straight-line fit through the tracked beats (over [a, z] if given)."""
    bb = beats if a is None else beats[(beats >= a - .5) & (beats <= z + .5)]
    if len(bb) < 8:
        bb = beats
    n = np.round((bb - bb[0]) / np.median(np.diff(bb)))
    p, ic = np.polyfit(n, bb, 1)
    return float(p), float(ic), float(np.std(bb - (ic + p * n))), len(bb)


def fold(p, hint):
    if not hint:
        return p
    while 60 / p < hint / 1.41:
        p /= 2
    while 60 / p > hint * 1.41:
        p *= 2
    return p


RATIOS = {1.: '1:1', 4 / 3: '4:3', 3 / 4: '3:4', 3 / 2: '3:2', 2 / 3: '2:3'}


def pulse_score(ac, q):
    """How well a beat period of q frames explains the onsets: autocorrelation at the beat, half-bar and bar lags."""
    lags = np.arange(len(ac))
    return float(np.mean([np.interp(k * q, lags, ac) for k in (1, 2, 4)]))


def choose_pulse(o, p, hint, a=None, z=None):
    """The beat period (seconds) among p and its 4:3, 3:4, 3:2, 2:3 relatives (each folded 2:1 towards the hint) that best
    explains the onset autocorrelation over [a, z]. Returns (period, ratio to p, {ratio name: score})."""
    seg = o[int(a * FR):int(z * FR) + 1] if a is not None and z - a > 8 else o
    ac = autocorr(seg)
    best, scores = None, {}
    for r, name in RATIOS.items():
        q = fold(p * r, hint)
        if not 55 <= 60 / q <= 200 or 4 * q * FR >= len(ac) - 1:
            continue
        s = pulse_score(ac, q * FR)
        w = np.exp(-.5 * (np.log2(60 / q / hint) / .5) ** 2) if hint else 1.
        scores[name] = round(s, 3)
        if best is None or s * w > best[0]:
            best = (s * w, q, r)
    return (best[1], best[2], scores) if best else (p, 1., scores)


def grid_on(x, hint, a=None, z=None):
    """Beat grid of one signal: (period s, phase s, fit sd s, beats fitted, chosen ratio, pulse scores)."""
    o = onset_env(x)
    beats = track(o, beat_period(o, hint))
    p, ic, sd, nb = fit_grid(beats, a, z)
    p = fold(p, hint)
    q, r, scores = choose_pulse(o, p, hint, a, z)
    if abs(r - 1) > .01:                       # the tracker followed another pulse: track again at the chosen one
        beats = track(o, q * FR)
        p, ic, sd, nb = fit_grid(beats, a, z)
        p = fold(p, hint)
    return p, ic, sd, nb, r, scores


def simple_ratio(r):
    for name, v in (('1:1', 1), ('4:3', 4 / 3), ('3:4', 3 / 4), ('3:2', 1.5), ('2:3', 2 / 3), ('2:1', 2), ('1:2', .5)):
        if abs(r / v - 1) < .03:
            return name
    return f'{r:.3f}'


def downbeat(o_low, p, ic, a, z, beats_per_bar=4):
    k0, k1 = int(np.ceil((a - ic) / p)), int(np.floor((z - ic) / p))
    ks = np.arange(k0, max(k0 + 1, k1 + 1))
    s = np.interp(ic + p * ks, frame_time(np.arange(len(o_low))), o_low)
    scores = [float(s[(ks % beats_per_bar) == ph].mean()) if np.any((ks % beats_per_bar) == ph) else 0. for ph in range(beats_per_bar)]
    best = int(np.argmax(scores))
    k = ks[(ks % beats_per_bar) == best][0]
    srt = sorted(scores)
    return float(ic + p * k), round(srt[-1] / (srt[-2] + 1e-9), 2)


def key_fits(x):
    S = stft_mag(x, 8192, 4096)
    f = np.fft.rfftfreq(8192, 1 / SR)
    band = (f >= 55) & (f < 4200)
    pc = np.round(69 + 12 * np.log2(f[band] / 440)).astype(int) % 12
    C = np.stack([S[:, band][:, pc == k].sum(axis=1) for k in range(12)], axis=1)
    C = (C / (C.max(axis=1, keepdims=True) + 1e-12))[C.sum(axis=1) > 1e-3 * C.sum(axis=1).max()].mean(axis=0)
    out = [(float(np.corrcoef(C, np.roll(prof, k))[0, 1]), f'{NAMES[k]} {m}') for k in range(12) for m, prof in (('major', MAJOR), ('minor', MINOR))]
    return sorted(out, reverse=True)


def frame_rms(v, hop):
    """RMS per block of `hop` samples."""
    n = len(v) // hop
    return np.sqrt((v[:n * hop].reshape(n, hop) ** 2).mean(axis=1))


def voice_runs(v, gap=.6, min_len=.3):
    """Runs of singing in a vocal stem: (start, end) seconds, split at quiet gaps of at least `gap` seconds."""
    hop = int(.02 * SR)
    e = frame_rms(v, hop)
    on = e > .1 * np.percentile(e, 95)
    runs, start, quiet = [], None, 0
    for i, x in enumerate(on):
        if x:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * .02 >= gap:
                runs.append((start * .02, (i - quiet + 1) * .02))
                start, quiet = None, 0
    if start is not None:
        runs.append((start * .02, len(on) * .02))
    return [(a, b) for a, b in runs if b - a >= min_len]


def signature(key):
    tonic, mode = key.split()
    k = NAMES.index(tonic)
    return k if mode == 'major' else (k + 3) % 12


def measure(take, vocals=None, backing=None, hint=None, max_duration=None, lines=None, sec=None):
    """sec: this take's entry from take_sections.py (sung_from, sung_to, sections, cut_off ...), or None."""
    x = load_mono(backing or take)
    dur = len(x) / SR
    o_low = onset_env(x, fmax=150)
    va = vz = None
    if vocals:
        runs = voice_runs(load_mono(vocals))
        if runs:
            va, vz = runs[0][0], runs[-1][1]
    if sec and sec.get('sung_from') is not None:
        a, z = float(sec['sung_from']), float(sec['sung_to'])
    else:
        a, z = va, vz
    p, ic, sd, nb, r, scores = grid_on(x, hint, a, z)
    bar = 4 * p
    db_t, ratio = downbeat(o_low, p, ic, a if a is not None else 0., z if z is not None else dur)
    fits = key_fits(x)
    pk, clipped = peak_of(take)
    row = {'take': str(take), 'id': Path(take).stem, 'seconds': round(dur, 2), 'key': fits[0][1], 'key_fit': round(fits[0][0], 2),
           'key_next': fits[1][1], 'key_gap': round(fits[0][0] - fits[1][0], 3), 'bpm': round(60 / p, 2), 'beat_s': round(p, 5),
           'bar_s': round(bar, 4), 'beat_fit_sd_ms': round(sd * 1000), 'beats_fitted': nb, 'downbeat': round(db_t, 3),
           'downbeat_ratio': ratio, 'pulse': RATIOS.get(r, f'{r:.3f}'), 'pulse_scores': scores,
           'voice_from': round(va, 2) if va is not None else None, 'voice_to': round(vz, 2) if vz is not None else None,
           'peak': round(pk, 3), 'clipped_samples': clipped, 'measured_on': 'backing' if backing else 'mix',
           'hit_cap': bool(max_duration and dur >= max_duration - .3) if max_duration else None}
    notes = []
    if abs(r - 1) > .01:
        notes.append(f"the tracker followed a {RATIOS.get(r)} pulse; re-tracked at {60 / p:.1f} BPM")
    if backing:                                  # the same measurement on the mix: the two must agree
        xm = load_mono(take)
        pm, icm, sdm = grid_on(xm, hint, a, z)[:3]
        dbm = downbeat(onset_env(xm, fmax=150), pm, icm, a if a is not None else 0., z if z is not None else dur)[0]
        row['bpm_mix'] = round(60 / pm, 2)
        row['grid_mix'] = {'bpm': round(60 / pm, 2), 'beat_s': round(pm, 5), 'bar_s': round(4 * pm, 4), 'downbeat': round(dbm, 3),
                           'beat_fit_sd_ms': round(sdm * 1000)}
        row['tempo_agree'] = bool(abs(pm / p - 1) <= .02)
        if not row['tempo_agree']:
            notes.append(f"tempo disagrees: backing {60 / p:.1f}, mix {60 / pm:.1f} BPM ({simple_ratio(p / pm)}): one followed another "
                         f"pulse; tap along or ask the person, then use --grid-from mix in cut_take.py / harmony_groove.py if the mix is right")
    if sec and sec.get('sung_from') is not None:
        row['sung_from'], row['sung_to'] = round(a, 2), round(z, 2)
        row['sung_bars'] = round((z - a) / bar, 1)
        row['sections'] = []
        for s in sec.get('sections', []):
            if s.get('start') is None:
                row['sections'].append({'name': s['name'], 'lines': s['lines'], 'found': False})
                continue
            per = s.get('line_period_s')
            row['sections'].append({'name': s['name'], 'lines': s['lines'], 'start': s['start'], 'end': s['end'],
                                    'bars': round((s['end'] - s['start']) / bar, 1),
                                    'bars_per_line': round(per / bar, 2) if per else None})
        if sec.get('cut_off'):
            notes.append('cut off: ' + sec['cut_off'])
        if (sec.get('after') or {}).get('seconds', 0) > 2:
            af = sec['after']
            notes.append(f"voice for {af['seconds']:.0f} s after the last lyric line" + (f" (sings the lyric again from {af['repeat_from']:.1f} s)" if af.get('repeat_from') else '')
                         + ': cut it off with cut_take.py')
    elif a is not None:
        row['sung_bars'] = round((z - a) / bar, 1)
        if lines:
            row['bars_per_line'] = round((z - a) / bar / lines, 2)
            notes.append('bars_per_line from voice energy (counts hums, lead-ins, ad-libs, a second pass): rough; use --sections')
        if z > dur - .5:
            notes.append('voice runs into the end: the last line cut off, or humming / singing on; take_sections.py tells which')
    if row['hit_cap']:
        notes.append('reached max_duration: maybe cut off, maybe sang on after the lyric; check after separating')
    if row['key_gap'] < .03:
        notes.append(f"key near tie with {row['key_next']}")
    if pk >= .999 or clipped:
        notes.append(f'peak {pk:.3f}, {clipped} samples at full scale: lower the level before joining')
    if sd > .02:
        notes.append(f'beat fit spread {sd * 1000:.0f} ms: tempo unsteady or tracked badly')
    row['notes'] = notes
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('takes', nargs='+')
    ap.add_argument('--stems')
    ap.add_argument('--bpm-hint', type=float)
    ap.add_argument('--key')
    ap.add_argument('--relative-ok', action='store_true')
    ap.add_argument('--bpm', type=float)
    ap.add_argument('--tol', type=float, default=3., help='tempo tolerance, percent')
    ap.add_argument('--max-duration', type=float)
    ap.add_argument('--lines', type=int)
    ap.add_argument('--sections', help="take_sections.py's output: per-section timing from the lyric lines")
    ap.add_argument('--out')
    a = ap.parse_args()
    secs = {r['id']: r for r in json.load(open(a.sections, encoding='utf-8'))} if a.sections else {}
    rows = []
    for t in a.takes:
        name = Path(t).stem
        st = Path(a.stems) if a.stems else None
        bk = st / f'{name}_instrumental.flac' if st and (st / f'{name}_instrumental.flac').exists() else None
        vo = st / f'{name}_vocals.flac' if st and (st / f'{name}_vocals.flac').exists() else None
        r = measure(t, vo, bk, a.bpm_hint or a.bpm, a.max_duration, a.lines, secs.get(name))
        ok = True
        if a.key:
            same = r['key'] == a.key or (a.relative_ok and signature(r['key']) == signature(a.key))
            ok &= same and r['key_gap'] >= .03
        if a.bpm:
            ok &= abs(r['bpm'] - a.bpm) / a.bpm * 100 <= a.tol
        r['pass'] = bool(ok)
        rows.append(r)
        if r.get('sections'):
            bpl = ' '.join(f"{s['name']} {s['start']:.1f}-{s['end']:.1f}s {s['bars']} bars {s['bars_per_line']}/line" if s.get('start') is not None
                           else f"{s['name']} not found" for s in r['sections'])
        else:
            bpl = f"{r.get('bars_per_line')} bars/line (rough)" if r.get('bars_per_line') else ''
        mix = '' if 'bpm_mix' not in r else f" mix {r['bpm_mix']:.2f}{'' if r['tempo_agree'] else ' ?'}"
        print(f"{'PASS' if ok else '    '} {r['id']:<28} {r['key']:<9} ({r['key_fit']:.2f}, next {r['key_next']} +{r['key_gap']:.3f})  "
              f"{r['bpm']:7.2f} BPM{mix} sd {r['beat_fit_sd_ms']:>3} ms  downbeat {r['downbeat']:7.3f}  voice {r['voice_from']}-{r['voice_to']}  {bpl}  "
              f"peak {r['peak']}  {'; '.join(r['notes'])}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
