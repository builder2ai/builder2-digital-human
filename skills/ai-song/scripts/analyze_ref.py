"""扒曲: measure the form of a song you like, so its structure and style can be written into your own lyrics and style text.

Only the form is learnt. The reference's audio, notes and words never go into generation, and its stems, transcripts and
this card stay in a private working folder.

Inputs (any audio format; everything is decoded with ffmpeg):
  song           the mix
  --vocals       its vocal stem          (separate.py --model vocals)
  --parts P      Demucs layers P_voice/_drums/_bass/_other.wav (separate.py --model parts): instruments and meter
  --lines J      windowed Whisper segments of the vocal stem (transcribe_ref.py): sung lines and spoken parts
Measures, each marked as settled by measurement or as 待人听核 (a person must listen):
  - tempo: the strongest tempogram peaks, with 2:1 and 3:2 pairs flagged; a meter check (4/4 against 12/8) on the drums;
  - a downbeat guess from low-band onsets on the beats where the band plays (the first beat is NOT a bar line);
  - key: Krumhansl fits on the mix (and on the harmonic layer); near ties (gap < 0.03) go to 待人听核;
  - sections: cuts on the beat grid from harmony and timbre, with per-section layer levels and rough instrument hints;
  - lines: each Whisper segment classed sung / not sung (spoken or rap) from its held-note share and characters per second;
    line length (seconds and bars), share sung, share not sung and where it sits, pitch range on sung lines only with
    octave errors folded, where the highest note sits, how many lines end lower. Any line over 8 s is warned about.
Writes <out>.json and prints the card.
usage: analyze_ref.py <song> [--vocals V] [--parts P] [--lines lines.json] [--sections N] [--out card]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import librosa  # noqa: E402

SR = 22050
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
# sung / not sung: a hint, not a verdict. Characters per second carry most weight (sung lines ran about 2-3 a second, speech up
# to about 7); the held-note share only overrules it when very low, because soft, gliding singing also holds few notes. Rap
# usually lands in 'not sung'. Every line's numbers go into the card, and a person decides.
SUNG_HELD, SUNG_CPS = .45, 3.5
NOT_SUNG_HELD, NOT_SUNG_CPS = .3, 4.5
# Tried and dropped: 'no band under the line -> spoken'. On one test it marked a sparse verse spoken and still missed slow
# skit lines. The band level is kept per line (band_db) for a person to read. Slow speech is the weak spot of this class.


def load_mono(path, sr=SR):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.float32).copy()


def db(x):
    return round(float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)), 1) if len(x) else -120.


def key_fits(y):
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR).mean(axis=1)
    out = [(float(np.corrcoef(chroma, np.roll(p, k))[0, 1]), f'{NAMES[k]} {m}') for k in range(12) for m, p in (('major', MAJOR), ('minor', MINOR))]
    return sorted(out, reverse=True)


def tempo_candidates(y, n=4):
    oenv = librosa.onset.onset_strength(y=y, sr=SR)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=SR)
    bpm = librosa.tempo_frequencies(tg.shape[0], sr=SR)
    prof = tg.mean(axis=1)
    peaks = [i for i in range(1, len(prof) - 1) if 40 < bpm[i] < 220 and prof[i] > prof[i - 1] and prof[i] >= prof[i + 1]]
    peaks.sort(key=lambda i: -prof[i])
    out = []
    for i in peaks:
        if all(abs(bpm[i] - b) > 3 for b, _ in out):
            out.append((round(float(bpm[i]), 1), round(float(prof[i] / prof[peaks[0]]), 2)))
        if len(out) == n:
            break
    return out


def ratio_flags(cands):
    flags = []
    for i, (a, _) in enumerate(cands):
        for b, _ in cands[i + 1:]:
            hi, lo = max(a, b), min(a, b)
            r = hi / lo
            if abs(r - 2) < .08:
                flags.append(f'{hi}:{lo} is 2:1 (half/double time)')
            elif abs(r - 1.5) < .06:
                flags.append(f'{hi}:{lo} is 3:2 (4/4 at {hi} or 12/8 at {lo}: see the meter check)')
    return flags


def meter_check(x, eighth):
    """The bar from the onset envelope's autocorrelation at 8 against 12 eighth notes, only where the drums (or the mix) play;
    3 against 4 eighths is reported as the accent pattern (a 3+3+2 groove is still 4/4)."""
    hop = 128
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=hop))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    e = np.log1p(S[(f >= 30) & (f < 11000)].sum(axis=0))
    o = np.maximum(0, np.diff(e, prepend=e[0]))
    fr = SR / hop
    w = int(fr * .5)
    level = 20 * np.log10(np.sqrt(np.convolve(S.sum(axis=0) ** 2 / S.shape[0] ** 2, np.ones(w) / w, 'same')) + 1e-9)
    o = np.where(level > np.percentile(level, 95) - 25, o, 0.)          # leave out the stretches where the band is quiet
    o = o - o[o > 0].mean() if np.any(o > 0) else o
    n = 1 << int(np.ceil(np.log2(2 * len(o))))
    F = np.fft.rfft(o, n)
    ac = np.fft.irfft(F * np.conj(F), n)[:len(o)]
    ac /= ac[0] + 1e-12

    def at(k):
        c, d = int(round(k * eighth * fr)), int(round(.03 * fr))
        return round(float(ac[max(1, c - d):c + d + 1].max()), 2) if c + d < len(ac) else None
    v = {k: at(k) for k in (3, 4, 8, 12)}
    if None in v.values():
        return v, 'unclear (too short)'
    accent = '3+3+2 accents' if v[3] > v[4] else 'straight accents'
    if v[8] > v[12] + .1:
        return v, f'4/4 ({accent})'
    if v[12] > v[8] + .1:
        return v, '12/8 or 6/8 (beat = dotted quarter)'
    return v, 'unclear'


def sections(y, beats, n):
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    frames = librosa.time_to_frames(beats, sr=SR)
    feat = np.vstack([librosa.util.sync(chroma, frames), librosa.util.sync(mfcc, frames) / 50.])
    bounds = librosa.segment.agglomerative(feat, n)
    return sorted(set(float(t) for t in librosa.frames_to_time(librosa.util.fix_frames(frames)[bounds], sr=SR)))


def decay_hint(x):
    """Median level 200 ms after each onset, dB against the onset: plucked/struck notes die, bowed or held ones stay."""
    on = librosa.onset.onset_detect(y=x, sr=SR, units='samples')
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    d = [20 * np.log10(rms[o // 256 + int(.2 * SR / 256)] / rms[o // 256] + 1e-9) for o in on
         if o // 256 + int(.2 * SR / 256) < len(rms) and rms[o // 256] > 1e-5]
    if not d:
        return None, 'too few notes'
    m = float(np.median(d))
    return round(m, 1), 'held (bowed strings, pads, organ-like)' if m > -4 else 'plucked or struck' if m < -5.5 else 'mixed'


def band_share(x, lo=None, hi=None):
    S = np.abs(librosa.stft(x, n_fft=2048)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    k = np.ones_like(f, bool) if lo is None and hi is None else (f < lo) if hi is None else (f > hi)
    return float(S[k].sum() / (S.sum() + 1e-12))


def line_stats(v, lines, pitch=True, layers=None):
    """Per transcribed line: characters per second, the level of the band under it (with layers, for a person to read), and
    (pitch=True) pYIN on that line only: held-note share, median, notes."""
    out = []
    for ln in lines:
        a, b = float(ln['start']), float(ln['end'])
        chars = sum(1 for c in ln['text'] if '一' <= c <= '鿿')
        cps = chars / max(.1, b - a)
        row = {'start': round(a, 2), 'end': round(b, 2), 'seconds': round(b - a, 2), 'chars_per_s': round(cps, 1), 'text': ln['text'], 'held': None}
        if layers:
            row['band_db'] = max(db(layers['drums'][int(a * SR):int(b * SR)]), db(layers['other'][int(a * SR):int(b * SR)]))
        m = np.array([])
        if pitch:
            x = v[int(max(0, a) * SR):int(b * SR)]
            if len(x) > 2048:
                f0, voiced, _ = librosa.pyin(x, fmin=65, fmax=1000, sr=SR, frame_length=2048, hop_length=512)
                m = librosa.hz_to_midi(f0[voiced])
                row['voiced'] = round(float(voiced.mean()), 2)
        if len(m) >= 6:
            med = float(np.median(m))
            folded = np.where(np.abs(m - med) > 9, m - 12 * np.sign(m - med), m)   # octave errors of the pitch tracker
            q = max(1, len(folded) // 4)
            row.update(held=round(float(np.mean(np.abs(np.diff(m)) < .4)), 2), median_midi=round(med, 1), _notes=folded,
                       high_pos=round(float(np.argmax(folded) / max(1, len(folded) - 1)), 2),
                       ends_lower=bool(np.median(folded[-q:]) < np.median(folded[:q]) - .5))
        held = row['held']
        if cps <= SUNG_CPS and (held is None or held >= SUNG_HELD):
            row['kind'] = 'sung'
        elif cps >= NOT_SUNG_CPS or (held is not None and held < NOT_SUNG_HELD):
            row['kind'] = 'not sung'
        else:
            row['kind'] = 'unsure'
        out.append(row)
    return out


def spans_of(lines, kinds=('not sung', 'unsure'), gap=3.):
    """Neighbouring lines of the given kinds grouped into parts (no sung line between, gaps under `gap` seconds)."""
    parts = []
    for r in lines:
        if r['kind'] in kinds and parts and parts[-1]['open'] and r['start'] - parts[-1]['end'] < gap:
            parts[-1]['end'] = r['end']
            parts[-1]['lines'] += 1
            parts[-1]['kinds'].add(r['kind'])
        elif r['kind'] in kinds:
            parts.append({'start': r['start'], 'end': r['end'], 'lines': 1, 'kinds': {r['kind']}, 'open': True})
        elif parts:
            parts[-1]['open'] = False
    return parts


def energy_phrases(v, gap=.25):
    hop = int(.02 * SR)
    e = np.array([np.sqrt(np.mean(v[i:i + hop] ** 2)) for i in range(0, len(v) - hop, hop)])
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
    return [(a, b) for a, b in runs if b - a >= .4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('song')
    ap.add_argument('--vocals')
    ap.add_argument('--parts')
    ap.add_argument('--lines')
    ap.add_argument('--sections', type=int)
    ap.add_argument('--no-pitch', action='store_true', help='skip pYIN on the lines (the slow part): no range, highest note or held share')
    ap.add_argument('--out', default='card')
    a = ap.parse_args()
    settled, listen, warn = [], [], []
    clock = [time.time()]
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - clock[0], 1)
        clock[0] = time.time()

    y = load_mono(a.song)
    seconds = len(y) / SR
    layers = {k: load_mono(f'{a.parts}_{k}.wav') for k in ('voice', 'drums', 'bass', 'other')} if a.parts else {}
    pulse = layers.get('drums', y)
    lap('load')

    # tempo and meter
    cands = tempo_candidates(y)
    if layers:
        cands_d = tempo_candidates(layers['drums'])
        flags = ratio_flags(cands + [c for c in cands_d if all(abs(c[0] - b) > 3 for b, _ in cands)])
    else:
        cands_d, flags = [], ratio_flags(cands)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, units='time')
    tempo = float(np.atleast_1d(tempo)[0])
    fast = max([tempo] + [c for c, _ in cands + cands_d if any(abs(c / tempo - r) < .06 for r in (1.5,))])
    meter_ac, meter = meter_check(pulse, 60 / fast / 2)
    beats_per_bar = 4
    if meter.startswith('12/8'):
        listen.append(f'meter looks like 12/8: the felt beat is about {fast * 2 / 3:.1f} per minute; check by ear')
    elif meter == 'unclear':
        listen.append('meter unclear (4/4 assumed below); check by ear')
    else:
        settled.append(f'meter {meter} (autocorrelation at 3/4/8/12 eighths: {meter_ac})')
    if flags:
        warn += flags
    bar = beats_per_bar * 60 / tempo
    settled.append(f'tempo about {tempo:.1f} BPM, a bar is {bar:.3f} s (tempogram peaks: mix {cands}' + (f', drums {cands_d})' if cands_d else ')')
                   + '; this is a coarse estimate for the style card: for timing a take or a join, use take_grid.py (line fit)')
    lap('tempo and meter')

    # key
    fits = key_fits(y)
    key, gap = fits[0][1], fits[0][0] - fits[1][0]
    key_note = f'key {key} (fit {fits[0][0]:.2f}; next {fits[1][1]} {fits[1][0]:.2f})'
    if layers:
        fo = key_fits(layers['other'])
        votes = {}
        for s in range(0, int(seconds) - 19, 20):
            k = key_fits(layers['other'][s * SR:(s + 20) * SR])[0][1]
            votes[k] = votes.get(k, 0) + 1
        key_note += f'; harmonic layer {fo[0][1]} ({fo[0][0]:.2f}), 20 s windows {votes}'
    (listen if gap < .03 else settled).append(key_note + (' -- near tie' if gap < .03 else ''))
    lap('key')

    # downbeat guess: the beat phase with the most low-band onset energy, on beats where the band plays
    low = layers['drums'] + layers['bass'] if layers else y
    oenv = librosa.onset.onset_strength(y=low, sr=SR, fmax=200, n_mels=32)
    st = np.interp(beats, librosa.times_like(oenv, sr=SR), oenv)
    band_on = np.array([db(pulse[int(b * SR):int((b + .25) * SR)]) > -45 for b in beats]) if len(beats) else np.array([], bool)
    use = np.where(band_on)[0]
    downbeat = None
    if len(use) >= 16:
        scores = [float(st[use[(use % 4) == p]].mean()) for p in range(4)]
        p = int(np.argmax(scores))
        first = use[(use % 4) == p][0]
        downbeat = round(float(beats[first]), 3)
        ratio = sorted(scores)[-1] / (sorted(scores)[-2] + 1e-9)
        (settled if ratio > 1.15 else listen).append(f'downbeat guess {downbeat} s (bar phase scores {[round(s, 2) for s in scores]})')
    else:
        listen.append('downbeat: too few beats with the band playing to guess')

    # sections and layers
    n = a.sections or max(3, int(round(seconds / 20)))
    cuts = [0.] + [t for t in sections(y, beats, n) if t > 1] + [seconds]
    secs = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        row = {'start': round(s, 1), 'seconds': round(e - s, 1), 'bars': round((e - s) / bar, 1), 'level_db': db(y[int(s * SR):int(e * SR)])}
        if layers:
            row['layers_db'] = {k: db(v[int(s * SR):int(e * SR)]) for k, v in layers.items()}
            dec, kind = decay_hint(layers['other'][int(s * SR):int(e * SR)])
            row['other_decay_db'], row['other_hint'] = dec, kind
        secs.append(row)
    settled.append('section cuts (usually close to the real boundaries, with some extra cuts and an occasional missed one: merge and check against the line transcript)')
    if layers:
        if max(db(layers['bass'][i:i + SR * 4]) for i in range(0, len(y), SR * 4)) < -55:
            warn.append('the bass layer is silent all song: a phone or low-bitrate recording often loses the bass; ask a person what plays low')
        lo, hi = band_share(layers['drums'], hi=None, lo=150), band_share(layers['drums'], hi=5000)
        if lo < .02 and hi < .02:
            warn.append(f'the drums layer has almost no energy below 150 Hz ({lo:.0%}) or above 5 kHz ({hi:.0%}): kick and cymbals may be lost in the recording')
        listen.append('instrument names: the layers only give roles (held vs plucked, where drums play); a person names the instruments')
    lap('downbeat, sections and layers')

    card = {'file': a.song, 'seconds': round(seconds, 1), 'tempo_bpm': round(tempo, 1), 'tempo_candidates': cands, 'tempo_candidates_drums': cands_d,
            'tempo_flags': flags, 'meter': meter, 'meter_autocorr': meter_ac, 'bar_seconds': round(bar, 3),
            'first_beat_not_a_bar_line': round(float(beats[0]), 3) if len(beats) else None, 'downbeat_guess': downbeat,
            'key': key, 'key_fits_top3': [(round(f, 2), k) for f, k in fits[:3]], 'sections': secs}

    # lines
    if a.vocals:
        v = load_mono(a.vocals)
        listen.append("singer's gender and voice quality")
        if a.lines:
            lines = line_stats(v, json.load(open(a.lines, encoding='utf-8')), pitch=not a.no_pitch, layers=layers or None)
            sung = [r for r in lines if r['kind'] == 'sung']
            other = [r for r in lines if r['kind'] != 'sung']
            for r in lines:
                if r['seconds'] > 8:
                    warn.append(f"line {r['start']}-{r['end']} s is {r['seconds']} s long: probably several lines merged; transcribe that stretch again "
                                f"(transcribe_ref.py --from {max(0, r['start'] - 1):.0f} --to {r['end'] + 1:.0f} --merge-into ...)")
            notes = np.concatenate([r['_notes'] for r in sung if '_notes' in r]) if any('_notes' in r for r in sung) else np.array([])
            lens = np.array([r['seconds'] for r in sung])
            spoken = []
            for p in spans_of(lines):
                pos = p['start'] / seconds
                where = 'opening' if pos < .12 else 'ending' if pos > .85 else 'middle'
                music = (max(db(layers['drums'][int(p['start'] * SR):int(p['end'] * SR)]), db(layers['other'][int(p['start'] * SR):int(p['end'] * SR)])) > -45) if layers else None
                overlap = any(min(p['end'], s['end']) - max(p['start'], s['start']) > .3 for s in sung)
                spoken.append({'start': p['start'], 'end': p['end'], 'lines': p['lines'], 'kinds': sorted(p['kinds']), 'where': where,
                               'over_music': music, 'overlaps_singing': overlap})
            card['lines'] = {'sung': len(sung), 'not_sung_or_unsure': len(other),
                             'line_seconds_median': round(float(np.median(lens)), 2) if len(lens) else None,
                             'line_bars_median': round(float(np.median(lens)) / bar, 2) if len(lens) else None,
                             'line_seconds_spread': round(float(np.percentile(lens, 75) - np.percentile(lens, 25)), 2) if len(lens) else None,
                             'share_sung': round(float(lens.sum() / seconds), 2),
                             'share_not_sung': round(float(sum(r['seconds'] for r in other) / seconds), 2),
                             'range_semitones_sung': round(float(np.percentile(notes, 95) - np.percentile(notes, 5)), 1) if len(notes) else None,
                             'highest_note_position': round(float(np.mean([r['high_pos'] for r in sung if 'high_pos' in r])), 2) if any('high_pos' in r for r in sung) else None,
                             'lines_ending_lower': round(float(np.mean([r['ends_lower'] for r in sung if 'ends_lower' in r])), 2) if any('ends_lower' in r for r in sung) else None,
                             'spoken_parts': spoken,
                             'per_line': [{k: v for k, v in r.items() if k != '_notes'} for r in lines]}
            for s in secs:   # line length per section, in bars
                ls = [r['seconds'] for r in sung if s['start'] <= r['start'] < s['start'] + s['seconds']]
                s['sung_line_bars'] = round(float(np.median(ls)) / bar, 2) if ls else None
            settled.append('sung line lengths (from the transcript; a line split at a window edge halves them, so check against the bars)')
            listen.append('sung / not sung for each line is a hint from characters per second and held notes: confirm by ear, '
                          'above all slow speech and soft, gliding singing')
            if other:
                listen.append('the not-sung parts: speech or rap, and what each is for (a skit, a call, a joke)')
            lap('lines' + (' (no pitch)' if a.no_pitch else ' (pYIN per line)'))
        else:
            ph = energy_phrases(v)
            lens = np.array([b - s for s, b in ph])
            warn.append('no --lines: phrases come from energy only, which merges lines under reverb and counts speech as singing. '
                        'Run transcribe_ref.py on the vocal stem and pass --lines; until then treat these phrase numbers as unreliable.')
            if len(lens) and lens.max() > 8:
                warn.append(f'longest energy phrase is {lens.max():.1f} s: lines are being merged')
            card['phrases_energy_only'] = {'n': len(ph), 'median_s': round(float(np.median(lens)), 2) if len(lens) else None}

    card['settled'], card['listen'], card['warnings'], card['timings_s'] = settled, listen, warn, timings
    with open(a.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=1, default=float)
    print(f'style card: {a.song}  ({card["seconds"]} s)')
    print('sections (start s, bars, level dB' + (', layers voice/drums/bass/other dB, other layer' if layers else '') + ', sung line bars):')
    for s in secs:
        lay = ' '.join(f"{v:>6}" for v in s['layers_db'].values()) + f"  {s['other_hint']}" if layers else ''
        print(f"  {s['start']:>6} {s['bars']:>5} {s['level_db']:>6}  {lay}  {s.get('sung_line_bars')}")
    if 'lines' in card:
        L = card['lines']
        print(f"lines: {L['sung']} sung, {L['not_sung_or_unsure']} not sung or unsure; sung lines median {L['line_seconds_median']} s ({L['line_bars_median']} bars),"
              f" sung {L['share_sung']:.0%} / not sung {L['share_not_sung']:.0%} of the song, range {L['range_semitones_sung']} semitones,"
              f" highest note at {L['highest_note_position']} of a line, {L['lines_ending_lower']} end lower")
        for r in L['spoken_parts']:
            print(f"  not sung: {r['start']}-{r['end']} s, {r['lines']} lines ({'/'.join(r['kinds'])}), {r['where']}, over music: {r['over_music']}, overlaps singing: {r['overlaps_singing']}")
    print('settled by measurement:')
    for x in settled:
        print('  -', x)
    print('待人听核 (a person listens):')
    for x in listen:
        print('  -', x)
    if warn:
        print('warnings:')
        for x in warn:
            print('  !', x)
    print('time per stage (s):', timings)


if __name__ == '__main__':
    main()
