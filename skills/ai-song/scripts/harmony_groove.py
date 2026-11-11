"""Chords, scale colour and groove per section: the things that make a style sound like itself, which key and tempo don't
catch. For 扒曲 (the reference) and for checking form samples against it before anyone listens.

In one test, three form samples sat at the reference's tempo and were heard as nothing like it. The reference's chorus ran
the Andalusian cadence (i-bVII-bVI-V) with the voice leaning on the flat 2nd over the dominant, and its drums played 3+3+2;
the samples had plain pop chords and a backbeat on 2 and 4. The style text had asked for "3+3+2" in words, which did nothing.
Per section it measures:
- chords: one per bar (--per-bar 2 for two), the best of 24 major/minor triads on the chroma of --harmony (the 'other' stem
  of separate.py --model parts: no drums, no voice; else the backing), 'N' when none fits. On a phone recording the backing
  stem and the 'other' stem agreed on only 25-60% of bars, and the backing read one chorus a whole tone off, so chords are
  the roughest of the three measurements. The key is a profile fit on the same layer. Roman numerals are in that key;
  named progressions are searched in every key: Andalusian i-bVII-bVI-V, pop
  I-V-vi-IV (and its rotations, e.g. vi-IV-I-V), 50s I-vi-IV-V, royal road IV-V-iii-vi, Mixolydian I-bVII-IV, Phrygian
  vamp I-bII;
- scale colour: the voice's pitch classes (YIN on the vocal stem, tuning offset removed) relative to the key's tonic, with
  characteristic notes flagged: b2 (Phrygian; over the major V of a minor key: flamenco / Phrygian dominant), #4 (Lydian),
  b7 in major (Mixolydian), raised 7 in minor (harmonic minor), 6 in minor (Dorian), b3 in major (blue note);
- groove: the accent profile per bar in eighths on the drum stem (separate.py --model parts) or the backing, a low-band
  (kick) and a mid-band (snare) profile, and a two-bar profile, scored against named patterns: 3+3+2 (tresillo), son clave,
  backbeat on 2 and 4, four on the floor, half-time, straight eighths.
Results come with plain words for the style card. They are measurements on separated stems of a possibly lossy recording:
hints for the person to confirm by ear. The pattern thresholds were set on one reference and three samples.
--compare ref.json sets a sample's sections next to the reference's (same name, or --pair chorus=chorus1) and marks chords,
colour and groove as match, miss or not counted:
- chords: if the reference's progression is pinned (--pin SECTION="C G Am F" when the reference is measured, after the
  person confirmed it, or --ref-chords at compare time), the sample's chord changes must follow that cycle in some key on
  half of its changes and bars. In one test this script read the reference chorus's Andalusian cadence as another
  progression and then marked a sample that played the cadence as a miss; pinning fixes that. Unpinned, both must fit the
  same named progression on 70% of their changes, or their chord changes (root interval and qualities, key-free) reach a
  cosine of 0.5. The sample's closest named progressions are always listed;
- colour: a flagged note of the reference is flagged in the sample too (not counted where the reference has none);
- groove: phase-free, i.e. at the best rotation of the bar, because the downbeat guess is often a few eighths off and the
  fixed-phase reading then calls a 3+3+2 half-time: profiles correlate at 0.7, or both are named 3+3+2 / son clave. The
  fixed-phase reading is printed next to it. Not counted where the drum layer is 30 dB below the backing (no drums there).
A column matches overall only when it matches in every section where it counts (pinned sections alone decide chords).
Chords weigh 1 when pinned and 0.5 otherwise, and unpinned chords never decide alone. Verdicts: MATCH; PARTIAL (which
columns miss); REJECT (colour and groove miss, and chords are not a pinned match); NOT MEASURED (a paired section was not
found by take_sections.py or has under half its lines: fix the timing first); NOT DECIDED (only chords compared).
With a takes.json grid, --grid-from mix uses the mix's grid when take_grid flagged a tempo disagreement.
In two tests: the three samples the person rejected came out REJECT; after genre words, the two takes with the cadence and
a 3+3+2 chorus came out PARTIAL (colour missing), and none matched all three. Measure both on the same kind
of layer (drums with drums, 'other' with 'other'); on a backing stem a strummed guitar hides the drum pattern.
Reference results are analysis of someone else's song: keep them in the private working folder.
numpy and ffmpeg only (the beat grid comes from take_grid.py).
usage: harmony_groove.py --backing inst.flac [--drums drums.wav] [--harmony other.wav] [--vocals vocals.flac]
           [--card card.json | --sections sections.json --id TAKE | --span verse=33-50 --span chorus=50-86.5]
           [--grid takes.json --id TAKE [--grid-from stem|mix] | --bpm 103 --downbeat 1.70] [--bpm-hint 105] [--key "G minor"]
           [--per-bar 1] [--pin chorus="C G Am F"] [--out hg]
           [--compare ref_hg.json [--pair chorus=chorus1 ...] [--ref-chords chorus="C G Am F"]]
  sections: from analyze_ref.py's card (named s1, s2 ...), take_sections.py's output, or --span name=start-end
  grid: per section from the backing (default; tempo folded as in take_grid.py), or one grid for the whole file
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import take_grid as tg  # noqa: E402

SR, FR, NAMES = tg.SR, tg.FR, tg.NAMES
ROMAN = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']
DEG = ['1', 'b2', '2', 'b3', '3', '4', '#4', '5', 'b6', '6', 'b7', '7']
TRIADS = {}
for _r in range(12):
    for _q, _iv in (('', (0, 4, 7)), ('m', (0, 3, 7))):
        _v = np.zeros(12)
        _v[[(_r + i) % 12 for i in _iv]] = 1
        TRIADS[(_r, _q)] = _v / np.linalg.norm(_v)
# cyclic progressions: (semitones above the progression's tonic, quality)
PROGRESSIONS = {
    'Andalusian cadence i-bVII-bVI-V': [(0, 'm'), (10, ''), (8, ''), (7, '')],
    'pop four chords I-V-vi-IV': [(0, ''), (7, ''), (9, 'm'), (5, '')],
    '50s I-vi-IV-V': [(0, ''), (9, 'm'), (5, ''), (7, '')],
    'royal road IV-V-iii-vi': [(5, ''), (7, ''), (4, 'm'), (9, 'm')],
    'Mixolydian I-bVII-IV': [(0, ''), (10, ''), (5, '')],
    'Phrygian vamp I-bII': [(0, ''), (1, '')],
}
PROG_ZH = {'Andalusian cadence i-bVII-bVI-V': '安达卢西亚终止 i–bVII–bVI–V（弗拉门戈常用）',
           'pop four chords I-V-vi-IV': '流行四和弦 I–V–vi–IV（或它的轮换，比如 vi–IV–I–V）',
           '50s I-vi-IV-V': '五十年代进行 I–vi–IV–V', 'royal road IV-V-iii-vi': '王道进行 IV–V–iii–vi',
           'Mixolydian I-bVII-IV': '混合利底亚 I–bVII–IV', 'Phrygian vamp I-bII': '弗里吉亚两和弦 I–bII'}


def chord_name(c):
    return 'N' if c is None else NAMES[c[0]] + c[1]


def chroma(x):
    """Chroma frames (65-2100 Hz, log-compressed) and their times."""
    S = tg.stft_mag(x, 8192, 2048)
    f = np.fft.rfftfreq(8192, 1 / SR)
    band = (f >= 65) & (f < 2100)
    pc = np.round(69 + 12 * np.log2(f[band] / 440)).astype(int) % 12
    Sb = np.log1p(20 * S[:, band] / (S[:, band].max() + 1e-12))
    C = np.stack([Sb[:, pc == k].sum(axis=1) for k in range(12)], axis=1)
    return C, (np.arange(len(C)) * 2048 + 4096) / SR


def best_chord(c):
    """The triad whose template correlates best with the chroma (Pearson, so a flat floor doesn't count); None under 0.4."""
    sc = sorted(((float(np.corrcoef(v, c)[0, 1]), k) for k, v in TRIADS.items()), reverse=True)
    return (sc[0][1] if sc[0][0] >= .4 else None), round(sc[0][0], 2), round(sc[0][0] - sc[1][0], 3)


def parse_chords(text):
    """'C G Am F' (spaces, commas or dashes) -> [(root pc, '' or 'm')]."""
    out = []
    for tok in text.replace('–', ' ').replace('-', ' ').replace(',', ' ').split():
        minor = tok.endswith('m') and not tok.endswith('maj')
        root = tok[:-1] if minor else tok
        root = {'Db': 'C#', 'D#': 'Eb', 'Gb': 'F#', 'G#': 'Ab', 'A#': 'Bb'}.get(root, root)
        out.append((NAMES.index(root), 'm' if minor else ''))
    return out


def fit_cycle(chords, pat):
    """How well one cyclic progression (pattern of (semitones above its tonic, quality)) explains the chords, in the best key:
    (share of chord changes that follow it, share of bars inside it, tonic pc, distinct chords of it used)."""
    seq = [c for c in chords if c is not None]
    seq = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    pairs = list(zip(seq, seq[1:]))
    best = (0., 0., 0, 0)
    for t in range(12):
        cyc = [((t + o) % 12, q) for o, q in pat]
        nxt = {cyc[i]: cyc[(i + 1) % len(cyc)] for i in range(len(cyc))}
        share = sum(1 for p, q in pairs if nxt.get(p) == q) / len(pairs) if pairs else 0.
        inside = sum(1 for c in chords if c in cyc) / max(1, len(chords))
        distinct = len({c for c in seq if c in cyc})
        if share * inside > best[0] * best[1]:
            best = (share, inside, t, distinct)
    return best


def progression_candidates(chords, top=3):
    """Every named progression with its best key, sorted by fit: listed even when none passes, so a reader sees what was close."""
    out = []
    for name, pat in PROGRESSIONS.items():
        share, inside, t, distinct = fit_cycle(chords, pat)
        out.append({'name': name, 'tonic': NAMES[t], 'changes_matched': round(share, 2), 'bars_inside': round(inside, 2),
                    'chords': [chord_name(((t + o) % 12, q)) for o, q in pat], 'distinct': distinct})
    out.sort(key=lambda d: -d['changes_matched'] * d['bars_inside'])
    return out[:top]


def name_progression(chords):
    """The named progression that best explains the chord changes (at least half of the changes and half of the bars, and
    three of its chords used), in whatever key; None when nothing passes."""
    for c in progression_candidates(chords, top=len(PROGRESSIONS)):
        if c['changes_matched'] >= .5 and c['bars_inside'] >= .5 and c['distinct'] >= min(3, len(PROGRESSIONS[c['name']])):
            return {k: c[k] for k in ('name', 'tonic', 'changes_matched', 'chords', 'bars_inside')}
    return None


def yin(x, fmin=80., fmax=1000., frame=2048, hop=512, thr=.15):
    """f0 per frame (nan where unvoiced or too quiet), plain YIN with an FFT difference function."""
    n = (len(x) - frame) // hop + 1
    if n <= 0:
        return np.array([])
    W = frame // 2
    tmin, tmax = int(SR / fmax), min(W - 2, int(SR / fmin))
    f0, rms = np.full(n, np.nan), np.zeros(n)
    nfft = 1 << int(np.ceil(np.log2(frame + W)))
    for c0 in range(0, n, 256):
        idx = np.arange(c0, min(n, c0 + 256))
        F = np.stack([x[i * hop:i * hop + frame] for i in idx])
        a = F[:, :W]
        rms[idx] = np.sqrt((a ** 2).mean(axis=1))
        r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(F, nfft), nfft)[:, :W]
        cs = np.cumsum(np.pad(F ** 2, ((0, 0), (1, 0))), axis=1)
        d = (a ** 2).sum(axis=1, keepdims=True) + cs[:, W:2 * W] - cs[:, :W] - 2 * r
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] * np.arange(1, W) / (np.cumsum(d[:, 1:], axis=1) + 1e-12)
        for row, i in enumerate(idx):
            seg = dn[row, tmin:tmax]
            below = np.flatnonzero(seg < thr)
            if len(below):
                k = below[0]
                while k + 1 < len(seg) and seg[k + 1] < seg[k]:
                    k += 1
            else:
                k = int(np.argmin(seg))
                if seg[k] > .25:
                    continue
            t = k + tmin
            y0, y1, y2 = dn[row, t - 1], dn[row, t], dn[row, t + 1]
            f0[i] = SR / (t + .5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12))
    f0[rms < .1 * np.percentile(rms, 95)] = np.nan
    return f0


def degrees(f0, tonic):
    """Share of voiced time on each scale degree relative to `tonic` (pitch class), tuning offset removed."""
    f = f0[~np.isnan(f0)]
    if len(f) < 20:
        return None
    m = 69 + 12 * np.log2(f / 440)
    off = np.angle(np.mean(np.exp(2j * np.pi * (m - np.round(m))))) / (2 * np.pi)
    pcs = (np.round(m - off).astype(int) - tonic) % 12
    h = np.bincount(pcs, minlength=12) / len(pcs)
    return {DEG[i]: round(float(h[i]), 3) for i in range(12)}


def colours(deg, minor, over_v=None):
    if not deg:
        return []
    out = []
    if minor:
        if deg['b2'] >= .08:
            out.append('b2 (Phrygian)')
        if deg['7'] >= .05 and deg['7'] > deg['b7']:
            out.append('raised 7 (harmonic minor)')
        if deg['6'] >= .08 and deg['6'] > deg['b6']:
            out.append('6 (Dorian)')
    else:
        if deg['#4'] >= .05 and deg['#4'] > deg['4']:
            out.append('#4 (Lydian)')
        if deg['b7'] >= .06 and deg['b7'] > deg['7']:
            out.append('b7 (Mixolydian)')
        if deg['b3'] >= .08:
            out.append('b3 (blue note)')
    if over_v and over_v['b2'] >= .12:
        out.append('b2 over V (flamenco / Phrygian dominant)')
    return out


def v_flag(over_v, v_major):
    """Pass the over-V degrees on for the flamenco flag when the V chord was seen as major, or when b2 over V is strong
    enough on its own (chords from a phone recording miss a major V often)."""
    return over_v if over_v and (v_major or over_v['b2'] >= .15) else None


COLOUR_ZH = {'b2 (Phrygian)': '常落在 b2 上（弗里吉亚色彩）', 'raised 7 (harmonic minor)': '小调里用升高的 7 级（和声小调）',
             '6 (Dorian)': '小调里用还原 6 级（多利亚）', '#4 (Lydian)': '大调里用 #4（利底亚）',
             'b7 (Mixolydian)': '大调里用 b7（混合利底亚）', 'b3 (blue note)': '大调里常落在 b3（蓝调音）',
             'b2 over V (flamenco / Phrygian dominant)': '在属和弦（V）上方常落 b2（弗拉门戈 / 弗里吉亚属音阶的色彩）'}


def accent(o, spans, n):
    """Mean onset strength at n even positions inside each (start, end) span (max within +-35 ms), scaled to max 1."""
    w = int(.035 * FR)
    prof = np.zeros(n)
    for b0, b1 in spans:
        for k in range(n):
            i = int(round((b0 + (b1 - b0) * k / n) * FR))
            prof[k] += o[max(0, i - w):i + w + 1].max(initial=0)
    return prof / (prof.max() + 1e-12)


def beat_grid(x, hint, s0, s1, bar_line=None):
    """Beats of one section, tracked (not a straight line: a band's tempo drifts), at the pulse take_grid.choose_pulse
    picks (a 3+3+2 drum part tracks at the dotted-quarter pulse otherwise), and the bar lines: every 4th beat, phase from
    --bar-line if given, else the phase with the most low-band (kick) onset. Returns (beats, bar lines, BPM, phase ratio)."""
    a0 = max(0., s0 - 2)
    seg = x[int(a0 * SR):int((s1 + 2) * SR)]
    o = tg.onset_env(seg)
    P = tg.beat_period(o, hint)
    beats = tg.track(o, P)
    p = tg.fold(tg.fit_grid(beats, s0 - a0, s1 - a0)[0], hint)
    q = tg.choose_pulse(o, p, hint, s0 - a0, s1 - a0)[0]
    if abs(q * FR / P - 1) > .02:
        beats = tg.track(o, q * FR)
    beats = beats[(beats >= s0 - a0 - .1) & (beats <= s1 - a0 + .1)] + a0
    if len(beats) < 8:
        return beats, beats[:0], 60 / q, 0.
    if bar_line is not None:
        k = int(np.argmin(np.abs(beats - bar_line))) if s0 <= bar_line <= s1 else \
            int(np.argmin(np.abs(((beats - bar_line) / q + .5) % 4 - .5)))
        ph, ratio = k % 4, None
    else:
        o_low = tg.onset_env(seg, fmax=150)
        sc = [float(np.mean(np.interp(beats[ph::4] - a0, tg.frame_time(np.arange(len(o_low))), o_low))) for ph in range(4)]
        ph = int(np.argmax(sc))
        srt = sorted(sc)
        ratio = round(srt[-1] / (srt[-2] + 1e-9), 2)
    return beats, beats[ph::4], float(60 / np.median(np.diff(beats))), ratio


def groove(o, o_low, o_mid, spans):
    p8, pl, pm = accent(o, spans, 8), accent(o_low, spans, 8), accent(o_mid, spans, 8)
    pairs = [(spans[i][0], spans[i + 1][1]) for i in range(0, len(spans) - 1, 2)]
    p16 = accent(o, pairs, 16) if len(pairs) >= 2 else None

    def contrast(p, on):
        off = [k for k in range(len(p)) if k not in on]
        return float(np.mean(p[on]) - np.mean(p[off]))
    tres = [contrast(np.roll(p8, -r), [0, 3, 6]) for r in range(8)]
    sc = {'3+3+2': round(tres[0], 2), 'straight 8ths': round(contrast(p8, [0, 2, 4, 6]), 2),
          'backbeat': round(float(np.mean(pm[[2, 6]]) - np.mean(pm[[0, 4]])), 2),
          'half-time': round(float(pm[4] - np.mean(pm[[2, 6]])), 2),
          'four on the floor': round(float(pl[[0, 2, 4, 6]].min() / (pl[[0, 2, 4, 6]].max() + 1e-12)
                                           - np.mean(pl[[1, 3, 5, 7]]) / (np.mean(pl[[0, 2, 4, 6]]) + 1e-12)), 2)}
    if p16 is not None:
        sc['son clave'] = round(max(contrast(p16, [0, 3, 6, 10, 12]), contrast(p16, [2, 4, 8, 11, 14])), 2)
    names = []
    if sc['3+3+2'] >= .25 and sc['3+3+2'] > sc['straight 8ths']:
        names.append('3+3+2 (tresillo)')
    if sc.get('son clave', 0) >= .3 and sc['son clave'] > sc['3+3+2'] + .05:
        names.append('son clave')
    if sc['backbeat'] >= .15 and p8[3] < .5 * np.mean(p8[[2, 6]]):    # 2 and 4 without the 3+3+2 accent between them
        names.append('backbeat on 2 and 4')
    if sc['half-time'] >= .15:
        names.append('half-time (snare on 3)')
    if sc['four on the floor'] >= .3:
        names.append('four on the floor')
    if not names and sc['straight 8ths'] >= .25:
        names.append('straight eighths')
    out = {'profile_8ths': [round(float(v), 2) for v in p8], 'kick_band': [round(float(v), 2) for v in pl],
           'snare_band': [round(float(v), 2) for v in pm], 'scores': sc, 'names': names}
    if p16 is not None:
        out['two_bars_8ths'] = [round(float(v), 2) for v in p16]
    # phase-free: the same patterns at every rotation, so a wrong downbeat guess can't hide them. It can't tell a backbeat
    # from half-time or four on the floor (those are rotations of one another), so those keep their phase-based names.
    free = {'3+3+2': round(max(tres), 2),
            'straight 8ths': round(max(contrast(p8, [0, 2, 4, 6]), contrast(p8, [1, 3, 5, 7])), 2)}
    if p16 is not None:
        free['son clave'] = round(max(contrast(np.roll(p16, -r), [0, 3, 6, 10, 12]) for r in range(16)), 2)
    names_free = []
    if free['3+3+2'] >= .25 and free['3+3+2'] > free['straight 8ths']:
        names_free.append('3+3+2 (tresillo)')
    elif free['straight 8ths'] >= .25:
        names_free.append('straight (backbeat, half-time or four on the floor)')
    if free.get('son clave', 0) >= .3 and free['son clave'] > free['3+3+2'] + .05:
        names_free.append('son clave')
    out['scores_phase_free'], out['names_phase_free'] = free, names_free
    best_shift = int(np.argmax(tres))
    if best_shift and tres[best_shift] > tres[0] + .15:
        out['note'] = f'3+3+2 fits better shifted by {best_shift} eighths: the downbeat may be off (the phase-free reading counts it)'
    return out


GROOVE_ZH = {'3+3+2 (tresillo)': '3+3+2（tresillo）', 'son clave': '颂克拉维（son clave）', 'backbeat on 2 and 4': '二四拍的反拍军鼓（流行的 backbeat）',
             'half-time (snare on 3)': '半拍感（军鼓在第三拍）', 'four on the floor': '四拍都是底鼓（four on the floor）',
             'straight eighths': '平均的八分音符'}


def sections_from(a, dur):
    """[(name, start, end, why not measured or None)]."""
    if a.span:
        out = []
        for s in a.span:
            name, rng = s.split('=')
            s0, s1 = (float(v) for v in rng.split('-'))
            out.append((name, s0, s1, None))
        return out
    if a.sections:
        row = {r['id']: r for r in json.load(open(a.sections, encoding='utf-8'))}[a.id]
        out = []
        for s in row['sections']:
            why = ('not found in the take' if s.get('start') is None else
                   f"only {s['heard_lines']} of {s['lines']} lines found by take_sections.py" if s['heard_lines'] < s['lines'] / 2 else None)
            out.append((s['name'], s.get('start') or 0., s.get('end') or 0., why))
        return out
    if a.card:
        card = json.load(open(a.card, encoding='utf-8'))
        return [(f's{i + 1}', float(s['start']), float(s['start']) + float(s['seconds']), None) for i, s in enumerate(card['sections'])]
    return [('whole', 0., dur, None)]


def measure(a):
    xb = tg.load_mono(a.backing)
    dur = len(xb) / SR
    xd = tg.load_mono(a.drums) if a.drums else xb
    xh = tg.load_mono(a.harmony) if a.harmony else xb
    o, o_low, o_mid = tg.onset_env(xd), tg.onset_env(xd, fmax=150), tg.onset_env(xd, 200, 2500)
    C, ct = chroma(xh)
    xv = tg.load_mono(a.vocals) if a.vocals else None
    fixed = None
    pins = dict(p.split('=', 1) if '=' in p else ('*', p) for p in a.pin)
    if a.grid:
        g = {r['id']: r for r in json.load(open(a.grid, encoding='utf-8'))}[a.id]
        if a.grid_from == 'mix':
            if not g.get('grid_mix'):
                raise SystemExit('no grid_mix in that takes.json row: run take_grid.py --stems again')
            fixed = (g['grid_mix']['beat_s'], g['grid_mix']['downbeat'])
        else:
            fixed = (g['beat_s'], g['downbeat'])
            if g.get('tempo_agree') is False:
                print(f"warning: take_grid says backing ({g['bpm']}) and mix ({g.get('bpm_mix')}) disagree on the tempo; "
                      f"this uses the backing's grid; --grid-from mix uses the mix's")
    elif a.bpm and a.downbeat is not None:
        fixed = (60 / a.bpm, a.downbeat)
    rows = []
    for name, s0, s1, why in sections_from(a, dur):
        if why or s1 - s0 < 6:
            rows.append({'name': name, 'start': round(s0, 2), 'end': round(s1, 2), 'skipped': why or 'shorter than 6 s'})
            continue
        ratio = None
        if fixed:
            p, d = fixed
            k0 = int(np.ceil((s0 - d) / p - .1))
            beats = np.array([d + k * p for k in range(k0, k0 + 2000) if d + k * p <= s1 + .1])
            bars = beats[((np.round((beats - d) / p)).astype(int) % 4) == 0]
            bpm = 60 / p
        else:
            beats, bars, bpm, ratio = beat_grid(xd, a.bpm_hint, s0, s1, a.bar_line)
        ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 60 / bpm
        ends = list(bars[1:]) + [bars[-1] + 4 * ibi] if len(bars) else []
        spans = [(b0, b1) for b0, b1 in zip(bars, ends) if b1 <= s1 + ibi]
        if len(spans) < 2:
            rows.append({'name': name, 'start': round(s0, 2), 'end': round(s1, 2), 'skipped': 'fewer than 2 whole bars'})
            continue
        chords, conf = [], []
        for b0, b1 in spans:
            step = (b1 - b0) / a.per_bar
            for h in range(a.per_bar):
                ww = (ct >= b0 + h * step) & (ct < b0 + (h + 1) * step)
                c, s, gap = best_chord(C[ww].mean(axis=0)) if ww.any() else (None, 0., 0.)
                chords.append(c)
                conf.append(s)
        prog = name_progression(chords)
        fits = tg.key_fits(xh[int(s0 * SR):int(s1 * SR)])
        if a.key:
            key, key_from = a.key, 'given'
        elif prog and fits[0][0] - fits[1][0] < .03:   # a near tie: the progression's tonic decides
            mode = 'minor' if PROGRESSIONS[prog['name']][0][1] == 'm' else 'major'
            key, key_from = f"{prog['tonic']} {mode}", 'progression'
        else:
            key, key_from = fits[0][1], 'chroma profile'
        tonic, minor = NAMES.index(key.split()[0]), key.split()[1] == 'minor'
        roman = ['N' if c is None else (ROMAN[(c[0] - tonic) % 12].lower() if c[1] == 'm' else ROMAN[(c[0] - tonic) % 12]) for c in chords]
        row = {'name': name, 'start': round(s0, 2), 'end': round(s1, 2), 'bpm': round(bpm, 2), 'first_bar_line': round(float(spans[0][0]), 3),
               'downbeat_ratio': ratio, 'bars': len(spans), 'key': key, 'key_from': key_from, 'key_profile': fits[0][1],
               'key_profile_next': fits[1][1], 'key_profile_gap': round(fits[0][0] - fits[1][0], 3),
               'chords': [chord_name(c) for c in chords], 'roman': roman,
               'chord_fit_median': round(float(np.median(conf)), 2), 'progression': prog,
               'progression_candidates': progression_candidates(chords),
               'groove_layer': 'drums' if a.drums else 'backing', 'groove': groove(o, o_low, o_mid, spans)}
        if pins.get(name) or pins.get('*'):
            row['chords_pinned'] = pins.get(name) or pins.get('*')
        if a.drums:                                     # a near-silent drum layer gives a pattern of noise
            lv = lambda x: 20 * np.log10(np.sqrt(np.mean(x[int(s0 * SR):int(s1 * SR)] ** 2)) + 1e-9)  # noqa: E731
            gap_db = lv(xd) - lv(xb)
            row['groove']['drums_below_backing_db'] = round(float(-gap_db), 1)
            if gap_db < -30:
                row['groove']['not_measured'] = f'the drum layer is {-gap_db:.0f} dB below the backing here: no drums to read'
        if xv is not None:
            f0 = yin(xv[int(s0 * SR):int(s1 * SR)])
            row['voiced_seconds'] = round(float(np.sum(~np.isnan(f0)) * 512 / SR), 1)
            row['degrees'] = degrees(f0, tonic)
            over_v = None
            v_root = (tonic + 7) % 12
            if minor:                                   # the flamenco view: the same notes relative to the dominant
                over_v = degrees(f0, v_root)
                row['degrees_over_V'] = over_v
            v_major = any(c == (v_root, '') for c in chords)
            row['colour'] = colours(row['degrees'], minor, v_flag(over_v, v_major))
        row['words'] = words(row)
        rows.append(row)
    return {'backing': str(a.backing), 'drums': str(a.drums) if a.drums else None, 'vocals': str(a.vocals) if a.vocals else None,
            'harmony': str(a.harmony) if a.harmony else None,
            'note': 'measured on separated stems of a possibly lossy recording: hints for the person to confirm by ear',
            'sections': rows}


def words(r):
    out = []
    seq = ' '.join(r['chords'][:16])
    pr = r.get('progression')
    if pr:
        out.append(f"和弦（每小节，从伴奏层量）：{seq}；走向像{PROG_ZH[pr['name']]}，在 {pr['tonic']} 上"
                   f"（{'–'.join(pr['chords'])}），换和弦时 {pr['changes_matched']:.0%} 照这个走")
    else:
        out.append(f'和弦（每小节，从伴奏层量）：{seq}；没认出常见的套路')
    if r.get('degrees'):
        top = sorted(r['degrees'].items(), key=lambda kv: -kv[1])[:4]
        line = f"人声音阶（相对 {r['key']} 的主音）：" + '，'.join(f'{d} {v:.2f}' for d, v in top)
        if r.get('degrees_over_V'):
            tv = sorted(r['degrees_over_V'].items(), key=lambda kv: -kv[1])[:4]
            line += '；相对属音（V）：' + '，'.join(f'{d} {v:.2f}' for d, v in tv)
        if r.get('colour'):
            line += '；' + '；'.join(COLOUR_ZH[c] for c in r['colour'])
        out.append(line)
    g = r['groove']
    layer = '鼓层' if r['groove_layer'] == 'drums' else '伴奏层'
    names = [GROOVE_ZH[n] for n in g['names']]
    if '3+3+2 (tresillo)' in g.get('names_phase_free', []) and '3+3+2 (tresillo)' not in g['names']:
        names.append('3+3+2（不管小节从哪拍算起）')
    out.append(f"{layer}每小节八个八分音符的重音：{' '.join(f'{v:.2f}' for v in g['profile_8ths'])}；"
               + ('像' + '、'.join(names) if names else '没认出常见的律动'))
    return out


def rot_corr(p, q):
    """Best correlation of profile p with q over every rotation of p: the bar phase doesn't matter."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return max(float(np.corrcoef(np.roll(p, r), q)[0, 1]) for r in range(len(p)))


def chord_tuples(names):
    return [None if c == 'N' else parse_chords(c)[0] for c in names]


def compare(me, ref, pairs, ref_pins):
    """Per section: chords, colour, groove as True (match), False (miss) or None (not counted: the reference has no flagged
    colour there). A column matches overall only when it matches in every section where it counts.
    Chords are the roughest column, so they carry a weight: 1 when the reference's progression is pinned (--ref-chords, or
    --pin when the reference was measured, by the person or from a reading they confirmed), 0.5 when it is only this
    script's own reading. When some sections are pinned, only those decide the chords column. Unpinned chords never decide
    alone: they can't save a sample whose colour and groove both miss, or sink one whose colour and groove both match.
    Groove is judged phase-free (every rotation of the bar), because the downbeat guess is often off by a few eighths; the
    fixed-phase reading is shown next to it.
    A paired section the sample doesn't have measured (take_sections.py didn't find it, or too few lines) makes the verdict
    NOT MEASURED, not REJECT."""
    refs = {s['name']: s for s in ref['sections'] if 'skipped' not in s}
    rows, missing = [], []
    for s in me['sections']:
        rn = pairs.get(s['name'], s['name'])
        if 'skipped' in s:
            if rn in refs:
                missing.append(f"{s['name']} ({s['skipped']})")
            continue
        r = refs.get(rn)
        if r is None:
            print(f"{s['name']}: no reference section '{rn}' (use --pair {s['name']}={'|'.join(refs)})")
            continue
        mine = chord_tuples(s['chords'])
        pin = ref_pins.get(rn) or ref_pins.get('*') or r.get('chords_pinned')
        if pin:
            pc = parse_chords(pin)
            pat = [((c[0] - pc[0][0]) % 12, c[1]) for c in pc]
            share, inside, t, distinct = fit_cycle(mine, pat)
            ch_ok = share >= .5 and inside >= .5 and distinct >= min(3, len(pat))
            ch_info = f"pinned {pin}: {share:.0%} of changes, {inside:.0%} of bars, on {NAMES[t]}"
            w_ch = 1.
        else:
            csim = float(np.dot(change_hist(s['chords']), change_hist(r['chords'])))
            strong = lambda x: (x.get('progression') or {}).get('name') if (x.get('progression') or {}).get('changes_matched', 0) >= .7 else None  # noqa: E731
            ch_ok = (strong(s) is not None and strong(s) == strong(r)) or csim >= CHANGE_SIM
            ch_info = f"change similarity {csim:.2f}; reference read as {(r.get('progression') or {}).get('name') or '-'}"
            w_ch = .5
        cand = (s.get('progression_candidates') or progression_candidates(mine))[0]
        found = (f"{cand['name']} on {cand['tonic']}, {cand['changes_matched']:.0%} of changes, {cand['bars_inside']:.0%} of bars"
                 if cand['changes_matched'] * cand['bars_inside'] > 0 else 'none')
        cs, cr = set(s.get('colour') or []), set(r.get('colour') or [])
        col_ok = bool(cs & cr) if cr else None
        gs, gr = s['groove'], r['groove']
        corr_fixed = float(np.corrcoef(gs['profile_8ths'], gr['profile_8ths'])[0, 1])
        corr_free = rot_corr(gs['profile_8ths'], gr['profile_8ths'])
        shared = set(gs.get('names_phase_free', [])) & set(gr.get('names_phase_free', [])) & {'3+3+2 (tresillo)', 'son clave'}
        gr_ok = None if gs.get('not_measured') or gr.get('not_measured') else (corr_free >= .7 or bool(shared))
        rows.append({'section': s['name'], 'reference': rn, 'chords_match': ch_ok, 'chords_weight': w_ch, 'chords_pinned': bool(pin),
                     'chords_detail': ch_info,
                     'progression_found': found, 'colour_match': col_ok, 'colour': sorted(cs), 'reference_colour': sorted(cr),
                     'groove_match': gr_ok, 'groove_corr_phase_free': round(corr_free, 2), 'groove_corr_fixed': round(corr_fixed, 2),
                     'groove_phase_free': gs.get('names_phase_free', []), 'groove_fixed': gs['names'],
                     'reference_groove_phase_free': gr.get('names_phase_free', []), 'layers': f"{s['groove_layer']} vs {r['groove_layer']}"})
    mark = {True: 'match', False: 'MISS', None: 'n/a'}
    for x in rows:
        print(f"{x['section']} vs reference {x['reference']}:")
        print(f"   chords {mark[x['chords_match']]} (weight {x['chords_weight']:g}): {x['chords_detail']}; best named fit here: {x['progression_found']}")
        print(f"   colour {mark[x['colour_match']]}: {', '.join(x['colour']) or '-'} vs {', '.join(x['reference_colour']) or '-'}")
        print(f"   groove {mark[x['groove_match']]}: phase-free r {x['groove_corr_phase_free']:+.2f} {'/'.join(x['groove_phase_free']) or '-'} "
              f"vs {'/'.join(x['reference_groove_phase_free']) or '-'}; fixed phase r {x['groove_corr_fixed']:+.2f} {'/'.join(x['groove_fixed']) or '-'}")
        if x['layers'].split(' vs ')[0] != x['layers'].split(' vs ')[1]:
            print(f"   groove measured on {x['layers']}: compare like with like")
    overall = {}
    pinned_rows = [x for x in rows if x['chords_pinned']]
    for col in ('chords_match', 'colour_match', 'groove_match'):
        use = pinned_rows if col == 'chords_match' and pinned_rows else rows    # pinned sections speak for the chords
        vals = [x[col] for x in use if x[col] is not None]
        overall[col.split('_')[0]] = None if not vals else all(vals)
    w_ch = 1. if pinned_rows else .5
    weights = {'chords': w_ch, 'colour': 1., 'groove': 1.}
    counted = {k: v for k, v in overall.items() if v is not None}
    score = sum(weights[k] for k, v in counted.items() if v) / (sum(weights[k] for k in counted) or 1)
    others = [v for k, v in counted.items() if k != 'chords']
    ch = counted.get('chords')
    if missing:
        verdict = 'NOT MEASURED: ' + '; '.join(missing) + ' (fix the timing, e.g. take_sections.py, then run again)'
    elif not others:
        verdict = 'NOT DECIDED: only chords could be compared'
    elif not any(others) and (ch is not True or w_ch < 1):
        verdict = 'REJECT: ' + ('chords, colour and groove all miss' if ch is False else 'colour and groove miss')
    elif all(others) and (ch is not False or w_ch < 1):
        verdict = 'MATCH' + (' (chords differ, but unpinned chords are only a hint)' if ch is False else '')
    else:
        verdict = 'PARTIAL: misses ' + ', '.join(k for k, v in counted.items() if v is False)
    print(f"overall: {', '.join(f'{k} {mark[v]}' for k, v in overall.items())}; weighted score {score:.2f} -> {verdict}")
    return {'sections': rows, 'overall': overall, 'weights': weights, 'score': round(score, 2), 'not_measured': missing, 'verdict': verdict}


CHANGE_SIM = .5


def change_hist(chords):
    """Chord changes as (root interval, quality before, quality after), counted: the same in every key, so a sample in
    another key still compares, while two different progressions of the same diatonic chords do not."""
    seq = [c for c in chords if c != 'N']
    seq = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    h = np.zeros(48)
    for p, q in zip(seq, seq[1:]):
        rp, rq = NAMES.index(p.rstrip('m')), NAMES.index(q.rstrip('m'))
        h[((rq - rp) % 12) * 4 + 2 * p.endswith('m') + q.endswith('m')] += 1
    return h / (np.linalg.norm(h) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backing', required=True)
    ap.add_argument('--drums')
    ap.add_argument('--harmony', help="the layer to read chords and key from: the 'other' stem of separate.py --model parts is "
                                      "best (no drums, no voice); default the backing")
    ap.add_argument('--vocals')
    ap.add_argument('--card')
    ap.add_argument('--sections')
    ap.add_argument('--id')
    ap.add_argument('--span', action='append')
    ap.add_argument('--grid')
    ap.add_argument('--bpm', type=float)
    ap.add_argument('--downbeat', type=float)
    ap.add_argument('--bpm-hint', type=float)
    ap.add_argument('--bar-line', type=float, help='the time of any bar line, to fix the bar phase (from the song map or by ear)')
    ap.add_argument('--key')
    ap.add_argument('--per-bar', type=int, default=1)
    ap.add_argument('--out', default='hg')
    ap.add_argument('--compare')
    ap.add_argument('--pair', action='append', default=[])
    ap.add_argument('--grid-from', choices=('stem', 'mix'), default='stem',
                    help="with --grid: the backing's grid (default) or the mix's, when take_grid flags a tempo disagreement")
    ap.add_argument('--pin', action='append', default=[],
                    help='when measuring the reference: SECTION="C G Am F", the progression the person confirmed; stored as chords_pinned')
    ap.add_argument('--ref-chords', action='append', default=[],
                    help='with --compare: SECTION="C G Am F" (or just "C G Am F" for every section) pins the reference progression')
    a = ap.parse_args()
    me = measure(a)
    for s in me['sections']:
        if 'skipped' in s:
            print(f"== {s['name']} {s['start']}-{s['end']} s: skipped ({s['skipped']})")
            continue
        g = s['groove']
        print(f"== {s['name']} {s['start']}-{s['end']} s  {s['bpm']} BPM, {s['bars']} bars from {s['first_bar_line']} s"
              + (f" (downbeat ratio {s['downbeat_ratio']})" if s.get('downbeat_ratio') else '')
              + f", key {s['key']} from the {s['key_from']} (profile fit {s['key_profile']}, next {s['key_profile_next']} +{s['key_profile_gap']})")
        print('   chords:', ' '.join(s['chords']), f"(median fit {s['chord_fit_median']})")
        print('   roman: ', ' '.join(s['roman']))
        if s.get('progression'):
            pr = s['progression']
            print(f"   progression: {pr['name']} on {pr['tonic']} ({'-'.join(pr['chords'])}), {pr['changes_matched']:.0%} of changes, "
                  f"{pr['bars_inside']:.0%} of bars")
        print('   closest named: ' + '; '.join(f"{c['name']} on {c['tonic']} {c['changes_matched']:.0%}/{c['bars_inside']:.0%}"
                                             for c in s['progression_candidates']) + '  (changes/bars)')
        if s.get('chords_pinned'):
            print(f"   pinned by the person: {s['chords_pinned']}")
        if s.get('degrees'):
            print('   voice degrees:', ' '.join(f'{k}:{v:.2f}' for k, v in sorted(s['degrees'].items(), key=lambda kv: -kv[1])[:6]),
                  f"({s['voiced_seconds']} s voiced)")
            if s.get('degrees_over_V'):
                print('   over V:      ', ' '.join(f'{k}:{v:.2f}' for k, v in sorted(s['degrees_over_V'].items(), key=lambda kv: -kv[1])[:6]))
            print('   colour:', ', '.join(s['colour']) or '-')
        print(f"   groove ({s['groove_layer']}) 8ths: {' '.join(f'{v:.2f}' for v in g['profile_8ths'])}  kick {' '.join(f'{v:.2f}' for v in g['kick_band'])}"
              f"  snare {' '.join(f'{v:.2f}' for v in g['snare_band'])}")
        print('   groove, fixed phase:', g['scores'], '->', ', '.join(g['names']) or '-', ('; ' + g['note']) if g.get('note') else '')
        print('   groove, phase-free: ', g['scores_phase_free'], '->', ', '.join(g['names_phase_free']) or '-')
        for w in s['words']:
            print('   卡片：' + w)
    if a.compare:
        ref = json.load(open(a.compare, encoding='utf-8'))
        pairs = dict(p.split('=') for p in a.pair)
        ref_pins = dict(p.split('=', 1) if '=' in p else ('*', p) for p in a.ref_chords)
        me['comparison'] = dict(reference=a.compare, **compare(me, ref, pairs, ref_pins))
    Path(a.out + '.json').write_text(json.dumps(me, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'-> {a.out}.json')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
