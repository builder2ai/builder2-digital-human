"""Put the finished song together on bar lines: an intro, the take's own accompaniment and its (comped) vocal, an ending,
and a spoken line with a short dip in the accompaniment under it. Then the delivery checks and the loudness rule.

Everything sits on the take's clock: take bar k = the take's downbeat + k bars (take_grid.py's grid). The song is the take
moved by one fixed shift (song seconds = take seconds + shift), so what is not changed stays the take's audio, sample for
sample. Song bar 0 = the first intro bar.
- intro: own = the take's own last --intro-bars bars before bar 0 (no join: check it first, YuE2 takes often have one), or
  an ACE-Step repaint (a file, or an id with --ace-spec) whose new bars come before a context cut from the take's
  accompaniment at take bar --intro-ctx-bar. The new bars count from the spec (-repainting_start) or --intro-new-bars. The
  switch to the take's own accompaniment is --intro-switch bars into the context (0.5), after checking that the repaint's
  context is the take's audio (correlation >= .98; least-squares gain).
- the take's accompaniment (its instrumental stem) and its vocal (--vocal: a comp from comp_vocal.py; default the take's
  vocal stem), the vocal only from --voice-from to --voice-to (take seconds; default from --sections: the first lyric line
  - 0.5 s to the last + 0.7 s) so a lead-in hum and the ad-libs after the lyric stay out.
- ending: an ACE-Step repaint whose context starts at take bar --ending-ctx-bar and whose new music starts at
  --ending-new-bar (or from the spec: ctx bar + repainting_start); the switch is at --ending-switch-bar (one bar before the
  new music), where both are the same audio; the tail is trimmed at -50 dBFS with a 0.5 s fade. Or none: the take runs to
  --end-bar (default the bar line after the last lyric line + 1 bar), faded over its last bar.
- spoken line (made in step 6 with a TTS model, mixed here because it often sits on the last note, which needs the
  ending): leading and trailing silence cut, level = the vocal's active level over --spoken-ref (default the 10 s before
  it) + --spoken-db, a little room (--room wet; 0 = dry), placed at --spoken-at (take seconds). Only the accompaniment
  dips, --duck dB (4) with --duck-ramp (80 ms) ramps, --duck-pad (0.1 s) either side of the line. In one test: speech 4.9
  dB over the accompaniment under it, Whisper heard the line in 3 of 3 windows of the mix, no quiet half bar. A whole
  section dipped 7.5 dB left a hole of about 10 dB that the listener heard at once.
- loudness: --lufs (-16) and --tp (-1.5 dBTP). The true peak wins: the gain is the smaller of the two, and when the peak
  binds, it says how far under the target the song stays and where the peak is. --limiter (optional, labelled in the
  record): gain to the target, then ffmpeg's alimiter at 4x oversampling (5 ms lookahead, 50 ms release, delay
  compensated) with its ceiling under --tp, then the true peak is measured again and any excess taken off by gain. It
  prints how much the limiter took off and for how long. Nothing is ever clipped: a sample at full scale stops the run.
Checks (in <out>.json): the same-audio correlation at each switch; beat_check.py at the intro join and into the ending;
which bars differ from the take's own mix, between the intro's context and the ending switch (the comp patches: a
listening list); quiet half bars against the take's own bar at the same place (bar_levels.quiet: a written break is not
a hole; an ending's die-away after its first bar is not checked); the largest sample jumps; the spoken line's level
against the accompaniment under it; LUFS and true peak of the wav (and mp3).
Writes <out>.wav (48 kHz 24-bit), <out>_voice.wav (sung + spoken, same gain), <out>_bed.wav (accompaniment after the dip),
<out>.json (the record: times on the song clock), <out>_map.json (a measured song map), and <out>.mp3 with --mp3 192.
Options can also come from an "assembly" block in a song_map.json (--map; the keys are the option names, paths relative to
that file); options given on the command line win.
numpy and ffmpeg only.
usage: assemble.py --take takes/X.flac --stems stems --grid takes.json --sections sections.json --out final/song
                   [--vocal comp/fixed_vocals.wav] [--intro own --intro-bars 4 | --intro intro_s3 --ace-spec ace/spec.json
                   --intro-ctx-bar 0] [--ending end_s2 --ending-ctx-bar 4 [--ending-new-bar 12] | --ending none --end-bar 13]
                   [--spoken tts/line.wav --spoken-at 39.15 --duck 4 --room .12] [--lufs -16 --tp -1.5 [--limiter]] [--mp3 192]
                   [--bpm 104.06 --downbeat 11.764] [--map song_map.json]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import bar_levels  # noqa: E402
import beat_check  # noqa: E402
from cut_take import grid_of, loudness as file_loudness  # noqa: E402
from songmap import SR, db, load, seg, write  # noqa: E402

SWITCH = .06          # seconds of crossfade at a switch; both sides hold the same audio there
END_FLOOR_DB = -50.   # an ending's tail is trimmed here
ROOM_SECONDS = 1.6
PATH_KEYS = {'take', 'stems', 'bed', 'vocal', 'grid', 'sections', 'ace_spec', 'spoken', 'out'}


def options():
    ap = argparse.ArgumentParser(description='Assemble the song on bar lines; see the module docstring.')
    ap.add_argument('--map', help='a song_map.json with an "assembly" block of these options')
    ap.add_argument('--take', help="the take's mix (its name finds the grid row, the sections row and the stems)")
    ap.add_argument('--stems', help='folder with <take>_instrumental.flac and <take>_vocals.flac')
    ap.add_argument('--bed', help='the accompaniment, if not the stems folder one')
    ap.add_argument('--vocal', help="the vocal (a comp); default the take's vocal stem")
    ap.add_argument('--grid', help="take_grid.py's takes.json")
    ap.add_argument('--grid-from', choices=('stem', 'mix'), default='stem')
    ap.add_argument('--bpm', type=float)
    ap.add_argument('--downbeat', type=float, help="a bar line of the take (take bar 0), seconds")
    ap.add_argument('--beats-per-bar', type=int, default=4)
    ap.add_argument('--sections', help="take_sections.py's sections.json")
    ap.add_argument('--ace-spec', help="ace_step_run.py's spec: intro and ending ids are found in its raw/")
    ap.add_argument('--intro', default='own', help='own, or an ACE-Step repaint (file or id)')
    ap.add_argument('--intro-bars', type=float, default=4.)
    ap.add_argument('--intro-ctx-bar', type=float, default=0.)
    ap.add_argument('--intro-new-bars', type=float)
    ap.add_argument('--intro-switch', type=float, default=.5)
    ap.add_argument('--ending', default='none', help='none, or an ACE-Step repaint (file or id)')
    ap.add_argument('--ending-ctx-bar', type=float)
    ap.add_argument('--ending-new-bar', type=float)
    ap.add_argument('--ending-switch-bar', type=float)
    ap.add_argument('--end-bar', type=float)
    ap.add_argument('--voice-from', type=float)
    ap.add_argument('--voice-to', type=float)
    ap.add_argument('--spoken')
    ap.add_argument('--spoken-at', type=float)
    ap.add_argument('--spoken-db', type=float, default=0.)
    ap.add_argument('--spoken-ref', help='take seconds A-B of the vocal to match the speech level to')
    ap.add_argument('--duck', type=float, default=4.)
    ap.add_argument('--duck-ramp', type=float, default=.08)
    ap.add_argument('--duck-pad', type=float, default=.1)
    ap.add_argument('--room', type=float, default=.12)
    ap.add_argument('--lufs', type=float, default=-16.)
    ap.add_argument('--tp', type=float, default=-1.5)
    ap.add_argument('--limiter', action='store_true')
    ap.add_argument('--mp3', type=int, help='also write an mp3 at this many kb/s')
    ap.add_argument('--out')
    first, _ = ap.parse_known_args()
    if first.map:
        mp = Path(first.map).resolve()
        plan = json.loads(mp.read_text(encoding='utf-8')).get('assembly') or {}
        dests = {x.dest for x in ap._actions}
        d = {}
        for k, v in plan.items():
            k = k.replace('-', '_')
            if k not in dests:
                raise SystemExit(f'unknown key "{k}" in the assembly block of {mp}')
            ace_file = k in ('intro', 'ending') and str(v).lower().endswith(('.wav', '.flac'))
            if v is not None and (k in PATH_KEYS or ace_file) and not Path(str(v)).is_absolute():
                v = str(mp.parent / v)
            d[k] = v
        ap.set_defaults(**d)
    a = ap.parse_args()
    for k in ('take', 'out'):
        if not getattr(a, k):
            ap.error(f'--{k} is required (or "{k}" in the assembly block)')
    return a


def same_audio(x, y, what):
    """The gain that makes y into x, after making sure they are the same audio on one clock."""
    n = min(len(x), len(y))
    g = float((x[:n] * y[:n]).sum() / ((y[:n] ** 2).sum() + 1e-12))
    match = float(np.corrcoef(x[:n].ravel(), y[:n].ravel())[0, 1])
    print(f'{what}: gain {g:.4f}, match {match:.4f}', flush=True)
    if match < .98:
        raise SystemExit(f'{what}: not the same audio on one clock (match {match:.3f}): check the context bar and the grid')
    return g, match


def spliced(x, y, at):
    """x up to sample `at`, then y, over a short crossfade; both on one clock."""
    n = int(SWITCH * SR)
    ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    out = y.copy()
    out[:at] = x[:at]
    k = min(n, len(out) - at)
    out[at:at + k] = x[at:at + k] * (1 - ramp[:k]) + y[at:at + k] * ramp[:k]
    return out


def trimmed(x):
    """x without the dead air at its end (under END_FLOOR_DB), with a half-second fade."""
    hop = SR // 20
    lv = np.sqrt((x[:len(x) // hop * hop].reshape(-1, hop, x.shape[1]) ** 2).mean(axis=(1, 2)))
    loud = np.where(lv > 10 ** (END_FLOOR_DB / 20))[0]
    x = x[:min(len(x), (int(loud[-1]) + 1) * hop)].copy() if len(loud) else x.copy()
    fade = min(len(x), int(.5 * SR))
    x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return x


def room(x, wet, seed=11):
    """A soft decaying tail for x (filtered noise, low in the top and the bottom), returned on its own, len(x) + 1.6 s."""
    rng = np.random.default_rng(seed)
    n = int(ROOM_SECONDS * SR)
    decay = np.exp(-6.9 * np.arange(n) / n)
    f = np.fft.rfftfreq(n, 1 / SR)
    shape = np.clip((f - 150) / 150, 0, 1) * np.clip((6000 - f) / 3000, 0, 1)
    tail = np.stack([np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n=n) * decay for _ in range(2)], axis=1)
    tail /= np.sqrt((tail ** 2).sum(axis=0))
    size = len(x) + n
    return (np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(tail, size, axis=0), n=size, axis=0) * wet).astype(np.float32)


def active_rms(x):
    """RMS over the 50-ms frames louder than a quarter of the loudest: the level of a voice while it sounds."""
    m = x.mean(axis=1)
    n = SR // 20
    f = np.array([np.sqrt((m[i:i + n] ** 2).mean()) for i in range(0, max(1, len(m) - n), n)])
    f = f[f > .25 * f.max()] if len(f) and f.max() > 0 else np.array([1e-9])
    return float(np.sqrt((f ** 2).mean()))


def meter(x):
    """(integrated LUFS, true peak dBTP) of a float array, by ffmpeg's loudnorm analysis."""
    r = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-af',
                        'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
                       input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True, check=True).stderr.decode(errors='replace')
    m = json.loads(r[r.rfind('{'):r.rfind('}') + 1])
    return float(m['input_i']), float(m['input_tp'])


def limited(x, ceiling_db):
    """x through ffmpeg's alimiter at 192 kHz (4x oversampled, so it catches most inter-sample peaks), 5 ms lookahead,
    50 ms release, no auto level, delay compensated; back at 48 kHz, same length."""
    af = (f'aresample=192000,alimiter=limit={10 ** (ceiling_db / 20):.5f}:attack=5:release=50:level=0:latency=1,'
          f'aresample={SR}')
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-af', af,
                          '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'],
                         input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    out = np.zeros_like(x)
    out[:min(len(x), len(y))] = y[:len(x)]
    return out


def main():
    a = options()
    bpb = a.beats_per_bar
    take_id = Path(a.take).stem
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    notes = []

    # grid of the take
    if a.bpm and a.downbeat is not None:
        BAR, D0, BPM = 60. / a.bpm * bpb, float(a.downbeat), a.bpm
    elif a.grid:
        row = {r['id']: r for r in json.loads(Path(a.grid).read_text(encoding='utf-8'))}.get(take_id)
        if not row:
            raise SystemExit(f'{take_id} is not in {a.grid}')
        BAR, D0 = grid_of(row, a.grid_from)
        BPM = (row.get('grid_mix') or {}).get('bpm') if a.grid_from == 'mix' else row.get('bpm')
        if row.get('tempo_agree') is False and a.grid_from == 'stem':
            notes.append('take_grid flags a tempo disagreement between the backing and the mix: check --grid-from')
    else:
        raise SystemExit('give --grid takes.json, or --bpm and --downbeat (the take\'s grid)')
    take_bar = lambda k: D0 + k * BAR

    sec = None
    if a.sections:
        sec = {r['id']: r for r in json.loads(Path(a.sections).read_text(encoding='utf-8'))}.get(take_id)
        if not sec or sec.get('sung_from') is None:
            raise SystemExit(f'no lyric found for {take_id} in {a.sections}')

    stems = Path(a.stems) if a.stems else None
    bed_path = Path(a.bed) if a.bed else (stems / f'{take_id}_instrumental.flac' if stems else None)
    vocal_path = Path(a.vocal) if a.vocal else (stems / f'{take_id}_vocals.flac' if stems else None)
    if not bed_path or not vocal_path:
        raise SystemExit('give --stems (or --bed and --vocal)')
    mix_take, inst, vocal = load(a.take), load(bed_path), load(vocal_path)

    spec = {}
    if a.ace_spec:
        spec = {c['id']: c for c in json.loads(Path(a.ace_spec).read_text(encoding='utf-8'))['candidates']}

    def ace(name):
        if name in spec:
            return Path(a.ace_spec).parent / 'raw' / f'{name}.wav', spec[name]
        return Path(name), spec.get(Path(name).stem)

    # intro: where the song starts on the take's clock
    if a.intro == 'own':
        N, C = float(a.intro_bars), 0.
        ipath = None
    else:
        ipath, ic = ace(a.intro)
        if a.intro_new_bars is not None:
            N = float(a.intro_new_bars)
        elif ic and float(ic['repainting_start']) < 0:
            N = -float(ic['repainting_start']) / BAR
        else:
            raise SystemExit('how many new intro bars? give --intro-new-bars, or --ace-spec with the candidate')
        if abs(N - round(N)) > .02:
            notes.append(f'the intro has {N:.2f} bars, not whole bars: its context may not start on a bar line')
        C = float(a.intro_ctx_bar)
    start_take = take_bar(C - N)
    shift = -start_take
    song = lambda t: t + shift
    if a.intro == 'own' and start_take < -1e-3:
        raise SystemExit(f'the take has no {N:g} bars before bar 0 (that would start at {start_take:.2f} s): fewer --intro-bars')

    # voice span
    vf, vt = a.voice_from, a.voice_to
    if sec:
        vf = sec['sung_from'] - .5 if vf is None else vf
        vt = sec['sung_to'] + .7 if vt is None else vt
    if vf is None or vt is None:
        notes.append('no --sections or --voice-from/--voice-to: the whole vocal is used, with any hum before the lyric and ad-libs after it')
    vf = max(vf if vf is not None else start_take, start_take, 0.)

    # ending
    if a.ending == 'none':
        if a.end_bar is not None:
            EB = float(a.end_bar)
        elif sec:
            EB = math.ceil((sec['sung_to'] - .1 - D0) / BAR) + 1
        else:
            raise SystemExit('with --ending none, give --end-bar (a take bar) or --sections')
        tail, SW, song_end = None, EB, song(take_bar(EB))
        if take_bar(EB) > len(inst) / SR + .01:
            raise SystemExit(f'the take ends at {len(inst) / SR:.2f} s, before take bar {EB:g} ({take_bar(EB):.2f} s)')
    else:
        epath, ec = ace(a.ending)
        if a.ending_ctx_bar is None:
            raise SystemExit('give --ending-ctx-bar: the take bar where the ending\'s context audio starts')
        E0 = float(a.ending_ctx_bar)
        if a.ending_new_bar is not None:
            EB = float(a.ending_new_bar)
        elif ec:
            EB = E0 + float(ec['repainting_start']) / BAR
        else:
            raise SystemExit('give --ending-new-bar, or --ace-spec with the candidate')
        SW = float(a.ending_switch_bar) if a.ending_switch_bar is not None else EB - 1
        if SW < E0 + .5:
            raise SystemExit('the ending switch must sit at least half a bar into its context')
        end_take = load(epath)
        ctx_at = lambda k: (k - E0) * BAR
        k0 = max(E0 + .1, EB - 3)
        g_end, m_end = same_audio(seg(inst, take_bar(k0), take_bar(EB - .1)), seg(end_take, ctx_at(k0), ctx_at(EB - .1)),
                                  f'ending against the take (take bars {k0:g}-{EB - .1:g})')
        tail = trimmed(seg(end_take, ctx_at(SW), len(end_take) / SR) * g_end)
        song_end = song(take_bar(SW)) + len(tail) / SR
    total = int(round(song_end * SR))
    sw_end = int(round(song(take_bar(SW)) * SR))
    checks = {}

    # accompaniment: the take's own, on the song clock
    body = np.zeros((total, 2), np.float32)
    i0 = int(round(start_take * SR))
    a0 = max(0, -i0)
    n = max(0, min(len(inst) - max(0, i0), (sw_end + int(SWITCH * SR) if tail is not None else total) - a0))
    body[a0:a0 + n] = inst[max(0, i0):max(0, i0) + n]
    if ipath is None:
        bed = body
        checks['intro'] = {'take': f'own: take bars {C - N:g}..{C:g} ({start_take:.3f}-{take_bar(C):.3f} s), no join'}
    else:
        it = load(ipath)
        ctx_len = min(len(it) / SR - N * BAR - .2, 6 * BAR)
        g_in, m_in = same_audio(seg(inst, take_bar(C) + .1, take_bar(C) + ctx_len), seg(it, N * BAR + .1, N * BAR + ctx_len),
                                f'intro against the take (from take bar {C:g})')
        head = np.zeros_like(body)
        k = min(len(it), total)
        head[:k] = it[:k] * g_in
        sw_in = int(round((N + a.intro_switch) * BAR * SR))
        bed = spliced(head, body, sw_in)
        checks['intro'] = {'take': str(ipath), 'new_bars': round(N, 2), 'ctx_take_bar': C, 'gain': round(g_in, 4), 'match': round(m_in, 4),
                           'switch_song_s': round(sw_in / SR, 3)}
    fi = int(.03 * SR)
    bed[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    if tail is not None:
        holder = np.zeros_like(bed)
        holder[sw_end:sw_end + len(tail)] = tail[:total - sw_end]
        bed = spliced(bed, holder, sw_end)
        checks['ending'] = {'take': str(epath), 'ctx_take_bar': E0, 'new_music_take_bar': round(EB, 2), 'switch_take_bar': SW,
                            'gain': round(g_end, 4), 'match': round(m_end, 4), 'switch_song_s': round(sw_end / SR, 3),
                            'new_music_song_s': round(song(take_bar(EB)), 3), 'tail_after_new_music_s': round(song_end - song(take_bar(EB)), 2)}
    else:
        k = min(total, int(round(BAR * SR)))
        bed[-k:] *= (np.linspace(1, 0, k) ** 1.5).astype(np.float32)[:, None]
        checks['ending'] = {'take': f'none: the take to take bar {EB:g}, faded over its last bar'}

    # voice, on the take's clock
    v = np.zeros_like(bed)
    piece = seg(vocal, vf, vt if vt is not None else len(vocal) / SR).copy()
    f_in, f_out = min(len(piece), int(.2 * SR)), min(len(piece), int(.3 * SR))
    piece[:f_in] *= np.linspace(0, 1, f_in, dtype=np.float32)[:, None]
    piece[-f_out:] *= np.linspace(1, 0, f_out, dtype=np.float32)[:, None]
    s0 = int(round(song(vf) * SR))
    k = max(0, min(len(piece), total - s0))
    v[s0:s0 + k] = piece[:k]

    # spoken line: level matched to the vocal, room, and a short dip in the accompaniment only
    spoken = np.zeros_like(bed)
    curve = np.ones(total, np.float32)
    if a.spoken:
        if a.spoken_at is None:
            raise SystemExit('give --spoken-at (take seconds where the line starts)')
        sp = load(a.spoken)
        e = np.sqrt((sp.mean(axis=1)[:len(sp) // 480 * 480].reshape(-1, 480) ** 2).mean(axis=1))
        on = np.where(e > .05 * e.max())[0]
        sp = sp[max(0, on[0] - 2) * 480:(on[-1] + 3) * 480]
        ra, rb = (float(x) for x in a.spoken_ref.split('-')) if a.spoken_ref else (a.spoken_at - 10, a.spoken_at)
        g_sp = active_rms(seg(vocal, ra, rb)) / active_rms(sp) * 10 ** (a.spoken_db / 20)
        dry = (sp * g_sp).astype(np.float32)
        dur = len(dry) / SR
        line = np.concatenate([dry, np.zeros((int(ROOM_SECONDS * SR), 2), np.float32)])
        if a.room > 0:
            line += room(dry, a.room)[:len(line)]
        t_sp = song(a.spoken_at)
        p0 = int(round(t_sp * SR))
        k = max(0, min(len(line), total - p0))
        spoken[p0:p0 + k] = line[:k]
        lo = 10 ** (-a.duck / 20)
        pts = [(0, 1.), (t_sp - a.duck_pad - a.duck_ramp, 1.), (t_sp - a.duck_pad, lo), (t_sp + dur + a.duck_pad, lo),
               (t_sp + dur + a.duck_pad + a.duck_ramp, 1.), (total / SR, 1.)]
        curve = np.interp(np.arange(total) / SR, *zip(*pts)).astype(np.float32)
        if p0 + len(dry) > total:
            notes.append('the spoken line runs past the end of the song')
    bed_d = bed * curve[:, None]
    music = bed_d + v + spoken

    # loudness: the true peak wins
    li, tp = meter(music)
    g_lufs, g_tp = a.lufs - li, a.tp - .1 - tp
    lim = None
    if a.limiter and g_lufs > g_tp:
        pre = music * 10 ** (g_lufs / 20)
        ceiling = a.tp - .5
        final = limited(pre, ceiling)
        li2, tp2 = meter(final)
        extra = min(0., a.tp - .1 - tp2)
        final *= 10 ** (extra / 20)
        thr = 10 ** (ceiling / 20)
        acted = np.abs(pre).max(axis=1) > thr
        hop = SR // 100
        frames = acted[:len(acted) // hop * hop].reshape(-1, hop).any(axis=1)
        G = 10 ** ((g_lufs + extra) / 20)
        lim = {'used': True, 'ceiling_dbfs': ceiling, 'gain_into_it_db': round(g_lufs, 2), 'extra_gain_after_db': round(extra, 2),
               'acted_seconds': round(float(frames.sum()) / 100, 2), 'most_taken_off_db': round(20 * math.log10(float(np.abs(pre).max()) / thr), 2)
               if np.abs(pre).max() > thr else 0.}
        print(f"limiter: {lim['most_taken_off_db']} dB off the loudest peak, acting on {lim['acted_seconds']} s in all", flush=True)
        bed_out, voice_out = bed_d * G, (v + spoken) * G     # the stems carry the gain, not the limiting
    else:
        gain = min(g_lufs, g_tp)
        final = music * 10 ** (gain / 20)
        G = 10 ** (gain / 20)
        bed_out, voice_out = bed_d * G, (v + spoken) * G
        if g_tp < g_lufs:
            at = int(np.argmax(np.abs(music).max(axis=1)))
            t = at / SR
            where = ('the new intro bars' if ipath is not None and t < N * BAR else
                     'the new ending' if tail is not None and t >= song(take_bar(EB)) else
                     'the spoken line' if a.spoken and t_sp <= t <= t_sp + dur else "the take's own audio")
            notes.append(f'true peak wins: {a.lufs - (g_lufs - g_tp):.1f} LUFS, {g_lufs - g_tp:.1f} LU under {a.lufs:g}; the peak is at '
                         f'{t:.2f} s (song) in {where}. Deliver as it is, or run again with --limiter')
    if np.abs(final).max() >= 1.:
        raise SystemExit(f'a sample reached full scale ({np.abs(final).max():.3f}): nothing written (no clipping)')
    write(final, out.with_suffix('.wav'))
    write(voice_out, out.parent / f'{out.name}_voice.wav')
    write(bed_out, out.parent / f'{out.name}_bed.wav')
    wav_i, wav_tp = file_loudness(out.with_suffix('.wav'))
    loud = {'target_lufs': a.lufs, 'max_true_peak_dbtp': a.tp, 'input_lufs': round(li, 2), 'input_tp': round(tp, 2),
            'gain_db': round(20 * math.log10(G), 2), 'limiter': lim or {'used': False}, 'wav': {'lufs': wav_i, 'true_peak_dbtp': wav_tp},
            'sample_peak': round(float(np.abs(final).max()), 3)}
    if a.mp3:
        mp3 = out.with_suffix('.mp3')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(out.with_suffix('.wav')), '-c:a', 'libmp3lame', '-b:a', f'{a.mp3}k', str(mp3)],
                       check=True, stdin=subprocess.DEVNULL)
        mi, mtp = file_loudness(mp3)
        loud[f'mp3_{a.mp3}k'] = {'lufs': mi, 'true_peak_dbtp': mtp}
        if mtp is not None and mtp > a.tp:
            notes.append(f'the mp3 peaks at {mtp} dBTP, over {a.tp:g}: encoding raised it; run again with a lower --tp')
    if wav_tp is not None and wav_tp > a.tp + .05:
        notes.append(f'the wav peaks at {wav_tp} dBTP, over {a.tp:g}')

    # checks
    if ipath is not None:
        bc = beat_check.check(bed, N * BAR, BAR, new='before', before_bars=N, after_bars=4, beats_per_bar=bpb)
        checks['intro']['beat'] = {k: bc[k] for k in ('verdict', 'jump_8th_ms', 'accent_shift_8ths', 'accent_corr_at_0', 'accent_corr_best', 'notes')}
        print('intro join:', beat_check.line('intro', bc), flush=True)
    if tail is not None:
        jn = song(take_bar(EB))
        bc = beat_check.check(bed, jn, BAR, new='after', ending=True, before_bars=min(4., EB - (C - N)),
                              after_bars=max(.5, min(4., (song_end - jn) / BAR)), beats_per_bar=bpb)
        checks['ending']['beat'] = {k: bc.get(k) for k in ('verdict', 'first_strong_onset_ms', 'new_keeps_groove', 'jump_8th_ms', 'accent_shift_8ths', 'notes')}
        print('into the ending:', beat_check.line('ending', bc), flush=True)
    # what differs from the take's own mix, per 2 bars (comp patches, the dip): where to listen
    ref = np.zeros_like(music)
    r0 = int(round(start_take * SR))
    n = max(0, min(len(mix_take) - max(0, r0), total - max(0, -r0)))
    ref[max(0, -r0):max(0, -r0) + n] = mix_take[max(0, r0):max(0, r0) + n]
    lo_s = N * BAR + .1 if ipath is not None else .1        # an intro's context is the take's audio (checked above)
    hi_s = sw_end / SR if tail is not None else total / SR - BAR
    blocks = []
    for b in np.arange(math.floor(lo_s / BAR), hi_s / BAR - 1e-6, 1):
        t0, t1 = max(lo_s, b * BAR), min(hi_s, (b + 1) * BAR)
        x, y = seg(music, t0, t1), seg(ref, t0, t1)
        if len(x) > SR // 4 and np.abs(y).max() > 0:
            blocks.append((int(b), t0, t1, float(np.corrcoef(x.ravel(), y.ravel())[0, 1])))
    spans = []
    for b, t0, t1, c in blocks:
        if c >= .999:
            continue
        if spans and spans[-1]['song_bars'][1] == b:
            spans[-1]['song_bars'][1], spans[-1]['song_s'][1] = b + 1, round(t1, 2)
            spans[-1]['lowest_corr'] = round(min(spans[-1]['lowest_corr'], c), 4)
        else:
            spans.append({'song_bars': [b, b + 1], 'song_s': [round(t0, 2), round(t1, 2)], 'lowest_corr': round(c, 4)})
    checks['take_kept'] = {'bars_checked': len(blocks), 'from_song_s': round(lo_s, 2), 'to_song_s': round(hi_s, 2), 'differs': spans}
    # quiet half bars, against the take's own bar at the same place; an ending's die-away and a fade-out are meant
    stop = song(take_bar(EB)) + BAR if tail is not None else total / SR - BAR
    rows = [r for r in bar_levels.levels(final, BAR, 0., BAR / 2, 0, int(total / SR / BAR * 2) / 2) if r[1] + BAR / 2 <= stop + 1e-6]
    checks['quiet_half_bars'] = bar_levels.quiet(rows, BAR / 2, mix_take, shift)
    jump = np.abs(np.diff(final.mean(axis=1)))
    near = {}
    for name, s in (('intro switch', checks['intro'].get('switch_song_s')), ('ending switch', checks['ending'].get('switch_song_s'))):
        if s is not None:
            j0, j1 = int((s - .1) * SR), int((s + .1) * SR)
            near[name] = round(float(jump[j0:j1].max()), 3)
    checks['sample_jumps'] = {'largest_at_s': sorted(round(float(k) / SR, 2) for k in np.argsort(jump)[-5:]), 'largest': round(float(jump.max()), 3),
                              'largest_near_switches': near}
    if a.spoken:
        under = slice(int((t_sp + .1) * SR), int((t_sp + dur - .1) * SR))
        sp_db, bed_db = db(spoken[under] * G), db(bed_d[under] * G)
        checks['spoken'] = {'file': a.spoken, 'at_take_s': a.spoken_at, 'song_s': [round(t_sp, 2), round(t_sp + dur, 2)], 'seconds': round(dur, 2),
                            'gain_db': round(20 * math.log10(g_sp), 1), 'level_ref_take_s': [round(ra, 2), round(rb, 2)], 'duck_db': a.duck,
                            'duck_ramp_s': a.duck_ramp, 'room_wet': a.room, 'spoken_db': sp_db, 'bed_under_it_db': bed_db,
                            'bed_without_duck_db': db(bed[under] * G), 'voice_under_it_db': db(v[under] * G), 'speech_over_bed_db': round(sp_db - bed_db, 1)}
        print(f"spoken line at song {t_sp:.2f}-{t_sp + dur:.2f} s: {sp_db} dB, the accompaniment under it {bed_db} dB "
              f"({checks['spoken']['bed_without_duck_db']} without the dip): check it with spoken_qc.py --mix", flush=True)

    # record and measured song map
    def song_bar(t_take, up=False):
        """Song bar of a section's first word (a pickup of up to half a bar belongs to the next bar) or last word (up)."""
        k = (t_take - D0) / BAR
        b = (math.ceil(k - .1) if up else math.floor(k + .5)) + (N - C)
        return int(b) if abs(b - round(b)) < 1e-9 else round(b, 2)
    sections = []
    whole = lambda x: int(round(x)) if abs(x - round(x)) < 1e-6 else round(x, 2)
    if N > 0:
        sections.append({'name': 'intro', 'start_bar': 0, 'bars': whole(N), 'manner': 'instrumental',
                         'made_by': 'the take\'s own bars' if ipath is None else f'ACE-Step repaint {ipath.name}'})
    if sec:
        for s in sec.get('sections', []):
            if s.get('start') is None:
                continue
            b0, b1 = song_bar(s['start']), song_bar(s['end'], up=True)
            sections.append({'name': s['name'], 'start_bar': b0, 'bars': b1 - b0, 'lines': s.get('lines'),
                             'first_word_s': round(song(s['start']), 2), 'last_word_end_s': round(song(s['end']), 2)})
    else:
        sections.append({'name': 'take', 'start_bar': whole(N), 'bars': whole(EB - C)})
    end_bar0 = whole(EB + N - C)
    if tail is not None:
        sections.append({'name': 'ending', 'start_bar': end_bar0, 'bars': round((song_end - song(take_bar(EB))) / BAR, 2),
                         'made_by': f'ACE-Step repaint {Path(epath).name}'})
    plan = json.loads(Path(a.map).read_text(encoding='utf-8')) if a.map else {}
    smap = {'bpm': BPM or round(60 * bpb / BAR, 2), 'beats_per_bar': bpb, 'downbeat': 0.0, 'measured': True, 'key': plan.get('key', ''),
            'target_seconds': plan.get('target_seconds'), 'note': f'measured by assemble.py for {out.name}.wav; song seconds = take seconds + {shift:.3f}',
            'sections': sections}
    (out.parent / f'{out.name}_map.json').write_text(json.dumps(smap, ensure_ascii=False, indent=1), encoding='utf-8')
    rec = {'name': out.name, 'seconds': round(total / SR, 2), 'take': str(a.take), 'vocal': str(vocal_path), 'bed': str(bed_path),
           'grid': {'bar_s': BAR, 'take_downbeat_s': D0, 'song_bar0_take_bar': round(C - N, 2), 'take_shift_s': round(shift, 3)},
           'voice_take_s': [round(vf, 2), round(vt, 2) if vt is not None else None], 'loudness': loud, 'checks': checks, 'notes': notes}
    (out.parent / f'{out.name}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'seconds': rec['seconds'], 'loudness': loud, 'notes': notes}, ensure_ascii=False), flush=True)
    for q in checks['quiet_half_bars']:
        print(f"quiet half bar {q['bar']} at {q['s']} s: {q['under_median_db']} dB under the median; {q.get('verdict', '')}")
    if checks['take_kept']['differs']:
        print('differs from the take (listen here):', ', '.join(f"{d['song_s'][0]}-{d['song_s'][1]} s (r {d['lowest_corr']})"
                                                               for d in checks['take_kept']['differs']))


if __name__ == '__main__':
    main()
