"""EXAMPLE from our song, not a general tool: the bar numbers, section names, file layout and gains below are that song's.
Read it for the method; run it only after adapting it. Files are found under SONG_DIR (default: the current folder), with
the ACE-Step takes and specs in SONG_DIR/ace_beds.

Give a draft that ends on his third pass an opening and an ending from ACE-Step, written next to the song's own accompaniment (ace_beds/spec_p2.json). Voice and every other bar stay as they are.

Opening: the take's new bars replace the two-bar prelude and everything after moves later by the difference. Her first word stays where it falls against the bar line.
Ending: the take's bars carry the third pass's own accompaniment on past its last bar. The bars the take shares with the draft are the same audio, so the switch sits well before the new music.
With quiet, his last two lines are nearly unaccompanied: the take is split into parts (ace_beds/stems), the drums leave in the breath before those lines, the rest drops far under his voice and the closing phrase stays quiet after his last word. His voice gets a little room from there, so its end does not stop dead.
An ending take may also have rewritten the bars under his last two lines (spec_p4.json: an erhu line there, the ending take's bars after them kept); its lead part then stays audible under his voice.
An ending take may also be written in one piece from before his last two lines to the end (spec_p5.json: the band drops out by itself); without quiet it is used as it is, and room gives his voice the same tail as quiet does.
With rit, his last two lines slow down bit by bit (the listener: 「最后那两句 应该慢慢慢下来」), from the bar where the
band has dropped out: the accompaniment by ffmpeg's rubberband, his dry voice by WSOLA that holds vowels and breaths longer
and keeps consonants at speed (ritard.py), and the room goes on afterwards. Stretching the finished mix in one go
sounded 「像是磁带坏了」.
Usage: grow_ends.py <assembled draft> <opening take> <ending take> <new name> [quiet] [room] [rit]
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ritard  # noqa: E402
from songmap import SR, load, load_map, seg  # noqa: E402

D_BAR = load_map().bar_seconds
D = Path(os.environ.get('SONG_DIR', '.')).resolve()
S = D / 'stems'
A = D / 'ace_beds'
OLD_PRELUDE_BARS = 2
SWITCH = .06          # seconds of crossfade at each switch; both sides hold the same audio there
OPEN_SWITCH_BAR = .5  # bars after the new bars end
END_SWITCH_BAR = 14   # bar of the third pass, before its accompaniment starts fading in the draft
END_FLOOR_DB = -50.   # the ending's tail is trimmed here: below this it is only dead air
QUIET_SWITCH_BAR = 11 # bar of the third pass where the parts take over, before anything has changed
QUIET_VOICE_FROM = 11.62   # bar of the third pass where his second-last line starts
QUIET_GAINS = {'drums': ((11.3, 1.), (11.6, 0.)),   # (bar of the third pass, gain): out in the breath before the line
               'bass': ((11.3, 1.), (12., .25), (14., .1), (16., 0.)),
               'other': ((11.3, 1.), (12., .5), (13.5, .4), (15.4, .4), (15.8, .22), (16.1, .22), (16.6, .45)),   # the accompaniment swells before bar 16: held down, so the level only falls after his last word
               'voice': ((11.3, 1.), (12., .5), (13.5, .4), (15.4, .4), (15.8, .22), (16.1, .22), (16.6, .45))}   # Demucs files the lead line under this name; the take has no singing
# With an erhu rewritten under those lines (spec_p4), the erhu sits about 8 dB under his voice, then takes the closing phrase after his last word.
UNDER_LEAD = ((11.3, 1.), (11.6, .25), (16., .25), (16.5, 1.))   # its held note after his last word is loud: kept down until the closing phrase, which is quieter in the take
QUIET_GAINS_UNDER = {'drums': QUIET_GAINS['drums'], 'bass': QUIET_GAINS['bass'], 'other': UNDER_LEAD, 'voice': UNDER_LEAD}
ROOM_SECONDS, ROOM_WET = 1.6, .16
# The ritardando, in bars of the third pass: from the bar line where the band has dropped out (p7r_202619670; his second-last
# line started at 11.62) to the end of his last word 讲 (the voice stem is silent from 15.5), then held at its last tempo, so the
# drums are never stretched. The curve is the final ritard measured in performances (Friberg & Sundberg 1999):
# tempo = (1 + (w^q - 1) x)^(1/q), which slows little at first and most at the end. (stage7_v11 ran it from 11.5.)
RIT_FROM, RIT_TO, RIT_FINAL, RIT_Q = 12., 15.5, .6, 2.
# The listener on a version slowed to .6: 「慢的太多了」; on one held at .80 (last line 24% slower): 「只需要慢6%-9%」.
# So the curve stops at .93 (reached in his second-last line) and everything after keeps that tempo: the last line comes
# out about 7.5% slower.
RIT_HOLD = .93
RIT_PREROLL = 1.      # bars processed at the song's tempo before the curve starts; the slowed audio comes in half a bar before it


def rit_tempo(b):
    """Tempo against the song's at bar b of the third pass."""
    x = np.clip((b - RIT_FROM) / (RIT_TO - RIT_FROM), 0, 1)
    return np.maximum((1 + (RIT_FINAL ** RIT_Q - 1) * x) ** (1 / RIT_Q), RIT_HOLD)


def trimmed(x):
    """x without the dead air at its end (below END_FLOOR_DB), with a half-second fade."""
    levels = np.sqrt((x[:len(x) // 2205 * 2205].reshape(-1, 2205, 2) ** 2).mean(axis=(1, 2)))
    loud = np.where(levels > 10 ** (END_FLOOR_DB / 20))[0]
    x = x[:min(len(x), (int(loud[-1]) + 1) * 2205)].copy()
    fade = int(.5 * SR)
    x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return x


def spliced(a, b, at):
    """a up to sample `at`, then b, over a short crossfade; both on one clock."""
    n = int(SWITCH * SR)
    ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    out = b.copy()
    out[:at] = a[:at]
    out[at:at + n] = a[at:at + n] * (1 - ramp) + b[at:at + n] * ramp
    return out


def room(x, seed=11):
    """A soft decaying tail for x: filtered noise, low in the top and the bottom, returned on its own."""
    rng = np.random.default_rng(seed)
    n = int(ROOM_SECONDS * SR)
    decay = np.exp(-6.9 * np.arange(n) / n)
    f = np.fft.rfftfreq(n, 1 / SR)
    shape = np.clip((f - 150) / 150, 0, 1) * np.clip((6000 - f) / 3000, 0, 1)
    tail = np.stack([np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n=n) * decay for _ in range(2)], axis=1)
    tail /= np.sqrt((tail ** 2).sum(axis=0))
    size = len(x) + n
    return (np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(tail, size, axis=0), n=size, axis=0) * ROOM_WET).astype(np.float32)


base, opening_take, ending_take, name = sys.argv[1:5]
quiet = 'quiet' in sys.argv[5:]
room_on = quiet or 'room' in sys.argv[5:]
rec = json.loads((D / 'review' / f'mix_{base}.json').read_text(encoding='utf-8'))
spec = {c['id']: {'from_bar': d.get('from_bar', 8), **c} for p in sorted(A.glob('spec_p*.json')) for d in [json.loads(p.read_text(encoding='utf-8'))] for c in d['candidates']}   # from_bar: the bar of the third pass where an ending take's context starts
spec_of = lambda take: spec[take.split('__')[0]]    # a take with __ is a treated copy of the take before it
s = rec['sections']
if s['third'] < s['final']:
    sys.exit('the ending here assumes the song ends on the third pass')
bed, voice, hum = load(S / f'{base}_bed.wav'), load(S / f'{base}_voice.wav'), load(S / f'{base}_hum.wav')
points = rec['dip']
dip = np.interp(np.arange(len(bed)) / SR, [p[0] for p in points], [p[1] for p in points], left=1., right=1.)

# Ending: same audio as the draft's accompaniment up to the switch, at the draft's level, then the new bars.
ctx = load(A / spec_of(ending_take)['src_audio'])
take = load(A / 'raw' / f'{ending_take}.wav')
third_at = round(s['third'] / D_BAR) * D_BAR    # the record rounds to 10 ms, enough to turn the bass over; sections sit on whole bars
third_bar = lambda b: third_at + b * D_BAR
from_bar = spec_of(ending_take)['from_bar']
ctx_at = lambda b: (b - from_bar) * D_BAR
rewrite_bar = from_bar + float(spec_of(ending_take)['repainting_start']) / D_BAR   # where the take's own music starts
lead_under = quiet and rewrite_bar < 16


def same_audio(a, b, what):
    """The gain that makes b into a, after making sure they are the same audio."""
    n = min(len(a), len(b))
    g = float((a[:n] * b[:n]).sum() / (b[:n] ** 2).sum())
    match = float(np.corrcoef(a[:n].ravel(), b[:n].ravel())[0, 1])
    print(what, 'gain', round(g, 3), 'match', round(match, 4))
    if match < .98:
        sys.exit(f'{what}: the two are not the same audio on one clock')
    return g


g_end = same_audio(seg(bed, third_bar(9), third_bar(14)), seg(ctx, ctx_at(9), ctx_at(14)), 'third pass accompaniment')
same_audio(seg(ctx, ctx_at(9), ctx_at(min(14, rewrite_bar))), seg(take, ctx_at(9), ctx_at(min(14, rewrite_bar))), 'ending take against its context')
switch_bar = QUIET_SWITCH_BAR if quiet else min(END_SWITCH_BAR, int(rewrite_bar) - 1)   # always where the take still holds the draft's own audio
end_from = third_bar(switch_bar)
tail = seg(take, ctx_at(switch_bar), len(take) / SR) * g_end
if quiet:
    # On the take's clock, each part follows its own line of gains.
    tail = np.zeros_like(tail)
    for part, points in (QUIET_GAINS_UNDER if lead_under else QUIET_GAINS).items():
        stem = seg(load(A / 'stems' / f'{ending_take}_{part}.wav'), ctx_at(switch_bar), len(take) / SR)
        bars = switch_bar + np.arange(len(stem)) / SR / D_BAR
        n = min(len(stem), len(tail))
        tail[:n] += stem[:n] * np.interp(bars[:n], *zip(*points)).astype(np.float32)[:, None] * g_end
tail = trimmed(tail)
song_end = end_from + len(tail) / SR

# Opening: the take's new bars, then the draft's accompaniment moved later.
take = load(A / 'raw' / f'{opening_take}.wav')
new_seconds = -float(spec_of(opening_take)['repainting_start'])
shift = new_seconds - OLD_PRELUDE_BARS * D_BAR
shift_samples = int(round(shift * SR))
total = int(song_end * SR) + shift_samples
same_audio(seg(bed, 3 * D_BAR, 7 * D_BAR), seg(take, new_seconds + D_BAR, new_seconds + 5 * D_BAR), 'opening take against the draft')

music_bed = np.zeros((total, 2), np.float32)
n = min(len(bed), int(end_from * SR) + int(SWITCH * SR))
music_bed[shift_samples:shift_samples + n] = (bed[:n] * dip[:n, None]).astype(np.float32)
at = shift_samples + int(end_from * SR)
holder = np.zeros_like(music_bed)
holder[at:at + len(tail)] = tail[:total - at]
music_bed = spliced(music_bed, holder, at)
head = np.zeros_like(music_bed)
head[:min(len(take), total)] = take[:total]
music_bed = spliced(head, music_bed, int((new_seconds + OPEN_SWITCH_BAR * D_BAR) * SR))
music_bed[:int(.03 * SR)] *= np.linspace(0, 1, int(.03 * SR), dtype=np.float32)[:, None]

vocal = np.zeros_like(music_bed)
n = min(len(voice), total - shift_samples)
vocal[shift_samples:shift_samples + n] = (voice[:n] + hum[:n]).astype(np.float32)
rit = None
if 'rit' in sys.argv[5:]:
    # From a bar before the curve to the end, on the output's clock: accompaniment and dry voice each slowed their own way
    # (ritard.py), on maps that meet at every beat. The new audio comes in half a bar before the curve, where both are as they were.
    s0 = shift + third_bar(RIT_FROM - RIT_PREROLL)
    i0 = int(round(s0 * SR))
    tail_of = lambda x: np.concatenate([x[i0:], np.zeros((SR, 2), np.float32)])
    bar_of = lambda t: RIT_FROM - RIT_PREROLL + t / D_BAR
    cmap = ritard.CommonMap((len(music_bed) - i0) / SR + 1, lambda t: rit_tempo(bar_of(t)))
    slow_bed = ritard.accompaniment(tail_of(music_bed), cmap)
    beats = (np.arange(RIT_FROM, RIT_TO + 1e-9, .25) - bar_of(0)) * D_BAR
    t_in, t_out = ritard.voice_map(tail_of(vocal), cmap, beats)
    slow_voice = ritard.wsola(tail_of(vocal), t_in, t_out)
    at = int(.5 * RIT_PREROLL * D_BAR * SR)
    same_audio(seg(tail_of(music_bed), .1, at / SR + .3), seg(slow_bed, .1, at / SR + .3), 'accompaniment slowed, before the curve')
    same_audio(seg(tail_of(vocal), .1, .7), seg(slow_voice, .1, .7), 'voice slowed, before the curve')
    n = i0 + max(len(slow_bed), len(slow_voice))

    def joined(x, y):
        before, after = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
        k = min(len(x), n)
        before[:k] = x[:k]
        after[i0:i0 + len(y)] = y
        return spliced(before, after, i0 + at)
    music_bed, vocal = joined(music_bed, slow_bed), joined(vocal, slow_voice)
    new_at = lambda b: s0 + float(np.interp((b - bar_of(0)) * D_BAR, t_in, t_out))    # bar of the third pass -> output seconds, on the voice's map
    rit = {'from_bar': RIT_FROM, 'to_bar': RIT_TO, 'final_tempo': RIT_FINAL, 'q': RIT_Q, 'held_at': RIT_HOLD, 'voice': 'wsola, extra time on steady frames, meets the accompaniment every beat',
           'second_last_line': [round(new_at(11.62), 2), round(new_at(13.43), 2)], 'last_line': [round(new_at(13.62), 2), round(new_at(15.48), 2)],
           'slower_by': {'second_last_line': round((new_at(13.43) - new_at(11.62)) / (1.81 * D_BAR) - 1, 3), 'last_line': round((new_at(15.48) - new_at(13.62)) / (1.86 * D_BAR) - 1, 3)}}
if room_on:
    # after any slowing: the room keeps its own length
    at = shift_samples + int(third_bar(QUIET_VOICE_FROM) * SR)
    wet = room(vocal[at:])
    vocal[at:] += wet[:len(vocal) - at]
music = music_bed + vocal
if rit:
    music = trimmed(music)
    rit['after_last_word'] = round(len(music) / SR - rit['last_line'][1], 2)
    print('ritardando', json.dumps(rit))
total = len(music)
meter = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
                       input=music.astype(np.float32).tobytes(), capture_output=True, check=True).stderr.decode(errors='replace')
m = json.loads(meter[meter.rfind('{'):meter.rfind('}') + 1])
final = min(-16 - float(m['input_i']), -1.5 - float(m['input_tp']))
out = D / 'audio' / f'{name}.mp3'
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-af', f'volume={final:.3f}dB', '-c:a', 'libmp3lame', '-b:a', '256k', str(out)],
               input=music.astype(np.float32).tobytes(), check=True)
sections = {k: round(v + shift, 2) for k, v in s.items()}
sections['intro_voice'] = sections.pop('intro')
record = {'file': out.name, 'seconds': round(total / SR, 2), 'sections': sections, 'grown_from': base, 'opening_take': opening_take, 'ending_take': ending_take, 'shift': round(shift, 3), 'new_bars_end': round(new_seconds, 2), 'ending_new_bars_at': round(third_bar(16) + shift, 2), 'ending_gain': round(g_end, 3), 'quiet_ending': quiet, 'room': room_on, 'ending_written_from': round(third_bar(rewrite_bar) + shift, 2), 'lead_under_from': round(third_bar(rewrite_bar) + shift, 2) if lead_under else None, 'last_lines_at': round(third_bar(QUIET_VOICE_FROM) + shift, 2), 'ritardando': rit, 'final_gain_db': round(final, 2), 'input_lufs': float(m['input_i'])}
print(json.dumps(record, ensure_ascii=False))
(D / 'review' / f'mix_{name}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
