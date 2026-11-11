"""Join two takes on a bar line: section A (e.g. a verse) runs to the first bar line after its last sung note, section B
(e.g. a chorus) comes in on its first sung bar line, with its pickup, through a 60 ms equal-power switch.

- grids from take_grid.py (beats tracked on the backing stem, period from a line fit), so bar lines don't drift;
- keys must match (or be relatives with --relative-ok), else it refuses;
- B is stretched as a whole with ffmpeg rubberband (transients=crisp, detector=compound) when the tempos differ by more than
  0.5%; we have not needed more than a few percent, and more than about 7% changes the feel;
- B's level is matched to A's on the backing, bar before against bar after, capped at --gain-cap dB;
- raw takes can clip (YuE2 output can sit at full scale): the joined sample is brought to -1 dBTP at most;
- checks printed (beat_check.py, 4 bars each side, on A's grid): the eighth-note phase jump at the join (0 = no jump; over
  about 20 ms, run again with --shift-ms set to the printed jump), and the accent profile of B's bars against A's: a best
  shift that is not 0 means B's accents sit that many eighths off (B's downbeat guess may be wrong, or the chorus plays
  another pattern: listen; if it is the downbeat, run again with the printed --shift-ms). Measured at the eighth, not the
  beat: on a 3+3+2 groove a beat-period reading was pulled by up to half a beat. Also the level of the bar before and
  after, integrated loudness and true peak (ffmpeg loudnorm).
usage: join_takes.py <take A> <take B> --out sample.wav [--stems DIR] [--bpm-hint 108] [--relative-ok] [--shift-ms 0]
                     [--gain-cap 6] [--lead-bars 1] [--tail-bars 1]
  stems: DIR/<take>_instrumental.flac and _vocals.flac (needed: the grid comes from the backing, the sung span from the vocals)
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import beat_check  # noqa: E402
import take_grid as tg  # noqa: E402
from songmap import SR, db, load, seg, write  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('a')
ap.add_argument('b')
ap.add_argument('--out', required=True)
ap.add_argument('--stems', required=True)
ap.add_argument('--bpm-hint', type=float)
ap.add_argument('--relative-ok', action='store_true')
ap.add_argument('--shift-ms', type=float, default=0.)
ap.add_argument('--gain-cap', type=float, default=6.)
ap.add_argument('--lead-bars', type=float, default=1.)
ap.add_argument('--tail-bars', type=float, default=1.)
args = ap.parse_args()
st = Path(args.stems)


def stems(take):
    n = Path(take).stem
    bk, vo = st / f'{n}_instrumental.flac', st / f'{n}_vocals.flac'
    if not bk.exists() or not vo.exists():
        raise SystemExit(f'missing stems for {n} in {st} (separate.py --model vocals)')
    return bk, vo


bkA, voA = stems(args.a)
bkB, voB = stems(args.b)
mA = tg.measure(args.a, voA, bkA, args.bpm_hint)
mB = tg.measure(args.b, voB, bkB, args.bpm_hint)
for m in (mA, mB):
    print(f"{m['id']}: {m['key']} {m['bpm']} BPM (sd {m['beat_fit_sd_ms']} ms), downbeat {m['downbeat']}, voice {m['voice_from']}-{m['voice_to']}, peak {m['peak']}")
if mA['key'] != mB['key'] and not (args.relative_ok and tg.signature(mA['key']) == tg.signature(mB['key'])):
    raise SystemExit(f"keys differ ({mA['key']} vs {mB['key']}): pick another pair, or continue B from A with render_yue2.py keep_from")

pA, pB = mA['beat_s'], mB['beat_s']
factor = pB / pA if abs(pB / pA - 1) > .005 else 1.
A, Ai = load(args.a), load(bkA)
with tempfile.TemporaryDirectory() as tmp:
    def stretched(path):
        if factor == 1.:
            return load(path)
        o = str(Path(tmp) / (Path(path).stem + '.wav'))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-af', f'rubberband=tempo={factor:.6f}:transients=crisp:detector=compound',
                        '-ar', str(SR), '-ac', '2', o], check=True, stdin=subprocess.DEVNULL)
        return load(o)
    B, Bi = stretched(args.b), stretched(bkB)
barA = 4 * pA
dA, vfA, vtA = mA['downbeat'], mA['voice_from'], mA['voice_to']
dB, vfB, vtB = mB['downbeat'] / factor, mB['voice_from'] / factor, mB['voice_to'] / factor


def bar_line(d, t, up):
    k = (t - d) / barA
    return d + (math.ceil(k) if up else math.floor(k)) * barA


SA = bar_line(dA, vfA + .15 * barA, False)
start = max(0., SA - args.lead_bars * barA)
J = bar_line(dA, vtA - .1, True)
if J > len(A) / SR - .1:
    raise SystemExit(f"{mA['id']} ends at {len(A) / SR:.2f} s, before its bar line at {J:.2f} s (cut off?): pick another take or a longer max_duration")
S = bar_line(dB, vfB + .15 * barA, False)
if S < vfB - .75 * barA:
    S += barA
S += args.shift_ms / 1000
E = min(bar_line(dB, vtB - .1, True) + args.tail_bars * barA, len(B) / SR)
xf = .06
sw = max(0., S - vfB) + .08          # switch a little before B's first sung note, so its pickup comes with it
gain_db = float(np.clip(db(seg(Ai, J - barA, J)) - db(seg(Bi, S, S + barA)), -args.gain_cap, args.gain_cap))
g = 10 ** (gain_db / 20)
n = int(round(xf * SR))
head = seg(A, start, J - sw - xf)
a_x, b_x = seg(A, J - sw - xf, J - sw)[:n], g * seg(B, S - sw - xf, S - sw)[:n]
n = min(len(a_x), len(b_x))
fo, fi = np.sqrt(np.linspace(1, 0, n))[:, None], np.sqrt(np.linspace(0, 1, n))[:, None]
tail = g * seg(B, S - sw, E)
fade = np.ones((len(tail), 1))
k = min(len(tail), int(round(args.tail_bars * barA * SR)))
if k:
    fade[-k:, 0] = np.linspace(1, 0, k) ** 1.5
fin = np.ones((len(head), 1))
if start > 0:
    m = min(len(head), int(.3 * SR))
    fin[:m, 0] = np.linspace(0, 1, m)
out = np.concatenate([head * fin, a_x[:n] * fo + b_x[:n] * fi, tail * fade]).astype(np.float32)
jt = J - start
bc = beat_check.check(out, jt, barA, new='after', before_bars=4, after_bars=4, tol_ms=20.)
jump = bc['jump_8th_ms'][0] / 1000
eighth = barA / 8


def loudness(x):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-af', 'loudnorm=print_format=json',
                        '-f', 'null', '-'], input=np.ascontiguousarray(x, np.float32).tobytes(), capture_output=True, check=True).stderr.decode(errors='replace')
    m = json.loads(r[r.rfind('{'):r.rfind('}') + 1])
    return float(m['input_i']), float(m['input_tp'])


lufs, tp = loudness(out)
if tp > -1.:
    out = out * 10 ** ((-1. - tp - .1) / 20)
    lufs, tp = loudness(out)
write(out, args.out)
info = {'a': mA['id'], 'b': mB['id'], 'key': mA['key'], 'b_key': mB['key'], 'a_bpm': mA['bpm'], 'b_bpm': mB['bpm'], 'b_stretch': round(factor, 4),
        'shift_ms': args.shift_ms, 'gain_b_db': round(gain_db, 1), 'gain_capped': abs(gain_db) >= args.gain_cap - 1e-6,
        'a_span_s': [round(start, 2), round(J, 2)], 'b_span_s_after_stretch': [round(S - sw, 2), round(E, 2)],
        'join_bar_line_in_sample_s': round(jt, 2), 'sample_seconds': round(len(out) / SR, 2),
        'eighth_phase_jump_ms': bc['jump_8th_ms'], 'eighth_phase_strength_before_after': [bc['context']['strength'], bc['new_part']['strength']],
        'accents_8ths_before_after': [bc['context']['accents_8ths'], bc['new_part']['accents_8ths']],
        'accent_shift_8ths': bc['accent_shift_8ths'], 'accent_corr_at_0_best': [bc['accent_corr_at_0'], bc['accent_corr_best']],
        'beat_verdict': bc['verdict'],
        'level_db_bar_before_after_join': [db(seg(out, jt - barA, jt)), db(seg(out, jt, jt + barA))],
        'lufs': round(lufs, 1), 'true_peak_dbtp': round(tp, 1), 'raw_peaks': [mA['peak'], mB['peak']]}
Path(args.out).with_suffix('.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(info, ensure_ascii=False))
print('beat check:', beat_check.line('join', bc))
if 'displaced' in bc['verdict']:
    k = bc['accent_shift_8ths']
    fine = jump if 'jump' in bc['verdict'] else 0.
    print(f"B's accents sit {k:+d} eighths off A's: B's downbeat guess may be wrong (take_grid's downbeat is a guess), or the "
          f"chorus plays another pattern. Listen; if it is the downbeat, run again with --shift-ms {args.shift_ms + (k * eighth + fine) * 1000:.0f}")
elif 'jump' in bc['verdict']:
    print(f'eighth-note phase jumps {jump * 1000:+.0f} ms at the join: run again with --shift-ms {args.shift_ms + jump * 1000:.0f}')
elif 'weak' in bc['verdict']:
    print('no clear eighth-note pulse on one side: listen to the join')
