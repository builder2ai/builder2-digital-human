"""QC for a spoken line made with a TTS model from a designed voice: the takes on their own, then the line inside the mix.

The digital-human skill's tts_qc.py is made for many lines in one cloned voice: its speaker check needs a reference
profile (CAM++) and it has no pitch check, so it does not fit a single line in a voice designed from text, where every
take is its own voice. Here, per take:
- words: Whisper's free transcript against the line, as an error rate by pinyin (a homophone Whisper writes with another
  character counts as right) and by character;
- pace: characters per second over the voiced span; where the voice starts and ends inside the file;
- pitch: median f0 and its 10th-90th percentiles (YIN), with a hint when the median is outside the usual speaking range
  for --voice (male about 85-180 Hz, female about 165-255 Hz; excited speech runs higher). In one test six takes of a
  young man's line had medians of 170-215 Hz; the lowest with every word right was picked, and the ear decides.
With --mix (the finished song) and --at A-B (song seconds of the line, from assemble.py's record), the line is transcribed
in three windows of the mix (and of --voice-bus if given): it should be heard in all three (it was, in one test, at a 4 dB
dip of the accompaniment).
Needs faster-whisper and pypinyin (the score group of check_setup.py); WHISPER_MODEL, WHISPER_DEVICE as in score_lines.py.
usage: spoken_qc.py --line "你还没睡吧？" [--voice male|female] take1.wav [take2.wav ...] [--out qc.json]
       spoken_qc.py --line "..." --mix final.wav --at 36.61-38.51 [--voice-bus final_voice.wav]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
from difflib import SequenceMatcher  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pypinyin import lazy_pinyin  # noqa: E402

import harmony_groove as hg  # noqa: E402
import take_grid as tg  # noqa: E402
from score_lines import SR, cjk, load_audio, load_model, transcribe  # noqa: E402

RANGE = {'male': (85, 180), 'female': (165, 255)}


def folded(text):
    return [p.replace('ing', 'in').replace('eng', 'en') for p in lazy_pinyin(text)]


def error_rate(want, got):
    return round(1 - SequenceMatcher(None, want, got).ratio(), 3)


def heard_in(want, got):
    """Share of the line's syllables found, in order, in what was heard."""
    w, g = folded(want), folded(got)
    blocks = SequenceMatcher(None, w, g).get_matching_blocks()
    return round(sum(b.size for b in blocks) / max(1, len(w)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('takes', nargs='*')
    ap.add_argument('--line', required=True)
    ap.add_argument('--voice', choices=tuple(RANGE))
    ap.add_argument('--mix')
    ap.add_argument('--at', help='song seconds A-B of the line in --mix')
    ap.add_argument('--voice-bus')
    ap.add_argument('--out')
    a = ap.parse_args()
    line = cjk(a.line)
    m, _ = load_model()
    rows = []
    for f in a.takes:
        x = load_audio(f)
        hop = int(.01 * SR)
        e = np.array([np.sqrt((x[i:i + 2 * hop] ** 2).mean()) for i in range(0, max(1, len(x) - 2 * hop), hop)])
        on = np.where(e > .05 * e.max())[0]
        s0, s1 = on[0] * .01, on[-1] * .01 + .02
        heard = transcribe(m, x, 0., len(x) / SR)
        y = tg.load_mono(f)                                  # 22.05 kHz for YIN
        f0 = hg.yin(y[int(s0 * tg.SR):int(s1 * tg.SR)], fmin=60., fmax=500., frame=1024, hop=256)
        f0 = f0[np.isfinite(f0)]
        r = {'take': str(f), 'voiced_s': [round(s0, 2), round(s1, 2)], 'heard': heard, 'error_pinyin': error_rate(folded(line), folded(heard)),
             'error_chars': error_rate(line, heard), 'chars_per_s': round(len(line) / max(.1, s1 - s0), 1),
             'f0_median_hz': round(float(np.median(f0))) if len(f0) >= 10 else None,
             'f0_p10_p90_hz': [round(float(np.percentile(f0, 10))), round(float(np.percentile(f0, 90)))] if len(f0) >= 10 else None,
             'voiced_frames': int(len(f0)), 'tail_60ms_db': round(float(20 * np.log10(np.sqrt((x[-int(.06 * SR):] ** 2).mean()) + 1e-9)), 1)}
        hints = []
        if r['error_pinyin'] > 0:
            hints.append('words not all right')
        if r['f0_median_hz'] is None:
            hints.append('pitch unreliable (few voiced frames)')
        elif a.voice:
            lo, hi = RANGE[a.voice]
            if not lo <= r['f0_median_hz'] <= hi:
                hints.append(f"median {r['f0_median_hz']} Hz is outside the usual {a.voice} range {lo}-{hi} Hz: listen")
        if r['tail_60ms_db'] > -40:
            hints.append('the file ends loud: cut off?')
        r['hints'] = hints
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    if rows:
        ok = [r for r in rows if r['error_pinyin'] == 0 and r['f0_median_hz']]
        if ok:
            key = (lambda r: r['f0_median_hz']) if a.voice == 'male' else (lambda r: -r['f0_median_hz']) if a.voice == 'female' else (lambda r: 0)
            print('words right:', ', '.join(f"{Path(r['take']).name} {r['f0_median_hz']} Hz" for r in sorted(ok, key=key)), '(the ear picks)')
    mix = None
    if a.mix:
        if not a.at:
            raise SystemExit('give --at A-B (song seconds of the line)')
        s0, s1 = (float(v) for v in a.at.split('-'))
        mix = {}
        for name, path in (('mix', a.mix), ('voice_bus', a.voice_bus)):
            if not path:
                continue
            x = load_audio(path)
            got = [transcribe(m, x, max(0., s0 - .5 + d), s1 + .6) for d in (0., -.4, .25)]
            share = [heard_in(line, g) for g in got]
            mix[name] = {'heard': got, 'share': share, 'windows_heard': sum(s >= .8 for s in share)}
            print(f"{name}: the line heard in {mix[name]['windows_heard']} of 3 windows {share}: {got}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({'line': a.line, 'takes': rows, 'in_mix': mix}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
