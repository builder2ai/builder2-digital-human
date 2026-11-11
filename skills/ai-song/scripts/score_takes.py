"""Which takes sing the words most clearly? Each vocal stem is transcribed by Whisper in several windows (a window that starts
in a lead-in makes Whisper invent text), and each expected line is matched against what it heard by sound: pinyin without
tones, with in/ing and en/eng folded together, so a near-homophone counts as heard.

Known music hallucinations (subtitle credits, "thanks for watching", song credits such as 「词曲」 followed by a songwriter's
name) are cut out segment by segment; a window that is mostly invented is dropped, not scored low. A marker that really is
in the lyric does not count. The row lists windows_dropped and segments_cut with the marker that caused each.
Where to listen: takes start singing anywhere (YuE2 lead-ins of hums or 「啊」 ran up to 20 s) and often sing on after the
lyric (ad-libs, a second pass of the song, which would give a line a second chance). So:
- --sections sections.json (take_sections.py, best): windows run from the first lyric line (plus each offset) to 1 s after
  the last lyric line;
- --at auto (default without it): from the first voice run; when every window from there is invented or matches the lyric
  badly (a hum start), the next voice run after a 1 s gap is tried, up to 3 starts;
- --at-from takes.json uses take_grid.py's sung_from (or voice_from); a number applies to every stem.
Which words are wrong: per line, the characters each window missed by pinyin (in the stretch of the transcript that
matches the line best). A character counts as wrong only when it is missed in at least 2 of the windows (2 of 2 when one
window was dropped; with one window nothing is called wrong); the row lists them as "wrong". One miss is noise: on the
same audio, two runs moved a line by up to 0.29, about one character a line. Forced alignment (score_lines.py) only
finds where a word is: in one test it put six characters near p = 0 in all six renders of a take, and the free
transcripts heard two of them right and a polyphone among them was sung right. A polyphone that is "wrong" here goes to
forced_reading.py before anything is re-sung (Whisper may hear another initial, as 让 for 行).
Cost: about 1.5-3 min a take on a laptop CPU (large-v3-turbo int8). On a GPU box with faster-whisper, set WHISPER_DEVICE=cuda.
Score only the takes that passed take_grid.py's key and tempo filter.
usage: score_takes.py --lines "line 1|line 2|..." [--sections sections.json | --at auto|SECONDS | --at-from takes.json]
                      [--offsets 0,-1,0.4] [--length 25] stem [stem ...]
  --length  window length, default to the end of the lyric (with --sections) or of the stem
Prints one JSON row per take (mean, per line, what was heard) and the best takes; --out writes them all.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from difflib import SequenceMatcher  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pypinyin import lazy_pinyin  # noqa: E402

import take_grid as tg  # noqa: E402
from score_lines import SR, cjk, load_audio, load_model  # noqa: E402

HALLUCINATIONS = ('字幕', '订阅', '点赞', '打赏', '独播剧场', '明镜', '请不吝', '谢谢观看', '感谢观看',
                  '词曲', '作词', '作曲', '编曲', '演唱', '混音', '制作人')


def folded(text):
    return [p.replace('ing', 'in').replace('eng', 'en') for p in lazy_pinyin(text)]


def found(line, heard):
    want, got = folded(line), folded(heard)
    best = 0.
    for i in range(max(1, len(got) - len(want) + 1)):
        best = max(best, SequenceMatcher(None, want, got[i:i + len(want)]).ratio())
    return round(best, 2)


def missed(line, heard):
    """Positions of the line's characters that the stretch of `heard` matching the line best does not have, by pinyin."""
    want, got = folded(cjk(line)), folded(heard)
    n = len(want)
    best, bi, bl = -1., 0, n
    for size in range(max(1, n - 1), n + 3):                # the heard stretch may have a character more or less
        for i in range(max(1, len(got) - size + 1)):
            r = SequenceMatcher(None, want, got[i:i + size]).ratio()
            if r > best:
                best, bi, bl = r, i, size
    hit = set()
    for b in SequenceMatcher(None, want, got[bi:bi + bl], autojunk=False).get_matching_blocks():
        hit.update(range(b.a, b.a + b.size))
    return [k for k in range(n) if k not in hit]


def wrong_words(lines, texts):
    """Characters missed in at least 2 windows (2 of 2 when only two are left); none with a single window."""
    need = 2 if len(texts) >= 2 else None
    out = []
    for i, line in enumerate(lines):
        chars = cjk(line)
        counts = [0] * len(chars)
        for t in texts:
            for k in missed(line, t):
                counts[k] += 1
        for k, c in enumerate(counts):
            if c and need and c >= need:
                out.append({'line': i + 1, 'char': chars[k], 'at': k, 'missed': f'{c}/{len(texts)}'})
    return out


ap = argparse.ArgumentParser()
ap.add_argument('stems', nargs='+')
ap.add_argument('--lines', required=True)
ap.add_argument('--sections', help="take_sections.py's output: listen from the first lyric line to the last")
ap.add_argument('--at', default='auto')
ap.add_argument('--at-from')
ap.add_argument('--offsets', default='0,-1,0.4')
ap.add_argument('--length', type=float)
ap.add_argument('--out')
a = ap.parse_args()
lines = a.lines.split('|')
marks = tuple(h for h in HALLUCINATIONS if h not in ''.join(lines))
known = {r['id']: r.get('sung_from', r.get('voice_from')) for r in json.load(open(a.at_from, encoding='utf-8'))} if a.at_from else {}
spans = {r['id']: (r.get('sung_from'), r.get('sung_to')) for r in json.load(open(a.sections, encoding='utf-8'))} if a.sections else {}


def take_id(path):
    stem = Path(path).stem
    for suffix in ('_vocals', '_voice'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def starts(path):
    """Candidate start times, best first, and the end of the stretch to listen to (None = to the end)."""
    tid = take_id(path)
    if spans.get(tid, (None,))[0] is not None:
        return [spans[tid][0]], spans[tid][1] + 1.
    if known.get(tid) is not None:
        return [float(known[tid])], None
    if a.at != 'auto':
        return [float(a.at)], None
    runs = tg.voice_runs(tg.load_mono(path))
    cand = [runs[0][0]] if runs else [0.]
    for (s0, e0), (s1, _) in zip(runs, runs[1:]):       # later entries after a gap of a second or more
        if s1 - e0 >= 1. and len(cand) < 3:
            cand.append(s1)
    return cand, None


def listen(audio, at, end):
    texts = []
    for off in (float(x) for x in a.offsets.split(',')):
        s = max(0., at + off)
        e = s + a.length if a.length else end
        x = audio[int(s * SR):int(e * SR)] if e else audio[int(s * SR):]
        stream, _ = m.transcribe(x, language='zh', beam_size=5, vad_filter=False, condition_on_previous_text=False)
        segs = [re.sub(r'\W', '', seg.text) for seg in stream]
        bad = [s for s in segs if any(h in s for h in marks)]
        good = ''.join(s for s in segs if s not in bad)
        if bad and len(''.join(bad)) >= len(good):        # mostly invented: the window started or sat in music
            dropped.append(f"{off:+g} s: " + '/'.join(h for h in marks if any(h in s for s in bad)))
        elif good:
            texts.append(good)
            if bad:
                cut.append(f"{off:+g} s: " + '/'.join(h for h in marks if any(h in s for s in bad)))
    return texts


m, _ = load_model()
rows = []
for path in a.stems:
    audio = load_audio(path)
    cands, end = starts(path)
    tried, best, dropped, cut = [], None, [], []
    for at0 in cands:
        tx = listen(audio, at0, end)
        pl = [round(max(found(l, t) for t in tx), 2) for l in lines] if tx else None
        tried.append(round(at0, 2))
        if pl and (best is None or np.mean(pl) > np.mean(best[2])):
            best = (at0, tx, pl)
        if pl and np.mean(pl) >= .25:                      # a real start: stop looking
            break
    at, texts, per_line = best if best else (cands[0], [], None)
    if not texts:
        rows.append({'take': path, 'mean': None, 'starts_tried': tried, 'windows_dropped': dropped,
                     'note': 'every window invented text; run take_sections.py and use --sections, or other offsets'})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        continue
    wrong = wrong_words(lines, texts)
    rows.append({'take': path, 'from': round(at, 2), 'to': round(end, 2) if end else None, 'starts_tried': tried,
                 'mean': round(float(np.mean(per_line)), 3), 'per_line': per_line, 'windows': len(texts),
                 'windows_dropped': dropped, 'segments_cut': cut, 'wrong': wrong, 'heard': texts[0], 'heard_all': texts})
    print(json.dumps({k: v for k, v in rows[-1].items() if k != 'heard_all'}, ensure_ascii=False), flush=True)
    if wrong:
        print('  wrong (missed in 2+ windows):', '; '.join(f"L{w['line']} {w['char']} {w['missed']}" for w in wrong), flush=True)
    elif len(texts) < 2:
        print('  only one window left: no word is called wrong; run again with other --offsets', flush=True)
ok = [r for r in rows if r['mean'] is not None]
print('best', [(r['mean'], r['take']) for r in sorted(ok, key=lambda r: -r['mean'])[:6]])
if a.out:
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
