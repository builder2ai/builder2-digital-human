"""Where each section of the lyric sits in a take: the start and end of every lyric line and section, what came before the
first line (hums, a vocalise lead-in) and after the last (ad-libs, humming, the whole song again), and whether the take was
cut off.

YuE2 takes sing a lot that is not the lyric: lead-ins of 「啊」 or 「嗯」 up to 20 s long, ad-libs after the chorus, and in one
take a whole second pass of the song. The voice energy counts all of that, so timing taken from it (bars per line, the
start for scoring, where to cut) comes out wrong. This tool takes the timing from the words instead.
- each vocal stem is transcribed with transcribe_ref.transcribe() (windows from breaths, invented clauses and song credits
  cut), or an earlier transcript is reused;
- the heard characters are aligned to the lyric's characters in order, by toneless pinyin (in/ing, en/eng folded): a
  near-homophone or a misheard character still carries its time. Characters heard before the first line, between lines and
  after the last are skipped; a hum syllable (啊, 嗯, 哦 ...) costs almost nothing to skip, other extra characters cost more,
  so the alignment stays on the first pass through the lyric;
- per line: start, end, the share of its characters heard (same pinyin, or same final); per section: start (first line's
  start), end (last line's end), line_period_s (the slope of line starts against line number: seconds per line, which
  take_grid.py turns into bars per line);
- after the last line: seconds of voice, what was heard, and repeat_from if the lyric's opening lines are heard again;
- cut_off: the last line runs into the end of the take and is only partly heard, or the last lines are missing while the
  voice runs into the end.
- lines with voice under them but nothing heard (Whisper turned one whole window into an invented TV credit and dropped
  three chorus lines of an otherwise clear take) are listened to again automatically: from the start of a line heard
  before the gap, in one 20-40 s window reaching past it; the new segments are kept only if more lines are heard, and the
  transcript file is updated. --no-fill turns this off. Each gap and what was tried is in "gaps" and "filled".
Chinese lyrics only (it compares Chinese characters). Needs faster-whisper (unless every transcript exists) and pypinyin.
About 50 s a take to transcribe on a laptop CPU; WHISPER_DEVICE=cuda on a GPU box.
take_grid.py --sections, score_takes.py --sections and cut_take.py --sections read the output.
usage: take_sections.py --lyrics lyrics.txt [--sections verse,chorus] [--transcripts DIR] [--out sections.json]
                        stems/<take>_vocals.flac [...]
  --sections  which sections of the lyrics file this take was asked to sing, in order (default: every sung section;
              [spoken] and [Instrumental] never count)
  --transcripts  folder for <take>.json transcripts (default take_lines/); an existing file is reused
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
from difflib import SequenceMatcher  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pypinyin import lazy_pinyin  # noqa: E402

import take_grid as tg  # noqa: E402
from lyrics_tools import sections as lyric_sections  # noqa: E402

FILLER = set('啊嗯哦喔呜啦耶哈噢唔呀哎诶欸嘿嗚哼')
INITIALS = ('zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w')
DEL = .7                       # a lyric character not heard
INS, INS_FILLER = .6, .1       # an extra heard character between lines (a hum syllable is almost free)
INS_IN, INS_IN_FILLER = 1.2, .3   # the same inside a line: a line stays together instead of stretching over a garbled pass


def cjk(text):
    return [c for c in text if '一' <= c <= '鿿']


def pinyin(chars):
    return [p.replace('ing', 'in').replace('eng', 'en') for p in lazy_pinyin(''.join(chars))]


def split_syl(p):
    for i in INITIALS:
        if p.startswith(i):
            return i, p[len(i):]
    return '', p


def sim(a, b):
    if a == b:
        return 2.
    (ia, fa), (ib, fb) = split_syl(a), split_syl(b)
    if fa == fb:
        return .6
    return 0. if ia and ia == ib else -1.


def lyric_lines(text, wanted=None):
    """[(section name, [line, ...])] of the sung sections, in order; names made unique (chorus, chorus#2)."""
    out, seen = [], {}
    for tag, ls in lyric_sections(text):
        t = tag.lower()
        if t in ('spoken', 'instrumental') or (wanted and t not in wanted):
            continue
        body = [l.strip() for l in ls if l.strip() and not l.lstrip().startswith('@') and cjk(l)]
        if not body:
            continue
        seen[t] = seen.get(t, 0) + 1
        out.append((tag + (f'#{seen[t]}' if seen[t] > 1 else ''), body))
    return out


def heard_chars(segments):
    """[(start, end, char)] for every Chinese character heard, word times split evenly across a word's characters."""
    out = []
    for s in segments:
        words = s.get('words') or [[s['start'], s['end'], s['text']]]
        for w0, w1, w in words:
            cs = cjk(w)
            for k, c in enumerate(cs):
                out.append((w0 + (w1 - w0) * k / len(cs), w0 + (w1 - w0) * (k + 1) / len(cs), c))
    return out


def align(E, H, Hc, line_of):
    """Semi-global alignment of expected pinyin E to heard pinyin H (heard chars Hc for the hum cost; line_of[i] = which
    lyric line expected char i is in). Returns match[i] = heard index aligned to expected char i (a substitution counts),
    or None, and the similarity matrix."""
    n, m = len(E), len(H)
    S = np.array([[sim(e, h) for h in H] for e in E]) if n and m else np.zeros((n, m))
    filler = np.array([c in FILLER for c in Hc])
    ins_b = np.where(filler, INS_FILLER, INS)
    ins_in = np.where(filler, INS_IN_FILLER, INS_IN)
    D = np.zeros((n + 1, m + 1))
    P = np.zeros((n + 1, m + 1), np.int8)              # 1 diag, 2 up (expected char not heard), 3 left (extra heard char)
    D[1:, 0] = -DEL * np.arange(1, n + 1)
    P[1:, 0] = 2
    P[0, 1:] = 3                                        # heard before the lyric starts: free
    for i in range(1, n + 1):
        Di, Dp, Si = D[i], D[i - 1], S[i - 1]
        ins = ins_b if i == n or line_of[i - 1] != line_of[i] else ins_in
        for j in range(1, m + 1):
            best, op = Dp[j - 1] + Si[j - 1], 1
            if Dp[j] - DEL > best:
                best, op = Dp[j] - DEL, 2
            if Di[j - 1] - ins[j - 1] > best:
                best, op = Di[j - 1] - ins[j - 1], 3
            Di[j], P[i, j] = best, op
    i, j = n, int(np.argmax(D[n])) if m else 0     # heard after the lyric ends: free
    match = [None] * n
    while i > 0:
        op = P[i, j]
        if op == 1:
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif op == 2 or j == 0:
            i -= 1
        else:
            j -= 1
    return match, S


def appears(want, got):
    """Best SequenceMatcher ratio of pinyin list `want` against any same-length stretch of `got`, and where it starts."""
    best, at = 0., None
    for k in range(max(1, len(got) - len(want) + 1)):
        r = SequenceMatcher(None, want, got[k:k + len(want)]).ratio()
        if r > best:
            best, at = r, k
    return best, at


def fill_edges(match, exp, hc, Hc):
    """A misheard first (or last) character of a line is often skipped by the alignment and its time lost. Give such edge
    characters the unaligned heard characters right next to the line (not hums, within 1.5 s), for timing only."""
    keys = sorted({(k, i) for k, i, _ in exp})
    for k, i in keys:
        idx = [q for q, e in enumerate(exp) if e[:2] == (k, i)]
        got = [q for q in idx if match[q] is not None]
        if not got:
            continue
        q0, q1 = got[0], got[-1]
        prev_j = max([match[q] for q in range(idx[0]) if match[q] is not None], default=-1)
        lead = [q for q in idx if q < q0]
        cands = [j for j in range(prev_j + 1, match[q0]) if Hc[j] not in FILLER and hc[match[q0]][0] - hc[j][1] < 1.5]
        for q, j in zip(reversed(lead), reversed(cands)):
            match[q] = j
        next_j = min([match[q] for q in range(idx[-1] + 1, len(exp)) if match[q] is not None], default=len(Hc))
        tail = [q for q in idx if q > q1]
        cands = [j for j in range(match[q1] + 1, next_j) if Hc[j] not in FILLER and hc[j][0] - hc[match[q1]][1] < 1.5]
        for q, j in zip(tail, cands):
            match[q] = j
    return match


def analyse(tid, segments, lyric, dur, runs, stem='', transcript=''):
    hc = heard_chars(segments)
    Hc = [c for _, _, c in hc]
    H = pinyin(Hc)
    exp = []                                            # (section k, line i, pinyin)
    for k, (_, lines) in enumerate(lyric):
        for i, line in enumerate(lines):
            exp += [(k, i, p) for p in pinyin(cjk(line))]
    match, S = align([p for _, _, p in exp], H, Hc, [(k, i) for k, i, _ in exp])
    match = fill_edges(match, exp, hc, Hc)
    secs, notes = [], []
    used = [j for j in match if j is not None]
    for k, (name, lines) in enumerate(lyric):
        rows = []
        for i, line in enumerate(lines):
            idx = [q for q, (kk, ii, _) in enumerate(exp) if kk == k and ii == i]
            js = [(q, match[q]) for q in idx if match[q] is not None]
            good = sum(1 for q, j in js if S[q, j] >= .6)
            row = {'text': line, 'heard': round(good / len(idx), 2) if idx else 0.}
            if js:
                row['start'] = round(hc[js[0][1]][0], 2)
                row['end'] = round(hc[js[-1][1]][1], 2)
                row['got'] = ''.join(Hc[j] for _, j in js)
            rows.append(row)
        timed = [(i, r) for i, r in enumerate(rows) if 'start' in r and r['heard'] >= .4]
        sec = {'name': name, 'lines': len(lines), 'heard_lines': sum(1 for r in rows if r['heard'] >= .5), 'line_detail': rows}
        if timed:
            sec['start'], sec['end'] = timed[0][1]['start'], timed[-1][1]['end']
            sec['seconds'] = round(sec['end'] - sec['start'], 2)
            if timed[0][0] != 0 or timed[-1][0] != len(lines) - 1:
                notes.append(f"{name}: first or last line not heard clearly; its span is from line {timed[0][0] + 1} to {timed[-1][0] + 1}")
            if len(timed) >= 2:
                sec['line_period_s'] = round(float(np.polyfit([i for i, _ in timed], [r['start'] for _, r in timed], 1)[0]), 3)
            spans = [r['end'] - r['start'] for _, r in timed]
            for i, r in timed:
                if len(spans) >= 3 and r['end'] - r['start'] > 3 * np.median(spans) + 1:
                    notes.append(f"{name} line {i + 1}: heard over {r['end'] - r['start']:.1f} s, much longer than the others; check its times")
        else:
            sec['start'] = sec['end'] = None
            notes.append(f'{name}: not found in what was heard')
        secs.append(sec)
    flat = [(s['name'], i, r) for s in secs for i, r in enumerate(s['line_detail'])]
    gaps, i = [], 0
    while i < len(flat):                                # runs of lines not heard, across section borders
        if 'start' in flat[i][2] and flat[i][2]['heard'] >= .4:
            i += 1
            continue
        i1 = i
        while i1 + 1 < len(flat) and not ('start' in flat[i1 + 1][2] and flat[i1 + 1][2]['heard'] >= .4):
            i1 += 1
        before = [x[2] for x in flat[:i] if 'start' in x[2] and x[2]['heard'] >= .4]
        after_ = [x[2] for x in flat[i1 + 1:] if 'start' in x[2] and x[2]['heard'] >= .4]
        lo = before[-1]['end'] if before else 0.
        hi = after_[0]['start'] if after_ else (runs[-1][1] if runs else dur)
        voiced = sum(min(e, hi) - max(s, lo) for s, e in runs if min(e, hi) > max(s, lo))
        if voiced > 1.5:
            which = ', '.join(f'{n} line {k + 1}' for n, k, _ in flat[i:i1 + 1])
            gaps.append({'from': round(lo, 2), 'to': round(hi, 2), 'lines': which, 'voice_s': round(voiced, 1),
                         'starts_before': [round(x['start'], 2) for x in before[::-1][:6]]})
            notes.append(f"not heard: {which}, but there is {voiced:.0f} s of voice at {lo:.1f}-{hi:.1f} s "
                         f"(transcribe_ref.py {stem} --from {max(0, lo - 1):.0f} --to {hi + 1:.0f} --merge-into {transcript})")
        i = i1 + 1
    found = [s for s in secs if s['start'] is not None]
    for s0, s1 in zip(found, found[1:]):
        v = sum(min(e, s1['start']) - max(s, s0['end']) for s, e in runs if min(e, s1['start']) > max(s, s0['end']))
        if v > 2:
            txt = ''.join(c for t0, _, c in hc if s0['end'] < t0 < s1['start'])
            notes.append(f"{v:.0f} s of voice between {s0['name']} and {s1['name']} ({s0['end']:.1f}-{s1['start']:.1f} s)"
                         + (f': {txt[:30]}' if txt else ''))
    out = {'id': tid, 'seconds': round(dur, 2), 'sections': secs, 'notes': notes, 'gaps': gaps,
           'heard_mean': round(float(np.mean([r['heard'] for s in secs for r in s['line_detail']])), 2) if secs else None}
    if not found:
        out.update(sung_from=None, sung_to=None, cut_off=None)
        return out
    a, z = found[0]['start'], found[-1]['end']
    out['sung_from'], out['sung_to'] = a, z
    first_j, last_j = (min(used), max(used)) if used else (0, -1)
    before = [r for r in runs if r[0] < a - .3]
    out['lead_in'] = {'seconds': round(sum(min(e, a - .3) - s for s, e in before), 1),
                      'from': round(before[0][0], 2) if before else None, 'text': ''.join(Hc[:first_j])}
    after_runs = [(max(s, z + .3), e) for s, e in runs if e > z + .3]
    tail = H[last_j + 1:]
    after = {'seconds': round(sum(e - s for s, e in after_runs), 1), 'to': round(after_runs[-1][1], 2) if after_runs else None,
             'text': ''.join(Hc[last_j + 1:])}
    first_two = [p for kk, ii, p in exp if kk == 0 and ii < 2]
    r, at = appears(first_two, tail)
    if tail and r >= .6:
        after['repeat_from'] = round(hc[last_j + 1 + at][0], 2)
    out['after'] = after
    if before and out['lead_in']['text']:
        r0, at0 = appears(first_two, H[:first_j])
        if r0 >= .6:
            notes.append(f"the opening lines may also be sung before {a:.1f} s (from {hc[at0][0]:.1f} s): two passes?")
    last = secs[-1]['line_detail']
    missing = 0
    for r in reversed(last):
        if r['heard'] >= .4:
            break
        missing += 1
    voice_to_end = bool(runs) and runs[-1][1] > dur - .5
    if missing and voice_to_end:
        out['cut_off'] = f'the last {missing} line(s) not heard and the voice runs into the end of the take'
    elif dur - z < 1. and last[-1]['heard'] < .6:
        out['cut_off'] = f'the last line runs into the end of the take ({z:.1f} of {dur:.1f} s) and is only partly heard'
    else:
        out['cut_off'] = None
        if missing:
            notes.append(f'the last {missing} line(s) not heard clearly (skipped, garbled or not transcribed)')
    return out


def heard_total(r):
    return sum(s['heard_lines'] for s in r['sections'])


def fill_gaps(stem, segments, lyric, dur, runs, r, tf, text, get_model):
    """Lines with voice under them but nothing heard: Whisper sometimes turns a whole 20 s window into an invented credit
    and drops it (one take lost three chorus lines that way), and a short re-listen of just the gap invents again. So listen
    again from the start of a line heard before the gap (a clean entry), with a window long enough to reach past the gap
    (20-40 s, which worked where short ones failed), and keep the new segments only if more lines get heard."""
    import transcribe_ref as tr
    filled, tried = [], set()
    for _ in range(8):                                  # at most 8 re-listens a take
        todo = None
        for g in r.get('gaps', []):
            sb = g['starts_before']
            cands = ([sb[2]] if len(sb) >= 3 else []) + [x for x in (r.get('sung_from'), sb[0] if sb else None,
                                                                       sb[1] if len(sb) > 1 else None, max(0., g['from'] - 6)) if x is not None]
            for t0 in dict.fromkeys(round(c, 2) for c in cands):
                t1 = round(min(dur, max(t0 + 30, g['to'] + 3)), 2)
                if t1 - t0 <= 40 and (t0, t1) not in tried:
                    todo = (g, t0, t1)
                    break
            if todo:
                break
        if not todo:
            break
        g, t0, t1 = todo
        tried.add((t0, t1))
        new = tr.transcribe(stem, max(0., t0 - .3), t1, window=t1 - t0 + .3, model=get_model(), expected=text)[0]
        add = [s for s in new if all(tr.overlap(s, b) < .3 * max(.01, s['end'] - s['start']) for b in segments)]
        gained = 0
        if add:
            trial = sorted(segments + add, key=lambda s: s['start'])
            r2 = analyse(r['id'], trial, lyric, dur, runs, stem, tf)
            gained = heard_total(r2) - heard_total(r)
            if gained > 0:
                segments, r = trial, r2
        filled.append({'gap': g['lines'], 'listened': [t0, t1], 'lines_gained': max(0, gained)})
    return segments, r, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--lyrics', required=True)
    ap.add_argument('--sections', help='comma list of section names this take sings, e.g. verse,chorus')
    ap.add_argument('--transcripts', default='take_lines')
    ap.add_argument('--out', default='sections.json')
    ap.add_argument('--no-fill', action='store_true', help='do not listen again where lines are missing')
    a = ap.parse_args()
    text = open(a.lyrics, encoding='utf-8').read()
    wanted = [s.strip().lower() for s in a.sections.split(',')] if a.sections else None
    lyric = lyric_lines(text, wanted)
    if not lyric:
        raise SystemExit('no sung lines found in the lyrics file for these sections')
    tdir = Path(a.transcripts)
    tdir.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}

    def get_model():
        if 'm' not in cache:
            from score_lines import load_model
            cache['m'] = load_model()[0]
        return cache['m']
    for stem in a.stems:
        tid = Path(stem).stem
        for suffix in ('_vocals', '_voice'):
            if tid.endswith(suffix):
                tid = tid[:-len(suffix)]
        tf = tdir / f'{tid}.json'
        if tf.exists():
            segments = json.load(open(tf, encoding='utf-8'))
        else:
            import transcribe_ref
            segments = transcribe_ref.transcribe(stem, model=get_model(), expected=text)[0]
            tf.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding='utf-8')
        v = tg.load_mono(stem)
        dur, runs = len(v) / tg.SR, tg.voice_runs(v)
        r = analyse(tid, segments, lyric, dur, runs, stem, tf)
        if r.get('gaps') and not a.no_fill:
            segments, r, filled = fill_gaps(stem, segments, lyric, dur, runs, r, tf, text, get_model)
            r['filled'] = filled
            if any(f['lines_gained'] for f in filled):
                tf.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding='utf-8')
        r['stem'], r['transcript'] = str(stem), str(tf)
        rows.append(r)
        print(f"== {tid}  ({r['seconds']} s, lines heard {r['heard_mean']})")
        for f in r.get('filled', []):
            print(f"   listened again for {f['gap']}: {f['listened']} -> {f['lines_gained']} more line(s) heard")
        li = r.get('lead_in') or {}
        if li.get('seconds'):
            print(f"   before the first line: {li['seconds']} s of voice from {li['from']} s  {li['text'][:30]}")
        for s in r['sections']:
            if s['start'] is None:
                print(f"   {s['name']:<10} not found")
                continue
            print(f"   {s['name']:<10} {s['start']:6.2f}-{s['end']:6.2f} s  {s['heard_lines']}/{s['lines']} lines heard  "
                  f"{s.get('line_period_s', '-')} s/line")
        af = r.get('after') or {}
        if af.get('seconds'):
            print(f"   after the last line: {af['seconds']} s of voice to {af['to']} s"
                  + (f", sings the lyric again from {af['repeat_from']} s" if af.get('repeat_from') else '') + f"  {af['text'][:30]}")
        if r.get('cut_off'):
            print(f"   CUT OFF: {r['cut_off']}")
        for n in r['notes']:
            print(f'   note: {n}')
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'-> {a.out}')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
