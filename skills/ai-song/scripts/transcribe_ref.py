"""Transcribe a vocal stem in windows that start where the singing starts, for 扒曲 (the sung lines and spoken parts of a
reference) and for checking takes.

One Whisper pass over a whole song misses whole sections and invents lines on music (subtitle credits, TV-station names).
- windows start where the voice starts: the first just before the first singing, each next one at the quietest moment (a
  breath between lines) in the last few seconds of the window before, so neighbouring windows overlap by 6-11 s and none
  starts in the middle of a word; stretches without voice are skipped. Windows that start in music or silence invent more,
  and shorter windows invent more still, so don't shrink them to fill a gap: run again on that stretch with --from/--to
  and --merge-into.
- invented text: only the clauses holding a known hallucination are cut (by word times); the real words in the same segment
  stay. Song credits count (词曲, 作词, 作曲, 编曲 ...: Whisper wrote 「词曲」 plus a songwriter's name over hums in two takes),
  unless the word really is in the lyric (transcribe(expected=...)).
- take_sections.py calls transcribe() on each take's vocal stem.
- overlapping windows: the segment that sits more centrally in its window wins.
- lines split at a window edge (two segments under 0.25 s apart, one of them ending or starting within 0.4 s of an edge) are
  joined again, so line lengths are not halved.
- --merge-into lines.json adds the new segments into the gaps of an existing file and writes the result there (or to --out).
- at the end it lists stretches of voice longer than 2 s that no segment covers, with the command to fill each.
Writes [{"start", "end", "text", "no_speech", "words": [[start, end, word], ...]}]; analyze_ref.py --lines reads it.
Runs wherever faster-whisper is installed; a CPU works (a 3:43 stem took about 2 min). Windows console: python -X utf8.
usage: transcribe_ref.py <vocal stem> [--out lines.json] [--from S] [--to S] [--window 20] [--hop 10] [--merge-into lines.json]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

import take_grid as tg  # noqa: E402
from score_lines import SR, cjk, load_audio, load_model  # noqa: E402

HALLUCINATIONS = ('字幕', '订阅', '点赞', '打赏', '独播剧场', '明镜', '请不吝', '谢谢观看', '感谢观看', 'Television', 'Subtitles', 'YoYo',
                  '词曲', '作词', '作曲', '编曲', '演唱', '混音', '制作人')   # song credits: Whisper writes them over hums and intros


def markers(expected=''):
    """Hallucination markers, minus any that really are in the expected lyric."""
    return tuple(h for h in HALLUCINATIONS if h not in expected)


CLAUSE_END = re.compile(r'[，。！？、,.!?；;：:\s]$')


def clean(seg, w0, marks=HALLUCINATIONS):
    """The segment with invented clauses cut out, or None. Times on the stem's clock. Inside a clause that holds a marker,
    the words before the marker stay and everything from the marker on is cut (a credit line runs straight on into a name,
    with no punctuation in between)."""
    words = [[round(w0 + q.start, 2), round(w0 + q.end, 2), q.word] for q in (seg.words or [])]
    text = seg.text.strip()
    if not any(h in text for h in marks):
        kept = words
    elif not words:
        return None
    else:
        clauses, cur = [], []
        for w in words:
            cur.append(w)
            if CLAUSE_END.search(w[2]):
                clauses.append(cur)
                cur = []
        if cur:
            clauses.append(cur)
        kept = []
        for c in clauses:
            txt = ''.join(x[2] for x in c)
            hit = [h for h in marks if h in txt]
            if not hit:
                kept += c
                continue
            pos = min(txt.find(h) for h in hit)                  # keep the real words before the marker in the same clause
            n = 0
            for x in c:
                if n + len(x[2]) <= pos:
                    kept.append(x)
                n += len(x[2])
        text = ''.join(w[2] for w in kept).strip()
    if not cjk(text):
        return None
    start = kept[0][0] if kept else round(w0 + seg.start, 2)
    end = kept[-1][1] if kept else round(w0 + seg.end, 2)
    return {'start': start, 'end': end, 'text': text, 'no_speech': round(seg.no_speech_prob, 2), 'words': kept}


def overlap(a, b):
    return max(0., min(a['end'], b['end']) - max(a['start'], b['start']))


def transcribe(stem, t0=0., t1=None, window=20., model=None, expected=''):
    """Windowed transcription of a vocal stem. Returns (segments, window starts, segments dropped, voice runs).
    expected: the lyric, if known; hallucination markers that really are in it are not cut."""
    x = load_audio(stem)
    dur = len(x) / SR
    t1 = min(dur, t1 or dur)
    v = tg.load_mono(stem)
    runs = [(max(t0, s), min(t1, e)) for s, e in tg.voice_runs(v) if e > t0 and s < t1]
    fps = 50
    hop = tg.SR // fps
    env = np.convolve(tg.frame_rms(v, hop), np.ones(10) / 10, 'same')     # 20 ms blocks, smoothed over 0.2 s
    thr = .1 * np.percentile(env, 95)
    starts = []
    t = max(t0, runs[0][0] - .3) if runs else t0
    while runs and t < t1 - 2:
        starts.append(t)
        target = t + window - 6
        if target >= t1:
            break
        lo, hi = max(t + 6, target - 5), target            # the next window starts at the quietest moment here: a breath
        k0, k1 = int(lo * fps), max(int(lo * fps) + 1, int(hi * fps))
        nxt = lo + int(np.argmin(env[k0:k1])) / fps - .2 if k0 < len(env) else t1
        ahead = [s for s, _ in runs if s >= nxt]
        if ahead and env[int(nxt * fps):int((nxt + 2) * fps)].max(initial=0) < thr and ahead[0] > nxt + 2:
            nxt = ahead[0] - .3                              # no voice for a while: start at the next entry
        elif not ahead and env[int(nxt * fps):].max(initial=0) < thr:
            break
        t = max(nxt, t + 4)
    if model is None:
        model, _ = load_model()
    marks = markers(expected)
    cands, dropped, edges = [], 0, []
    for w0 in starts:
        w1 = min(t1, w0 + window)
        edges += [w0, w1]
        segs, _ = model.transcribe(x[int(w0 * SR):int(w1 * SR)], language='zh', word_timestamps=True, beam_size=5,
                                   condition_on_previous_text=False, vad_filter=False)
        for s in segs:
            r = clean(s, w0, marks)
            if r is None:
                dropped += 1
                continue
            r['_central'] = min(r['start'] - w0, w1 - r['end'])
            cands.append(r)
    kept = []
    for r in sorted(cands, key=lambda r: -r['_central']):
        if all(overlap(r, k) < .3 * max(.01, r['end'] - r['start']) for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: r['start'])
    merged = []
    for r in kept:
        r.pop('_central', None)
        p = merged[-1] if merged else None
        near_edge = p is not None and any(abs(e - p['end']) < .4 or abs(e - r['start']) < .4 for e in edges)
        if p is not None and r['start'] - p['end'] < .25 and near_edge:
            p['end'], p['text'], p['words'] = r['end'], p['text'] + r['text'], p['words'] + r['words']
        else:
            merged.append(r)
    return merged, starts, dropped, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stem')
    ap.add_argument('--out')
    ap.add_argument('--from', dest='t0', type=float, default=0.)
    ap.add_argument('--to', dest='t1', type=float)
    ap.add_argument('--window', type=float, default=20.)
    ap.add_argument('--hop', type=float, default=10.)
    ap.add_argument('--merge-into')
    a = ap.parse_args()
    out_path = a.out or a.merge_into or 'lines.json'
    merged, starts, dropped, runs = transcribe(a.stem, a.t0, a.t1, a.window)
    if a.merge_into:
        base = json.load(open(a.merge_into, encoding='utf-8'))
        added = [r for r in merged if all(overlap(r, k) < .3 * max(.01, r['end'] - r['start']) for k in base)]
        print(f'merging: {len(base)} existing + {len(added)} new')
        merged = sorted(base + added, key=lambda r: r['start'])
    for r in merged:
        print(f"{r['start']:7.2f} {r['end']:7.2f}  {r['text']}", flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f'{len(merged)} segments -> {out_path} ({len(starts)} windows, {dropped} segments dropped as invented or empty)')
    for s, e in runs:
        if e - s > 2 and sum(overlap({'start': s, 'end': e}, r) for r in merged) < .5 * (e - s):
            print(f'  voice {s:.1f}-{e:.1f} s is not covered: transcribe_ref.py {a.stem} --from {max(0, s - 1):.0f} --to {e + 1:.0f} --merge-into {out_path}')


if __name__ == '__main__':
    main()
