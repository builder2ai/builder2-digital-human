"""Which reading did the singer give a character that has more than one (宿 xiǔ or sù, 便宜 pián or biàn)?

A free transcript cannot tell: Whisper writes the character itself. So Whisper is forced to the line several times with a
different single-reading character in that place (修 = xiu, 速 = su, 数 = shu), and the probability it gives that one
character is compared. Below 0.1 for all of them counts as unclear.
usage: forced_reading.py <vocal stem> <start s> <end s> --line "他问东坡你这夜里忙什么忙我说借你半{}陪我晒月亮" --try 修=xiu 速=su 数=shu [more stems...]
  {} marks the character being tested; write the line without punctuation; the window start/end should hold that line
  and not much else.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json

from score_lines import SR, align, cjk, load_audio, load_model

ap = argparse.ArgumentParser()
ap.add_argument('stem')
ap.add_argument('start', type=float)
ap.add_argument('end', type=float)
ap.add_argument('more', nargs='*', help='more stems, same window')
ap.add_argument('--line', required=True)
ap.add_argument('--try', dest='tries', nargs='+', required=True, help='character=label pairs')
a = ap.parse_args()
k = len(cjk(a.line.split('{}')[0]))
tries = [t.split('=') for t in a.tries]
m, tok = load_model()
out = {}
for path in [a.stem] + a.more:
    w = load_audio(path)[int(a.start * SR):int(a.end * SR)]
    res = {}
    for ch, label in tries:
        chars = []
        for word in align(m, tok, w, a.line.format(ch)):
            c = cjk(word['text'])
            for i, x in enumerate(c):
                step = (word['end'] - word['start']) / len(c)
                chars.append((x, a.start + word['start'] + i * step, word['p']))
        x, t, p = chars[k]
        res[label] = {'p': round(p, 3), 'at': round(t, 2)}
    best = max(res, key=lambda r: res[r]['p'])
    out[path] = {**res, 'verdict': best if res[best]['p'] >= .1 else 'unclear'}
    print(path, json.dumps(out[path], ensure_ascii=False), flush=True)
