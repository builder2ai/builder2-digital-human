"""The lyrics file is the one source of truth for every word in the song, spoken lines included.

Spoken lines (念白) sit in their own [spoken] section. A music model sings everything it is given, so they never go to it:
model_lyrics() strips [spoken] sections and @ note lines before a lyric is sent, and render_yue2.py and ace_step_run.py
call it themselves. Inside a [spoken] section, lines starting with @ are notes (who says it, the voice, where it goes).
An instrumental section for YuE2 is a bare [Instrumental] tag with nothing under it.

usage: lyrics_tools.py lyrics.txt [--per-row 2 | --per-row verse=2,chorus=1]   prints what the music model gets, the spoken
                                                  lines, and lines/characters per section
"""
import re
import sys

sys.dont_write_bytecode = True

TAG = re.compile(r'^\s*\[([A-Za-z][A-Za-z0-9 _-]*)\]\s*$')


def sections(text):
    """[(tag or '', [lines])] in order."""
    out = [('', [])]
    for line in text.lstrip('﻿').splitlines():
        m = TAG.match(line)
        if m:
            out.append((m.group(1).strip(), []))
        else:
            out[-1][1].append(line)
    return [(t, ls) for t, ls in out if t or any(l.strip() for l in ls)]


PUNCT = '，。！？、,.!?；;：:'


def rows_for(per_row, tag):
    """Lines per row for one section. per_row is a number for every section, or per section: {"verse": 2, "chorus": 1,
    "default": 1} (also as the string "verse=2,chorus=1"). A key matches the tag itself ("verse 2") or its leading word
    ("verse" matches [verse], [verse 2], [verse_b])."""
    if isinstance(per_row, str):
        per_row = int(per_row) if per_row.strip().isdigit() else {
            k.strip().lower(): int(v) for k, v in (p.split('=') for p in per_row.split(',') if '=' in p)}
    if not isinstance(per_row, dict):
        return int(per_row or 1)
    t = tag.lower().strip()
    lead = re.match(r'[a-z]+', t)
    for key in (t, lead.group(0) if lead else None, 'default', '*'):
        if key in per_row:
            return int(per_row[key])
    return 1


def model_lyrics(text, per_row=1):
    """The lyric a music model gets: no [spoken] sections, no @ notes, one blank line between sections.
    per_row=2 puts two lyric lines on one row; {"verse": 2, "chorus": 1} does it per section. YuE2 sings about one row per
    phrase; in one test it sang every short verse line over about 2 bars whatever the style text said, and two lines a row
    brought the verse to about one line a bar. The lyrics file itself stays one line a row."""
    parts = []
    for tag, lines in sections(text):
        if tag.lower() == 'spoken':
            continue
        body = [l.strip() for l in lines if l.strip() and not l.lstrip().startswith('@')]
        n = rows_for(per_row, tag)
        if n > 1:
            rows = []
            for i in range(0, len(body), n):
                chunk = body[i:i + n]
                rows.append(''.join(l if (j == len(chunk) - 1 or l[-1] in PUNCT) else l + '，' for j, l in enumerate(chunk)))
            body = rows
        parts.append('\n'.join(([f'[{tag}]'] if tag else []) + body))
    return '\n\n'.join(p for p in parts if p.strip()) + '\n'


def spoken(text):
    return [{'notes': [l.strip()[1:].strip() for l in ls if l.strip().startswith('@')],
             'lines': [l.strip() for l in ls if l.strip() and not l.strip().startswith('@')]}
            for tag, ls in sections(text) if tag.lower() == 'spoken']


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    text = open(sys.argv[1], encoding='utf-8').read()
    per_row = sys.argv[sys.argv.index('--per-row') + 1] if '--per-row' in sys.argv else 1
    print('--- to the music model ---')
    print(model_lyrics(text, per_row))
    print('--- spoken (to a speech model) ---')
    for s in spoken(text):
        print(' notes:', '; '.join(s['notes']))
        for l in s['lines']:
            print(' ', l)
    print('--- per section: lines, characters per line ---')
    for tag, ls in sections(text):
        body = [l for l in ls if l.strip() and not l.lstrip().startswith('@')]
        print(f"  [{tag or '-'}] {len(body)} lines", [sum(1 for c in l if '一' <= c <= '鿿') for l in body])
