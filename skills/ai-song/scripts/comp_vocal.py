"""Patch re-sung phrases into a vocal stem. Everything outside the patches stays the base take's audio, sample for sample.

Each patch is the vocal stem of a take continued from the base take's tokens (render_yue2.py keep_from), so its syllables
already sit on the base take's timeline. Give the join times exactly: pick them in the energy valleys and check them by
transcribing each side, because a valley can fall inside a word. Patches that meet on one time crossfade into each other.
After patching, judge the word again inside the full mix: a patch does not always win against the accompaniment.

comp.json: {"name": "verse2_fixed", "base": "stems/A_s1_vocals.flac",
            "patches": [["stems/A_fix_s7_vocals.flac", 30.0, 37.40, 37.95]]}
  a patch = [stem file, the base-take time where that stem starts, start s, end s]; paths relative to comp.json.
Writes <name>_vocals.wav and <name>_patches.json next to comp.json (or into --out).
usage: comp_vocal.py <comp.json> [--out DIR]
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

import numpy as np

from songmap import SR, load, write

FADE = .03


def active_rms(x):
    m = x.mean(axis=1)
    n = SR // 20
    f = np.array([np.sqrt((m[i:i + n] ** 2).mean()) for i in range(0, len(m) - n, n)])
    f = f[f > .25 * f.max()]
    return float(np.sqrt((f ** 2).mean()))


ap = argparse.ArgumentParser()
ap.add_argument('comp')
ap.add_argument('--out')
a = ap.parse_args()
C = Path(a.comp).resolve()
comp = json.loads(C.read_text(encoding='utf-8'))
out_dir = Path(a.out) if a.out else C.parent
base = load(C.parent / comp['base'])
out = base.copy()
report = []
patches = comp['patches']
for i, (stem, clip_start, start, end) in enumerate(patches):
    raw = load(C.parent / stem)
    patch = np.zeros_like(base)
    head = int(round(clip_start * SR))
    n = min(len(raw), len(base) - head)
    patch[head:head + n] = raw[:n]
    ia, ib = int(start * SR), int(end * SR)
    gain = active_rms(base[ia:ib]) / active_rms(patch[ia:ib])
    fade = int(FADE * SR)
    weight = np.zeros(len(base))
    weight[ia:ib] = 1
    weight[ia - fade // 2:ia + fade // 2] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    meets_next = i + 1 < len(patches) and patches[i + 1][2] == end
    weight[ib - fade // 2:ib + fade // 2] = 1 if meets_next else np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    w = weight[:, None]
    out = np.sqrt(1 - w) * out + np.sqrt(w) * patch * gain
    report.append({'stem': stem, 'start': start, 'end': end, 'gain_db': round(20 * np.log10(gain), 2)})
path = out_dir / f"{comp['name']}_vocals.wav"
write(out, path)
(out_dir / f"{comp['name']}_patches.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=1))
print(path)
