"""Render YuE2 takes through ComfyUI with the YuE2ContinueMusic node (scripts/yue2_continue), one at a time.

Every take saves its music tokens, so any take can later be continued from a chosen time: set keep_from (an earlier take's
id) and keep_seconds, and the new take keeps that take's voice, key and groove up to there and sings on from it.
The music seed decides the tokens (tune, rhythm, words); the render seed only decides how the tokens are turned into sound
(same notes, often very different diction), and a render seed sweep is cheap.

spec.json (paths in it are as the ComfyUI machine sees them):
  {"api": "http://127.0.0.1:8190", "checkpoint": "yue2_3b_int8_convrot.safetensors",
   "output_directory": "/path/to/ComfyUI/output", "output_prefix": "mysong/", "max_duration": 60,
   "shared": {"style": "English style text", "lyrics": "[verse]\\n...\\n\\n[chorus]\\n..."},
   "variants": [{"id": "A_s1", "seed": 1},
                {"id": "A_s1_r2", "seed": 1, "render_seed": 2},
                {"id": "B_s3", "seed": 3, "keep_from": "A_s1", "keep_seconds": 31.2, "lyrics": "...whole lyric again..."}]}
A variant may override style, lyrics, max_duration, temperature, top_p, top_k, repetition_penalty, cfg_scale, lines_per_row.
[spoken] sections and @ notes are stripped from every lyric before it is sent (lyrics_tools.model_lyrics);
"lines_per_row": 2 (spec or variant) sends two lyric lines a row, so YuE2 sings them in about the time it gives one row;
{"verse": 2, "chorus": 1} does it per section (a key matches the tag or its leading word: "verse" also covers [verse 2]).
max_duration: a take ends where YuE2 stops by itself or at max_duration, whichever comes first, so set it long enough:
planned seconds of the section x 2 (it may sing each line twice as slowly as planned) + 20 s (lead-ins of up to about 20 s
were seen). At 40 s for a planned 18 s verse, 5 of 16 takes were cut off. Once bars per line are measured, planned
seconds x 1.2 + 20 s. A take that reaches max_duration is not always cut off: many sang on past the lyric (hums, ad-libs,
the whole song again; a bare [Instrumental] at the end did not stop that). take_sections.py tells which, cut_take.py
cuts after the last line.
A section continued from a chosen take: keep_from = that take, keep_seconds = the bar line after its last sung note
(take_grid.py prints the grid), lyrics = everything sung so far plus the new section. Key, tempo and voice carry over.
A whole take re-rendered on its own tokens (the first thing to try for unclear words): {"id": "X_r9", "rerender": "X",
"render_seed": 9}. It keeps every token but the last (keep_seconds = length - 0.04, max_duration = length, the length read
from the take's token file; or give "length" in seconds), samples that one token, and renders the whole take again. The
music seed defaults to the take's own (from its record). Style and lyrics must be the ones the take was made with (it
warns when the take's workflow record in this spec's records/ differs). The take's own render seed reproduces it (corr
0.9999 in one test): render it once as the control. Other render seeds change the diction and keep the timing: every
line-boundary valley fell on the same 40-ms step in six renders, so whole lines can be patched in (comp_vocal.py).
About 18 s a render in one test.
Records go to <spec folder>/records, tokens to <spec folder>/tokens (or --tokens, a folder the ComfyUI machine can write).
The script refuses to submit while the queue is busy, and never cancels anything.
usage: render_yue2.py <spec.json> [id ...] [--tokens DIR]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

from lyrics_tools import model_lyrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('spec')
ap.add_argument('ids', nargs='*')
ap.add_argument('--tokens')
args = ap.parse_args()
SPEC = Path(args.spec).resolve()
spec = json.loads(SPEC.read_text(encoding='utf-8'))
API = spec['api'].rstrip('/')
R = SPEC.parent / 'records'
T = Path(args.tokens) if args.tokens else SPEC.parent / 'tokens'
R.mkdir(exist_ok=True)
T.mkdir(exist_ok=True)
FPS = 25   # YuE2 music tokens per second (the node keeps round(keep_seconds * 25) tokens)


def call(path, data=None, timeout=60):
    body = json.dumps(data, ensure_ascii=False).encode('utf-8') if data is not None else None
    req = urllib.request.Request(API + path, data=body, headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def graph(v):
    shared = spec['shared']
    keep = str(T / f"{v['keep_from']}.json") if v.get('keep_from') else ''
    return {
        'ck': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': spec['checkpoint']}},
        'music': {'class_type': 'YuE2ContinueMusic', 'inputs': {
            'clip': ['ck', 1], 'style': v.get('style', shared['style']),
            'lyrics': model_lyrics(v.get('lyrics', shared['lyrics']), v.get('lines_per_row', spec.get('lines_per_row', 1))),
            'seed': v['seed'], 'max_duration': v.get('max_duration', spec['max_duration']),
            'temperature': v.get('temperature', 1.), 'top_p': v.get('top_p', .95), 'top_k': v.get('top_k', 100),
            'repetition_penalty': v.get('repetition_penalty', 1.2), 'cfg_scale': v.get('cfg_scale', 1.01),
            'keep_tokens_file': keep, 'keep_seconds': v.get('keep_seconds', 0.),
            'save_tokens_file': str(T / f"{v['id']}.json")}},
        'neg': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['music', 0]}},
        'lat': {'class_type': 'EmptyYuE2LatentAudio', 'inputs': {'seconds': ['music', 1], 'batch_size': 1}},
        'ks': {'class_type': 'KSampler', 'inputs': {
            'model': ['ck', 0], 'positive': ['music', 0], 'negative': ['neg', 0], 'latent_image': ['lat', 0],
            'seed': v.get('render_seed', v['seed']), 'steps': 32, 'cfg': 1., 'sampler_name': 'dpm_2',
            'scheduler': 'sgm_uniform', 'denoise': 1.}},
        'decode': {'class_type': 'VAEDecodeAudio', 'inputs': {'samples': ['ks', 0], 'vae': ['ck', 2]}},
        'save': {'class_type': 'SaveAudio', 'inputs': {'audio': ['decode', 0], 'filename_prefix': spec['output_prefix'] + v['id']}},
        'duration': {'class_type': 'PreviewAny', 'inputs': {'source': ['music', 1]}},
    }


def rerender(v):
    """A variant with "rerender": keep all of that take's tokens but the last, so only the rendering changes."""
    v = dict(v)
    src = v['rerender']
    if 'render_seed' not in v:
        raise SystemExit(f"{v['id']}: a rerender needs a render_seed (the take's own render seed reproduces it: use it as the control)")
    if 'length' in v:
        n = int(round(float(v['length']) * FPS))
    elif (T / f'{src}.json').exists():
        n = len(json.loads((T / f'{src}.json').read_text(encoding='utf-8')))
    else:
        raise SystemExit(f"{v['id']}: no token file {T / (src + '.json')} here; give \"length\" (the take's seconds) or --tokens")
    v['keep_from'], v['keep_seconds'], v['max_duration'] = src, round((n - 1) / FPS, 2), round(n / FPS, 2)
    rec = R / f'generation_{src}.json'
    if 'seed' not in v:
        if not rec.exists():
            raise SystemExit(f"{v['id']}: give \"seed\" (the music seed of {src}; its record is not in {R})")
        v['seed'] = json.loads(rec.read_text(encoding='utf-8'))['seed']
    wf = R / f'workflow_{src}.json'
    if wf.exists():
        was = json.loads(wf.read_text(encoding='utf-8'))['music']['inputs']
        now = graph(v)['music']['inputs']
        for k in ('style', 'lyrics'):
            if was.get(k) != now[k]:
                print(f"{v['id']}: the {k} differs from what {src} was made with; a re-render should use the same", flush=True)
    return v


for v in spec['variants']:
    if args.ids and v['id'] not in args.ids:
        continue
    done = R / f"generation_{v['id']}.json"
    if done.exists():
        print(v['id'], 'already rendered, skipped')
        continue
    if v.get('rerender'):
        v = rerender(v)
    q = call('/queue')
    if q.get('queue_running') or q.get('queue_pending'):
        raise SystemExit('The queue is busy (maybe someone else is using this ComfyUI). Nothing was submitted.')
    g = graph(v)
    (R / f"workflow_{v['id']}.json").write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding='utf-8')
    t = time.time()
    pid = call('/prompt', {'prompt': g, 'client_id': str(uuid.uuid4())})['prompt_id']
    while time.time() - t < 2400:
        h = call('/history/' + pid)
        if pid in h:
            break
        time.sleep(2)
    else:
        raise SystemExit('Timed out; look up prompt ' + pid + ' before retrying.')
    e = h[pid]
    status = e.get('status', {}).get('status_str')
    files = [str(Path(spec['output_directory']) / a.get('subfolder', '') / a['filename'])
             for o in e.get('outputs', {}).values() for a in o.get('audio', [])]
    rec = {'id': v['id'], 'seed': v['seed'], 'render_seed': v.get('render_seed', v['seed']), 'keep_from': v.get('keep_from'),
           'keep_seconds': v.get('keep_seconds'), 'rerender': v.get('rerender'), 'prompt_id': pid, 'status': status, 'files': files,
           'render_seconds': round(time.time() - t, 1)}
    done.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    if status != 'success':
        raise SystemExit(str(e.get('status')))
