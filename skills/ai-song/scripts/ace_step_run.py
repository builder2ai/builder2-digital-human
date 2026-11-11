#!/usr/bin/env python3
"""ACE-Step 1.5 (XL SFT by default) from a spec file: a whole song from lyrics and a caption (text2music), new bars written
next to audio you already have (repaint: a negative repainting_start writes before it, an end past its length writes after
it, a span inside it rewrites those bars), a cover of a backing track, or one instrument added over one (lego).

Models are loaded once per run. Run it alone on the GPU: the XL model needs about 20 GB.
spec.json:
  {"bpm": 99, "keyscale": "D minor", "duration": 30, "lm_negative_prompt": "",
   "candidates": [
     {"id": "song_s1", "seed": 1, "caption": "English style text", "lyrics": "[verse]\\n...", "vocal_language": "zh"},
     {"id": "intro_s1", "seed": 1, "task": "repaint", "src_audio": "src/backing.wav", "caption": "...",
      "repainting_start": -9.7, "repainting_end": 0}]}
  lyrics "[Instrumental]" (the default) makes an instrumental; otherwise [spoken] sections and @ notes are stripped
  before the lyric is sent (lyrics_tools.model_lyrics). Paths in src_audio are relative to the spec file.
Writes <spec folder>/raw/<id>.wav and records/<id>.json (generation seconds, without loading).
Environment, set here before ACE-Step is imported (an install that worked before may depend on all of it):
- offline: HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1, so nothing is downloaded mid-run (--online turns it off);
- --runtime DIR (or ACESTEP_RUNTIME): the install's own cache folder. HF_HOME=DIR/hf_cache, HF_MODULES_CACHE=DIR/hf_modules,
  TRITON_CACHE_DIR=DIR/triton, TORCH_HOME=DIR/torch, XDG_CACHE_HOME=DIR/xdg, NUMBA_CACHE_DIR=DIR/numba, TMPDIR=DIR. With
  offline set, files the install fetched earlier are found only under the HF_HOME it used then;
- CPATH: torch sends some matrix products through Triton even without compiling the model, and Triton compiles a small C
  helper that needs Python.h. The headers of this Python are used if they exist; otherwise give --python-headers DIR (or
  ACESTEP_PY_HEADERS), a folder holding python3.X/Python.h and <arch>/python3.X/pyconfig.h (for example a python3.X-dev
  package unpacked there). Without them it fails at the first generation, after the models have loaded.
- ACESTEP_CHECKPOINTS_DIR=--checkpoints, TOKENIZERS_PARALLELISM=false, PYTHONDONTWRITEBYTECODE=1.
It prints what it set.
usage: ace_step_run.py <spec.json> [id ...] --repo <ACE-Step-1.5 checkout> --checkpoints <dir> [--dit NAME] [--lm NAME]
       [--runtime DIR] [--python-headers DIR] [--online]
       (or set ACESTEP_REPO, ACESTEP_CHECKPOINTS, ACESTEP_RUNTIME, ACESTEP_PY_HEADERS)
Written against the ACE-Step 1.5 Python API as we used it; check the parameter names against your checkout.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('spec')
ap.add_argument('ids', nargs='*')
ap.add_argument('--repo', default=os.environ.get('ACESTEP_REPO'))
ap.add_argument('--checkpoints', default=os.environ.get('ACESTEP_CHECKPOINTS'))
ap.add_argument('--dit', default='acestep-v15-xl-sft')
ap.add_argument('--lm', default='acestep-5Hz-lm-4B')
ap.add_argument('--device', default='cuda')
ap.add_argument('--runtime', default=os.environ.get('ACESTEP_RUNTIME'), help="the install's cache folder (HF, Triton, torch caches)")
ap.add_argument('--python-headers', default=os.environ.get('ACESTEP_PY_HEADERS'), help='a folder with python3.X/Python.h for Triton')
ap.add_argument('--online', action='store_true', help='allow Hugging Face downloads')
args = ap.parse_args()
if not args.repo or not args.checkpoints:
    sys.exit('Give --repo and --checkpoints (or ACESTEP_REPO / ACESTEP_CHECKPOINTS).')
SPEC = Path(args.spec).resolve()
HERE = SPEC.parent
spec = json.loads(SPEC.read_text(encoding='utf-8'))


def setup_env(a):
    import sysconfig
    import tempfile
    env = {'ACESTEP_CHECKPOINTS_DIR': str(a.checkpoints), 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}
    if not a.online:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if a.runtime:
        w = Path(a.runtime)
        for name, sub in (('HF_HOME', 'hf_cache'), ('HF_MODULES_CACHE', 'hf_modules'), ('TRITON_CACHE_DIR', 'triton'),
                          ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'), ('NUMBA_CACHE_DIR', 'numba')):
            (w / sub).mkdir(parents=True, exist_ok=True)
            env[name] = str(w / sub)
        env['TMPDIR'] = str(w)
        tempfile.tempdir = str(w)
    ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    if a.python_headers:
        root = Path(a.python_headers)
        dirs = [d for d in [root / ver, *root.glob(f'*/{ver}'), root] if d.is_dir()]
        if not any((d / 'Python.h').is_file() for d in dirs):
            sys.exit(f'--python-headers {root}: no {ver}/Python.h under it')
    else:
        paths = sysconfig.get_paths()
        dirs = [Path(p) for p in dict.fromkeys((paths.get('include'), paths.get('platinclude'))) if p]
        if not any((d / 'Python.h').is_file() for d in dirs):
            print(f'warning: no Python.h for this Python ({ver}); Triton may fail at the first generation. '
                  f'Install the {ver}-dev package or give --python-headers', flush=True)
            dirs = []
    if dirs:
        env['CPATH'] = os.pathsep.join([str(d) for d in dirs] + ([os.environ['CPATH']] if os.environ.get('CPATH') else []))
    os.environ.update(env)
    print('environment:', ', '.join(f'{k}={v}' for k, v in env.items()), flush=True)


setup_env(args)
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from lyrics_tools import model_lyrics  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from acestep.constants import TASK_INSTRUCTIONS  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402


def save(item, path):
    t = item['tensor']
    a = t.detach().float().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t, np.float32)
    a = a[:, None] if a.ndim == 1 else (a.T if a.shape[0] in (1, 2) else a)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), a, int(item.get('sample_rate', 48000)), subtype='FLOAT')


todo = [c for c in spec['candidates'] if (not args.ids or c['id'] in args.ids) and not (HERE / 'raw' / f"{c['id']}.wav").exists()]
if not todo:
    sys.exit('nothing to do')
t0 = time.monotonic()
dit = AceStepHandler()
status, ok = dit.initialize_service(project_root=str(args.repo), config_path=args.dit, device=args.device, use_flash_attention=False,
                                    compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None,
                                    use_mlx_dit=False, vae_checkpoint='official')
if not ok:
    sys.exit(status)
lm = LLMHandler()
needs_lm = any(not c.get('src_audio') for c in todo)
if needs_lm:
    status, ok = lm.initialize(checkpoint_dir=str(args.checkpoints), lm_model_path=args.lm, backend='pt', device=args.device, offload_to_cpu=False)
    if not ok:
        sys.exit(status)
print('models loaded', round(time.monotonic() - t0, 1), 's', flush=True)
(HERE / 'records').mkdir(exist_ok=True)
try:
    for c in todo:
        seed = int(c['seed'])
        lyrics = c.get('lyrics', '[Instrumental]')
        bed = lyrics.strip() == '[Instrumental]'
        if not bed:
            lyrics = model_lyrics(lyrics, c.get('lines_per_row', spec.get('lines_per_row', 1)))
        common = dict(caption=c['caption'], lyrics=lyrics, instrumental=bed, bpm=int(c.get('bpm', spec['bpm'])),
                      keyscale=c.get('keyscale', spec['keyscale']), timesignature='4',
                      vocal_language='unknown' if bed else c.get('vocal_language', spec.get('vocal_language', 'zh')),
                      duration=float(c.get('duration', spec['duration'])), seed=seed, inference_steps=int(c.get('inference_steps', 50)),
                      guidance_scale=float(c.get('guidance_scale', 7.0)), shift=1.0, infer_method='ode', sampler_mode='euler',
                      use_cot_metas=False, use_cot_caption=False, use_cot_language=False, use_cot_lyrics=False, use_adg=False,
                      cfg_interval_start=0.0, cfg_interval_end=1.0, dcw_enabled=False, enable_normalization=False,
                      fade_in_duration=0.0, fade_out_duration=0.0, latent_shift=0.0, latent_rescale=1.0, flow_edit_morph=False,
                      retake_variance=0.0)
        task, src = c.get('task'), c.get('src_audio')
        if task:
            instruction = TASK_INSTRUCTIONS['lego'].format(TRACK_NAME=c['track'].upper()) if task == 'lego' else TASK_INSTRUCTIONS[task]
            params = GenerationParams(task_type=task, instruction=instruction, src_audio=str(HERE / src), reference_audio=None, audio_codes='',
                                      audio_cover_strength=1.0, cover_noise_strength=0.0, thinking=False,
                                      repainting_start=float(c['repainting_start']), repainting_end=float(c['repainting_end']),
                                      chunk_mask_mode='explicit', repaint_mode=c.get('repaint_mode', 'conservative' if task == 'repaint' else 'aggressive'),
                                      **common)
        elif src:
            params = GenerationParams(task_type='cover', instruction=TASK_INSTRUCTIONS['cover'], src_audio=str(HERE / src), reference_audio=None,
                                      audio_codes='', audio_cover_strength=float(c['audio_cover_strength']),
                                      cover_noise_strength=float(c.get('cover_noise_strength', 0.)), thinking=False, **common)
        else:
            params = GenerationParams(task_type='text2music', instruction=TASK_INSTRUCTIONS['text2music'], src_audio=None, reference_audio=None,
                                      audio_codes='', audio_cover_strength=1.0, cover_noise_strength=0.0, thinking=True,
                                      use_constrained_decoding=True, lm_temperature=float(c.get('lm_temperature', .85)),
                                      lm_cfg_scale=float(c.get('lm_cfg_scale', 2.)), lm_top_p=float(c.get('lm_top_p', .9)), lm_top_k=0,
                                      lm_negative_prompt=c.get('lm_negative_prompt', spec.get('lm_negative_prompt', '')), **common)
        config = GenerationConfig(batch_size=1, allow_lm_batch=False, use_random_seed=False, seeds=[seed], audio_format='wav32', lm_batch_chunk_size=1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        begin = time.monotonic()
        result = generate_music(dit_handler=dit, llm_handler=lm, params=params, config=config, save_dir=None)
        if not result.success or len(result.audios) != 1:
            raise RuntimeError(result.error or 'expected one audio')
        save(result.audios[0], HERE / 'raw' / f"{c['id']}.wav")
        seconds = round(time.monotonic() - begin, 1)
        (HERE / 'records' / f"{c['id']}.json").write_text(json.dumps({'candidate': c, 'seconds': seconds}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(c['id'], seconds, 's', flush=True)
finally:
    if needs_lm:
        lm.unload()
    for name in ('model', 'vae', 'text_encoder', 'text_tokenizer'):
        if hasattr(dit, name):
            setattr(dit, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
