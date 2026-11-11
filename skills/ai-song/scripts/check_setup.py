"""Before starting: what this machine and this Python environment can do, grouped by step.

The tools need different environments (see reference/models.md), so run this once with each Python you plan to use, on
each machine, and read only the groups for the steps you will run there. A MISSING line in another step's group is not a
problem for this one. Nothing is installed or downloaded here.

Groups:
  separate  separating stems (audio-separator, and its Demucs path for --model parts, which also needs diffq or the stub)
  analyse   扒曲 (librosa, soundfile for analyze_ref.py; torch, transformers for tag_ref.py; harmony_groove.py needs only
            numpy and ffmpeg)
  score     transcribing and scoring words, and where each section sits in a take (faster-whisper, pypinyin:
            transcribe_ref.py, take_sections.py, score_takes.py); a CPU is fine
  take_grid.py, cut_take.py, join_takes.py, beat_check.py, assemble.py and harmony_groove.py need only numpy and ffmpeg
  (the audio group); spoken_qc.py is in the score group.
  yue2      YuE2 in ComfyUI (COMFY_URL, default http://127.0.0.1:8188, and its nodes)
  ace       ACE-Step 1.5 (ACESTEP_REPO, ACESTEP_CHECKPOINTS)
  tts       the spoken line: Qwen3-TTS VoiceDesign (the digital-human skill's voice_design.py; QWEN_TTS_DESIGN = the model
            folder). It often lives in its own venv: run this check with that Python. When qwen_tts is missing here, it
            looks for other venvs that have it and for local VoiceDesign model folders, and prints them.
  audio     ffmpeg (every script) and its rubberband filter (tempo changes, the ending slow-down)
usage: python check_setup.py [--for separate,analyse,score,yue2,ace,tts,audio]
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--for', dest='groups', default='separate,analyse,score,yue2,ace,tts,audio')
groups = ap.parse_args().groups.split(',')
print(f'machine {platform.node()} ({platform.system()} {platform.machine()}), python {sys.executable} {platform.python_version()}')
rows = []


def check(group, what, ok, note=''):
    rows.append((group, what, ok, note))


def has(mod):
    return importlib.util.find_spec(mod) is not None


if 'separate' in groups:
    check('separate', 'audio_separator', has('audio_separator'))
    if has('audio_separator'):
        stub = str(Path(__file__).resolve().parent / 'stubs')
        real = has('diffq')
        if not real:
            sys.path.insert(0, stub)
        try:
            importlib.import_module('audio_separator.separator.architectures.demucs_separator')
            check('separate', 'Demucs path (--model parts)', True, 'diffq installed' if real else 'works with the diffq stub, which separate.py adds by itself')
        except Exception as e:  # noqa: BLE001
            check('separate', 'Demucs path (--model parts)', False, f'{type(e).__name__}: {str(e)[:70]}')
    models = Path(os.environ.get('AUDIO_SEPARATOR_MODELS', str(Path.home() / 'models/audio-separator')))
    for m in ('model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'htdemucs_ft.yaml'):
        check('separate', f'model {m}', True, f'in {models}' if (models / m).exists() else f'not in {models} yet; downloaded on first use')
if 'analyse' in groups:
    for mod in ('librosa', 'soundfile', 'numpy'):
        check('analyse', mod, has(mod))
    for mod in ('torch', 'transformers'):
        check('analyse', f'{mod} (tag_ref.py)', has(mod), 'for the instrument and genre tags; any venv')
if 'score' in groups:
    for mod in ('faster_whisper', 'pypinyin', 'numpy'):
        check('score', mod, has(mod), 'take_sections.py, score_takes.py; can run on another machine than the GPU box' if mod != 'numpy' else '')
if 'audio' in groups:
    ff = shutil.which('ffmpeg')
    found = []
    if not ff:
        if os.environ.get('FFMPEG'):
            found.append(os.environ['FFMPEG'])
        if has('imageio_ffmpeg'):
            found.append(importlib.import_module('imageio_ffmpeg').get_ffmpeg_exe())
        home = Path.home()
        pats = ['tools/*/ffmpeg*', 'tools/*/*/ffmpeg*', 'tools/ffmpeg*', '.local/bin/ffmpeg*',
                '*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg*',
                'tools/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg*']
        for pat in pats:
            found += [str(p) for p in sorted(home.glob(pat)) if p.is_file() and os.access(p, os.X_OK)]
        found = list(dict.fromkeys(found))
    note = ff or ('not on PATH. Found: ' + ', '.join(found[:4]) + '. The scripts call "ffmpeg" by name: put a wrapper or link named '
                  'ffmpeg to one of these in a folder on PATH (one with the rubberband filter)' if found else 'not found')
    check('audio', 'ffmpeg on PATH', bool(ff), note)
    exe = ff or (found[0] if found else None)
    if exe:
        filters = subprocess.run([exe, '-hide_banner', '-filters'], capture_output=True, text=True).stdout
        check('audio', 'ffmpeg rubberband filter', ' rubberband ' in filters, exe)
if 'yue2' in groups:
    url = os.environ.get('COMFY_URL', 'http://127.0.0.1:8188').rstrip('/')
    try:
        info = json.load(urllib.request.urlopen(url + '/object_info', timeout=10))
        check('yue2', f'ComfyUI at {url}', True)
        for node, note in (('EmptyYuE2LatentAudio', 'a ComfyUI build with YuE2 support (see reference/models.md)'), ('VAEDecodeAudio', ''),
                           ('YuE2ContinueMusic', 'copy scripts/yue2_continue into custom_nodes/')):
            check('yue2', f'node {node}', node in info, note if node not in info else '')
    except Exception as e:  # noqa: BLE001
        check('yue2', f'ComfyUI at {url}', False, str(e)[:80])
if 'ace' in groups:
    repo, ck = os.environ.get('ACESTEP_REPO'), os.environ.get('ACESTEP_CHECKPOINTS')
    check('ace', 'ACE-Step 1.5 checkout (ACESTEP_REPO)', bool(repo) and (Path(repo) / 'acestep').is_dir(), repo or 'not set')
    check('ace', 'ACE-Step checkpoints (ACESTEP_CHECKPOINTS)', bool(ck) and Path(ck).is_dir(), ck or 'not set')
    check('ace', 'soundfile', has('soundfile'))
if 'tts' in groups:
    home = Path.home()
    q = has('qwen_tts')
    note = 'voice_design.py in the digital-human skill'
    if not q:
        def venv_of(p):
            for d in p.parents:
                if (d / 'pyvenv.cfg').exists() or (d / 'bin' / 'python').exists() or (d / 'Scripts' / 'python.exe').exists():
                    return d
            return p.parent
        envs = [venv_of(p) for pat in ('*/lib/python3*/site-packages/qwen_tts', '*/*/lib/python3*/site-packages/qwen_tts', '*/Lib/site-packages/qwen_tts',
                                       '*/*/Lib/site-packages/qwen_tts', '.*/lib/python3*/site-packages/qwen_tts')
                for p in home.glob(pat)]
        envs = list(dict.fromkeys(str(e) for e in envs))
        note = ('not in this Python. Venvs that have it: ' + ', '.join(envs[:4]) + ' (run this check with that venv\'s python)') if envs \
            else 'not in this Python, and no venv with it found under your home folder'
    check('tts', 'qwen_tts (Qwen3-TTS)', q, note)
    for mod in ('torch', 'soundfile'):
        check('tts', mod, has(mod), 'in the TTS venv')
    d = os.environ.get('QWEN_TTS_DESIGN')
    if d and Path(d).exists():
        check('tts', 'VoiceDesign model (QWEN_TTS_DESIGN)', True, d)
    else:
        found = [str(p) for pat in ('*/models/Qwen3-TTS-*VoiceDesign*', '*/*/models/Qwen3-TTS-*VoiceDesign*', 'models/Qwen3-TTS-*VoiceDesign*',
                                    '.cache/huggingface/hub/models--Qwen--Qwen3-TTS-*VoiceDesign*') for p in home.glob(pat) if p.is_dir()]
        hint = ('found: ' + ', '.join(found[:3]) + '; set QWEN_TTS_DESIGN to it (and HF_HUB_OFFLINE=1 to stay offline)') if found \
            else 'voice_design.py downloads Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign on first use (ask first)'
        check('tts', 'VoiceDesign model (QWEN_TTS_DESIGN)', False, (f'{d} is not a folder here; ' if d else 'not set; ') + hint)

w = max(len(r[1]) for r in rows) if rows else 10
last = None
for g, what, ok, note in rows:
    if g != last:
        print(f'[{g}]')
        last = g
    print(f"  {'ok     ' if ok else 'MISSING'} {what:<{w}}  {note}")
