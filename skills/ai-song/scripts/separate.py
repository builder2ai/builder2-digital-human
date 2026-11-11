"""Split audio files with python-audio-separator, loading the model once.

--model vocals (default): a BS-RoFormer model, <name>_vocals.flac and <name>_instrumental.flac. For scoring words and
  transcribing, and for judging a take's accompaniment.
--model parts: Demucs htdemucs_ft (MIT), <name>_voice / _drums / _bass / _other.wav. For a reference song's layers (which
  instruments play where), and for setting each part's level in an ending. Demucs often files a lead instrument (a flute,
  an erhu) under voice.
Any input format works: everything that is not WAV or FLAC is decoded with ffmpeg first (librosa 1.0 and later, which
audio-separator uses, reads only what libsndfile reads, so .m4a/.aac/.mp4 fail otherwise).
--start S cuts the first S seconds off before separating, so the stems start there; --end E stops at E seconds (of the
  original). A word-fix candidate only needs a few seconds around the patch: separation took 13.7 of 23 GPU minutes in one
  test, about 70 s for each 62-s take. Write window stems to their own --out folder (names are the same as full ones), and
  give comp_vocal.py the window's start as the patch's clip start.
Demucs in audio-separator imports diffq even for unquantized models. If diffq is missing, the stub in stubs/diffq is used
for --model parts (it raises if a quantized model ever really needs it).
Time: a 3:43 song took about 5.4 min (vocals) and 1.3 min (parts) on a shared GB10. Run long jobs detached, e.g.
  setsid nohup python separate.py ... > sep.log 2>&1 < /dev/null &
so they survive a dropped SSH session.
Models are downloaded into AUDIO_SEPARATOR_MODELS (default ~/models/audio-separator) on first use.
usage: separate.py [--model vocals|parts] [--out stems] [--start S] [--end E] file [file ...]
"""
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODELS = {'vocals': ('model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'FLAC', {'Vocals': 'vocals', 'Instrumental': 'instrumental'}),
          'parts': ('htdemucs_ft.yaml', 'WAV', {'Vocals': 'voice', 'Drums': 'drums', 'Bass': 'bass', 'Other': 'other'})}

ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+')
ap.add_argument('--model', choices=MODELS, default='vocals')
ap.add_argument('--out', default='stems')
ap.add_argument('--start', type=float)
ap.add_argument('--end', type=float)
a = ap.parse_args()
if a.end is not None and a.end <= (a.start or 0.):
    ap.error('--end must come after --start')
if a.model == 'parts' and importlib.util.find_spec('diffq') is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'stubs'))
    print('diffq not installed: using the import stub in stubs/diffq (fine for htdemucs_ft)', flush=True)

from audio_separator.separator import Separator  # noqa: E402

model, fmt, names = MODELS[a.model]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
sep = Separator(model_file_dir=os.environ.get('AUDIO_SEPARATOR_MODELS', str(Path.home() / 'models/audio-separator')),
                output_dir=str(out), output_format=fmt)
sep.load_model(model)
for f in a.files:
    name = Path(f).stem
    ext = '.flac' if fmt == 'FLAC' else '.wav'
    if (out / f"{name}_{next(iter(names.values()))}{ext}").exists():
        print('exists', name)
        continue
    src = f
    with tempfile.TemporaryDirectory() as tmp:
        if a.start or a.end is not None or Path(f).suffix.lower() not in ('.wav', '.flac'):
            src = str(Path(tmp) / f'{name}.flac')
            cut = (['-ss', str(a.start)] if a.start else []) + (['-t', str(a.end - (a.start or 0.))] if a.end is not None else [])
            cmd = ['ffmpeg', '-v', 'error', '-y'] + cut + ['-i', f, src]
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        sep.separate(src, {k: f'{name}_{v}' for k, v in names.items()})
    print('separated', name, flush=True)
