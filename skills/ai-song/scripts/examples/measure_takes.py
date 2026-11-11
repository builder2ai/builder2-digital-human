"""EXAMPLE from our YuE2 vs ACE-Step comparison (needs librosa, soundfile). File naming is that comparison's:
<work_dir>/stems/<take>_vocals.flac and _instrumental.flac, takes whose id starts with 'bed' are instrumentals,
ACE-Step records in <work_dir>/ace/records, YuE2 records (render_yue2.py) in <records dir> named generation_GCyue_*.json.
- every vocal take: tempo of its instrumental stem, length, vocal-stem share of the mix;
- every bed: level per second (steadiness, longest quiet stretch), vocal-stem level against the mix, energy above 6 kHz
  (hats and drums), onset rate (how busy), tempo.
Also gathers the generation seconds of both models.
usage: measure_takes.py <work_dir> <yue2 records dir>
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import sys

import librosa
import numpy as np
import soundfile as sf

wk, rec = Path(sys.argv[1]), Path(sys.argv[2])
st = wk / 'stems'


def mono(path, sr=None):
    y, s = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1)
    if sr and s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
        s = sr
    return y.astype(np.float32), s


def db(x):
    return float(20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9)))


def tempo(y, sr):
    t, _ = librosa.beat.beat_track(y=y, sr=sr)
    return round(float(np.atleast_1d(t)[0]), 1)


out = {'takes': {}, 'beds': {}, 'seconds': {}}
for f in sorted(st.glob('*_instrumental.flac')):
    tid = f.name[:-len('_instrumental.flac')]
    inst, sr = mono(f, 22050)
    voc, _ = mono(st / f'{tid}_vocals.flac', 22050)
    n = min(len(inst), len(voc))
    if tid.startswith('bed'):
        mix = inst[:n] + voc[:n]
        sec = [db(mix[k * sr:(k + 1) * sr]) for k in range(n // sr)]
        core = sec[3:-3] if len(sec) > 10 else sec
        quiet, run, longest = -45.0, 0, 0
        for v in sec:
            run = run + 1 if v < quiet else 0
            longest = max(longest, run)
        S = np.abs(librosa.stft(mix, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hf = float(S[freqs > 6000].sum() / max(S.sum(), 1e-9))
        onsets = librosa.onset.onset_detect(y=mix, sr=sr, units='time')
        out['beds'][tid] = {'seconds_long': round(n / sr, 1), 'level_db': round(db(mix), 1),
                            'level_sd_db': round(float(np.std(core)), 2), 'level_range_db': round(float(np.ptp(core)), 1),
                            'longest_quiet_s': longest, 'vocal_vs_mix_db': round(db(voc[:n]) - db(mix), 1),
                            'hf_above_6k_share': round(hf, 4), 'onsets_per_s': round(len(onsets) / (n / sr), 2),
                            'tempo': tempo(mix, sr), 'per_second_db': [round(v, 1) for v in sec]}
    else:
        out['takes'][tid] = {'seconds_long': round(n / sr, 1), 'tempo': tempo(inst, sr),
                             'vocal_vs_mix_db': round(db(voc[:n]) - db(inst[:n] + voc[:n]), 1)}
for r in sorted((wk / 'ace' / 'records').glob('*.json')):
    d = json.loads(r.read_text(encoding='utf-8'))
    out['seconds'][r.stem] = d['seconds']
for r in sorted(rec.glob('generation_GCyue_*.json')):
    d = json.loads(r.read_text(encoding='utf-8'))
    out['seconds'][d['id']] = d['render_seconds']
(wk / 'measure.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: {t: {kk: vv for kk, vv in v.items() if kk != 'per_second_db'} for t, v in d.items()} if k != 'seconds' else d
                  for k, d in out.items()}, ensure_ascii=False))
