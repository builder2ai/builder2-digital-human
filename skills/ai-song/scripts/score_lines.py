"""How well a vocal stem carries the expected lyric lines: Whisper is forced to the expected text and gives a probability and
a time for every character. It is evidence from the signal; it helps an ear, it does not replace one.

Also a library for the other scripts: load_model, load_audio, align, score_window, transcribe, cjk.
Model: faster-whisper, WHISPER_MODEL (default large-v3-turbo; a local folder also works), WHISPER_DEVICE (default cpu).
usage: score_lines.py <vocal file> <start s> <end s> "<line 1>|<line 2>|..."
"""
import sys
sys.dont_write_bytecode = True
import json
import os
import re
import subprocess
import sys

import numpy as np

SR = 16000
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def load_model():
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer
    device = os.environ.get('WHISPER_DEVICE', 'cpu')
    model = WhisperModel(os.environ.get('WHISPER_MODEL', 'large-v3-turbo'), device=device,
                         compute_type='int8' if device == 'cpu' else 'float16', cpu_threads=6)
    return model, Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task='transcribe', language='zh')


def load_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def cjk(text):
    return ''.join(re.findall(r'[㐀-䶿一-鿿]', text))


def align(model, tokenizer, audio, text):
    from faster_whisper.audio import pad_or_trim
    features = model.feature_extractor(audio)
    frames = features.shape[-1]
    encoded = model.encode(pad_or_trim(features))
    words = model.find_alignment(tokenizer, [tokenizer.encode(text)], encoded, frames)[0]
    return [{'text': w['word'], 'start': float(w['start']), 'end': float(w['end']), 'p': float(w['probability'])} for w in words]


def score_window(model, tokenizer, audio, start, end, lines):
    """Lines aligned together inside audio[start:end]: a score, a span and per-character detail for each line."""
    words = align(model, tokenizer, audio[int(start * SR):int(end * SR)], ''.join(cjk(l) for l in lines))
    chars = []
    for w in words:
        c = cjk(w['text'])
        for k, ch in enumerate(c):
            step = (w['end'] - w['start']) / len(c)
            chars.append({'c': ch, 'start': start + w['start'] + k * step, 'end': start + w['start'] + (k + 1) * step, 'p': max(w['p'], 1e-4)})
    rows, cursor = [], 0
    for line in lines:
        n = len(cjk(line))
        part = chars[cursor:cursor + n]
        cursor += n
        if not part:
            rows.append({'line': line, 'score': 0.})
            continue
        rows.append({'line': line, 'score': round(float(np.exp(np.mean([np.log(c['p']) for c in part]))), 3),
                     'worst': round(min(c['p'] for c in part), 3), 'start': round(part[0]['start'], 2), 'end': round(part[-1]['end'], 2),
                     'chars': [(c['c'], round(c['p'], 2)) for c in part]})
    return rows


def transcribe(model, audio, start, end):
    stream, _ = model.transcribe(audio[int(start * SR):int(end * SR)], language='zh', beam_size=5, vad_filter=False,
                                 condition_on_previous_text=False)
    return cjk(''.join(s.text for s in stream))


def main():
    model, tokenizer = load_model()
    audio = load_audio(sys.argv[1])
    start, end = float(sys.argv[2]), float(sys.argv[3])
    print(json.dumps(score_window(model, tokenizer, audio, start, end, sys.argv[4].split('|')), ensure_ascii=False, indent=1))
    print('unprompted:', transcribe(model, audio, start, end))


if __name__ == '__main__':
    main()
