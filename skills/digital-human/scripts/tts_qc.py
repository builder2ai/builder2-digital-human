"""Pick the best take per line: same speaker first, then the words, then pace and pitch.

1. Speaker: CAM++ similarity to the profile's reference voice must be >= 0.6. Takes fall into two
   clear groups: the same voice scores about 0.73-0.88 (short lines at the low end), a take that
   drifted into another speaker about 0.31-0.44. Short lines and lines opening with an interjection
   (「嗨」「诶」) drift most. If no take passes, the line is flagged and its most similar take is used.
2. Words: Whisper large-v3 transcribes each take; character error rate against the script, after
   dropping punctuation, spaces, case and all digits / Chinese numerals (Whisper writes 32 where
   the script says 三十二). Takes within 0.01 of the best stay in.
3. Among those: highest speaker similarity, then pace nearest 5 characters/s (14 letters/s in
   English) and median pitch nearest the reference's.
A take whose last syllable is cut off is avoided when another passes.
The picks are copied to <tts_dir>/best/L<id>.wav and strung together (0.35 s gaps) into
<tts_dir>/best/preview_all.wav. Lines flagged '<-- check' need a listen: a wrong character is
usually a polyphone (fix it in the line's "tts" field) or a homophone Whisper wrote differently.

usage: tts_qc.py <script.json> <tts_dir> <reference.wav>
env CAMPP_ONNX: path to campplus.onnx (from huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
"""
import sys, os, re, json, glob, shutil
import numpy as np, soundfile as sf, torch, torchaudio, whisper, pyworld as pw, onnxruntime as ort

script, d, ref = sys.argv[1], sys.argv[2], sys.argv[3]
CAMPP = os.path.expanduser(os.environ.get('CAMPP_ONNX', 'campplus.onnx'))
if not os.path.exists(CAMPP):
    sys.exit('set CAMPP_ONNX to campplus.onnx (see reference/models.md)')
js = json.load(open(script, encoding='utf-8'))
EN = js.get('lang') == 'English'
CPS_T = 14.0 if EN else 5.0
SIM_MIN = 0.6
lines, shown = {}, {}
for ln in js['lines']:
    lines[f"{ln['id']:02d}"] = ln.get('tts', ln['text'])       # compare with what the model was asked to read
    shown[f"{ln['id']:02d}"] = ln['text']                      # ...and with the subtitle: ASR writes 角色, not 绝色
    for k, t in ln.get('alt', {}).items():
        lines[f"{int(k[:-1]):02d}{k[-1]}"] = t
norm = lambda s: re.sub(r'[\W_0-9零一二两三四五六七八九十百²]+', '', s.lower())
sess = ort.InferenceSession(CAMPP, providers=['CPUExecutionProvider'])


def load(p):
    y, sr = sf.read(p)
    return (y.mean(1) if y.ndim > 1 else y), sr


def emb(y, sr):
    w = torch.tensor(y, dtype=torch.float32).reshape(1, -1)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    e = sess.run(None, {sess.get_inputs()[0].name: (f - f.mean(0, keepdim=True)).unsqueeze(0).numpy()})[0].flatten()
    return e / np.linalg.norm(e)


def cer(ref_text, hyp):
    r, h = norm(ref_text), norm(hyp)
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return dp[len(h)] / max(1, len(r))


def median_f0(y, sr):
    f, t = pw.dio(y.astype(np.float64), sr)
    f = pw.stonemask(y.astype(np.float64), f, t, sr)
    v = f[(f > 60) & (f < 500)]
    return float(np.median(v)) if len(v) else 0.0


_y, _sr = load(ref)
e_ref, F0_REF = emb(_y, _sr), median_f0(_y, _sr)
print(f'reference {os.path.basename(ref)}: F0 {F0_REF:.0f} Hz', flush=True)
asr = whisper.load_model('large-v3')
sel, out = {}, os.path.join(d, 'best')
os.makedirs(out, exist_ok=True)
for lid, text in sorted(lines.items()):
    rows = []
    for p in sorted(glob.glob(f'{d}/L{lid}_t*.wav')):
        y, sr = load(p)
        y16 = torchaudio.functional.resample(torch.tensor(y, dtype=torch.float32), sr, 16000).numpy()
        hyp = (asr.transcribe(y16, language='en')['text'].strip() if EN else
               asr.transcribe(y16, language='zh', initial_prompt='以下是普通话的句子。')['text'].strip())
        dur = len(y) / sr
        c = min(cer(text, hyp), cer(shown.get(lid, text), hyp))   # a same-sound stand-in heard right scores 0
        # cut off: the model sometimes stops mid-syllable and Whisper still "hears" the word
        fr = np.array([np.sqrt(np.mean(y[i:i + sr // 50] ** 2)) for i in range(0, len(y) - sr // 50, sr // 50)])
        db = 20 * np.log10(fr + 1e-9)
        if EN:
            cut = bool(db[-5:].max() > db.max() - 25)
        else:   # Chinese takes often end on a loud final syllable: cut only if no silence follows the speech
                # and the last frame is still within 10 dB of that syllable
            end = np.flatnonzero(db > db.max() - 30)[-1]
            cut = bool((len(db) - 1 - end) * 0.02 < 0.03 and db[max(0, end - 15):end + 1].max() - db[-1] < 10)
        rows.append(dict(take=os.path.basename(p), sim=round(float(emb(y, sr) @ e_ref), 3), cer=round(c, 3), cut=cut,
                         dur=round(dur, 2), cps=round(len(norm(text)) / dur, 2), f0=round(median_f0(y, sr)), asr=hyp))
    if not rows:
        print(f'L{lid}: no takes', flush=True)
        continue
    ok = ([r for r in rows if r['sim'] >= SIM_MIN and not r['cut']] or [r for r in rows if r['sim'] >= SIM_MIN]
          or [max(rows, key=lambda r: r['sim'])])
    best_cer = min(r['cer'] for r in ok)
    pool = [r for r in ok if r['cer'] <= best_cer + 0.01]
    best = max(pool, key=lambda r: (round(r['sim'], 2),
                                    -abs(r['cps'] - CPS_T) / CPS_T - abs(r['f0'] - F0_REF) / max(F0_REF, 1)))
    sel[lid] = dict(best=best, all=rows)
    shutil.copy(os.path.join(d, best['take']), os.path.join(out, f'L{lid}.wav'))
    flag = '  <-- check' if best['cer'] > 0.05 or best['sim'] < SIM_MIN or best['cut'] else ''
    print(f"L{lid}: {best['take']} sim {best['sim']:.2f} cer {best['cer']:.3f} {best['dur']:.1f}s f0 {best['f0']:.0f}{flag} "
          f"| passed {sum(r['sim'] >= SIM_MIN for r in rows)}/{len(rows)} | {best['asr']}", flush=True)
json.dump(sel, open(os.path.join(out, 'selection.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
parts, sr0 = [], None
for lid in sorted(k for k in sel if not k[-1].isalpha()):
    y, sr = sf.read(os.path.join(out, f'L{lid}.wav'))
    sr0 = sr0 or sr
    parts += [y, np.zeros(int(0.35 * sr))]
if parts:
    sf.write(os.path.join(out, 'preview_all.wav'), np.concatenate(parts), sr0)
    print('total', round(sum(len(p) for p in parts) / sr0, 1), 's')
