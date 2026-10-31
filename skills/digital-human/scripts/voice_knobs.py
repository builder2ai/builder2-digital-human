"""Turn three knobs on a reference voice with WORLD: pitch, formants (resonance) and intonation range.

The clone model copies the reference's pitch and melody, so changing the reference changes the voice.
  f0       x pitch: > 1 higher
  formant  x formant frequencies: > 1 brighter and smaller-sounding, < 1 fuller and deeper
  range    x pitch movement around the median (in log F0): > 1 livelier intonation
Move each a little (pitch and formant 3-8 %, range 10-20 %) and make 3-4 variants to listen to.

usage: voice_knobs.py <in.wav> <out_dir> <name=f0:formant[:range]> [...]
  e.g. voice_knobs.py design_t2.wav knobs lighter=1.04:1.03:1.15 calmer=0.97:0.98:0.9
writes <out_dir>/<name>.wav and prints each file's median pitch
"""
import sys, os, numpy as np, soundfile as sf, pyworld as pw

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
x, fs = sf.read(src)
x = (x.mean(1) if x.ndim > 1 else x).astype(np.float64)


def world(y, f0_ratio, formant_ratio, f0_range=1.0):
    f0, t = pw.harvest(y, fs)
    sp, ap = pw.cheaptrick(y, f0, t, fs), pw.d4c(y, f0, t, fs)
    kk = np.arange(sp.shape[1])
    sp2 = np.array([np.interp(kk / formant_ratio, kk, r) for r in sp])
    ap2 = np.array([np.interp(kk / formant_ratio, kk, r) for r in ap])
    v = f0 > 0
    f0n = f0.copy()
    if f0_range != 1.0 and v.any():
        lm = np.median(np.log(f0[v]))
        f0n[v] = np.exp(lm + f0_range * (np.log(f0[v]) - lm))
    z = pw.synthesize(f0n * f0_ratio, sp2, ap2, fs)[:len(y)]
    hop = int(0.01 * fs)
    env = np.sqrt(np.convolve(y ** 2, np.ones(hop) / hop, mode='same'))
    gate = np.clip((20 * np.log10(env + 1e-9) + 50) / 10, 0, 1)      # keep the pauses silent
    gate = np.convolve(gate, np.hanning(4 * hop) / np.hanning(4 * hop).sum(), mode='same')
    z = z * gate[:len(z)]
    return (0.9 * z / np.abs(z).max()).astype(np.float32), float(np.median(f0n[v] * f0_ratio)) if v.any() else 0.0


f0, _ = pw.harvest(x, fs)
print(f'{os.path.basename(src)}: median pitch {np.median(f0[f0 > 0]):.0f} Hz')
for spec in sys.argv[3:]:
    name, val = spec.split('=')
    y, med = world(x, *map(float, val.split(':')))
    p = os.path.join(out, f'{name}.wav')
    sf.write(p, y, fs)
    print(f'{p}: median pitch {med:.0f} Hz', flush=True)
