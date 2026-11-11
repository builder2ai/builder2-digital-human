"""Import stub for diffq, for `separate.py --model parts` only.

audio-separator's bundled Demucs code imports diffq when it loads (seen with audio-separator 0.47.1), even for models that
are not quantized. htdemucs_ft is not quantized, so these names are imported but never called. separate.py puts this folder
on the path only when the real diffq is missing. Any real use raises, so a quantized model fails loudly instead of quietly.
The real package (pip install diffq) has a compiled extension; install it if you can, then this stub is never used.
"""


class _NotInstalled:
    def __init__(self, *a, **k):
        raise RuntimeError('diffq stub: a quantized Demucs model needs the real diffq package (pip install diffq)')


DiffQuantizer = UniformQuantizer = _NotInstalled


def restore_quantized_state(*a, **k):
    raise RuntimeError('diffq stub: a quantized Demucs model needs the real diffq package (pip install diffq)')
