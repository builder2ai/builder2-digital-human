"""Widen a narrower picture to 16:9 by extending the sides only, so the face does not move at all.

prep : the picture goes in the middle of a 16:9 canvas of the same height; the side strips are
       filled with its mirrored, heavily blurred edges as placeholders for the edit model.
       Then run qwen_edit.py on the canvas with an English prompt that calls the strips placeholders
       and names what continues on each side, e.g. "This wide photo's left and right edges are only
       blurry placeholders. Extend the scene naturally into them: on the left, continue the
       bookshelf and the warm lamp glow; on the right, continue the bookshelf and the microphone
       boom arm running out of the frame. Keep the person and everything in the middle exactly the
       same." A vague "complete the sides" leaves the blur in place. Make 2-4 seeds and look at
       the edges at full size.
merge: the edited canvas is scaled back, and the ORIGINAL picture is laid over the middle
       (feathered over `feather` px just inside its edges), so only the new side strips come
       from the model.

usage: outpaint_169.py prep  <picture> <canvas_out>
       outpaint_169.py merge <picture> <edited_canvas> <out> [feather=60]
"""
import sys, cv2, numpy as np

mode, src = sys.argv[1], sys.argv[2]
im = cv2.imread(src)
H, W = im.shape[:2]
W2 = int(round(H * 16 / 9 / 2)) * 2
x0 = (W2 - W) // 2
x1 = x0 + W

if mode == 'prep':
    can = cv2.copyMakeBorder(im, 0, 0, x0, W2 - x1, cv2.BORDER_REFLECT)
    blur = cv2.GaussianBlur(can, (0, 0), 25)
    can[:, :x0], can[:, x1:] = blur[:, :x0], blur[:, x1:]
    cv2.imwrite(sys.argv[3], can)
    print(f'canvas {W2}x{H}, picture at x {x0}..{x1}')
else:
    ed, out = cv2.imread(sys.argv[3]), sys.argv[4]
    feather = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    ed = cv2.resize(ed, (W2, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    a = np.zeros(W2, np.float32)
    a[x0:x1] = 1
    a[x0:x0 + feather] = np.linspace(0, 1, feather)
    a[x1 - feather:x1] = np.linspace(1, 0, feather)
    a = a[None, :, None]
    full = np.zeros_like(ed)
    full[:, x0:x1] = im
    res = ed * (1 - a) + full * a
    cv2.imwrite(out, np.clip(res, 0, 255).astype(np.uint8))
    print(f'{out}: {W2}x{H}')
