"""Level a slightly tilted head (2-5 degrees) by rotating only the head region.

Prompted edits keep the input's head pose, so a few degrees of tilt survive every "straighten her
head" pass. A small tilt is an in-plane rotation and can be undone exactly: rotate the image about
the face centre until the eye line is horizontal, then blend the rotated head back through a soft
ellipse (face, glasses, hairline). Shoulders, long hair below the chin and the background stay as
they were. If the rim of the ellipse shows a kink in the hair, run a light refine pass afterwards
(qwen_edit.py with denoise 0.3).

usage: level_head.py <src> <out> [mask_scale=1.0]
"""
import sys, math, io, contextlib, warnings, cv2, numpy as np, insightface

warnings.filterwarnings('ignore', category=FutureWarning)     # insightface's numpy lstsq notice

src, out = sys.argv[1], sys.argv[2]
k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
im = cv2.imread(src)
with contextlib.redirect_stdout(io.StringIO()):          # InsightFace prints every model it loads
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
f = max(app.get(im), key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
(lx, ly), (rx, ry) = f.kps[0], f.kps[1]
angle = math.degrees(math.atan2(ry - ly, rx - lx))      # eye-line tilt in image coordinates
x0, y0, x1, y1 = f.bbox
w, h = x1 - x0, y1 - y0
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

# cv2 positive angle = counter-clockwise on screen; an eye line rising to the right (angle < 0)
# needs a clockwise turn, i.e. rotate by `angle` itself.
H, W = im.shape[:2]
M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
rot = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

mask = np.zeros((H, W), np.float32)
cv2.ellipse(mask, (int(cx), int(cy - 0.12 * h)), (int(0.82 * k * w), int(0.80 * k * h)), 0, 0, 360, 1.0, -1)
mask = cv2.GaussianBlur(mask, (0, 0), 0.12 * w)[..., None]
res = rot.astype(np.float32) * mask + im.astype(np.float32) * (1 - mask)
cv2.imwrite(out, np.clip(res, 0, 255).astype(np.uint8))
print(f"{src}: eye line {angle:+.1f} deg -> rotated head by {angle:+.1f} -> {out}")
