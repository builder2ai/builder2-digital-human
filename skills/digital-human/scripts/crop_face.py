"""Crop around the face so it takes a set share of the width, ready to be redrawn at full size.

Asked to "recompose", the edit model only pulls the camera back and the person shrinks. The
reliable way to a close podcast framing is to crop to it, then redraw the crop with qwen_edit.py.
The face centre goes at `cy` of the crop height (the eyes land about a third of the way down).

usage: crop_face.py <src> <out> [face_ratio=0.22] [cy=0.38] [out_w=1248] [out_h=832]
"""
import sys, io, contextlib, warnings, cv2, insightface

warnings.filterwarnings('ignore', category=FutureWarning)     # insightface's numpy lstsq notice

src, out = sys.argv[1], sys.argv[2]
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.22
cyf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.38
ow = int(sys.argv[5]) if len(sys.argv) > 5 else 1248
oh = int(sys.argv[6]) if len(sys.argv) > 6 else 832
im = cv2.imread(src)
H, W = im.shape[:2]
with contextlib.redirect_stdout(io.StringIO()):          # InsightFace prints every model it loads
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
f = max(app.get(im), key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
x0, y0, x1, y1 = f.bbox
fw, cx, cy = x1 - x0, (x0 + x1) / 2, (y0 + y1) / 2
cw = min(W, fw / ratio)
ch = min(H, cw * oh / ow)
cw = ch * ow / oh
bx = int(min(max(0, cx - cw / 2), W - cw))
by = int(min(max(0, cy - cyf * ch), H - ch))
crop = im[by:by + int(ch), bx:bx + int(cw)]
cv2.imwrite(out, cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LANCZOS4))
print(f"{src}: face {fw:.0f}px -> crop ({bx},{by}) {cw:.0f}x{ch:.0f} -> {ow}x{oh}, face ratio {fw / cw:.2f}")
