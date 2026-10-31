"""Measure the face in each image with InsightFace: how many faces, head tilt and turn, face size,
and (with --ref) how close it is to the locked face.

  faces         must be 1; more means the edit pasted in a second person
  eye_line_deg  tilt of the line between the eyes (positive = the eye on the image's right sits lower)
  yaw, pitch    head turn and nod from InsightFace's pose model
  roll          head tilt from the same pose model; measured differently, so it can be a degree or two off eye_line_deg
  face_w        face box width / image width (a podcast close-up is about 0.18-0.25)
  sim           cosine similarity of the face embedding to --ref (same face after an edit: >= 0.80)
A line ends with "<-- check" when faces != 1, or, without --ref, |eye_line_deg| > 2 (picking a new
face), or, with --ref, sim < 0.80 (edits and talking-video frames: the head sways a little there).

usage: face_check.py [--ref locked_face.png] <image> [<image> ...]
"""
import sys, os, math, json, io, contextlib, warnings
import numpy as np, cv2, insightface

warnings.filterwarnings('ignore', category=FutureWarning)     # insightface's numpy lstsq notice

args = sys.argv[1:]
ref = None
if args and args[0] == '--ref':
    ref, args = args[1], args[2:]
with contextlib.redirect_stdout(io.StringIO()):          # InsightFace prints every model it loads
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))


def faces(im):
    f = [x for x in app.get(im) if x.det_score > 0.5]
    if not f:      # small or tightly cropped faces sometimes need padding to be found
        p = im.shape[0] // 4
        f = [x for x in app.get(cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=0))
             if x.det_score > 0.5]
    return f


def biggest(f):
    return max(f, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))


e_ref = None
if ref:
    fr = faces(cv2.imread(ref))
    if not fr:
        sys.exit(f'no face found in {ref}')
    e_ref = biggest(fr).normed_embedding
for p in args:
    im = cv2.imread(p)
    fs = faces(im) if im is not None else []
    if not fs:
        print(json.dumps({"file": os.path.basename(p), "faces": 0}) + '  <-- check')
        continue
    f = biggest(fs)
    (lx, ly), (rx, ry) = f.kps[0], f.kps[1]
    eye = math.degrees(math.atan2(ry - ly, rx - lx))
    pitch, yaw, roll = (round(float(v), 1) for v in f.pose)
    row = {"file": os.path.basename(p), "faces": len(fs), "eye_line_deg": round(eye, 1), "roll": roll, "yaw": yaw,
           "pitch": pitch, "face_w": round(float(f.bbox[2] - f.bbox[0]) / im.shape[1], 3)}
    if e_ref is not None:
        row["sim"] = round(float(np.dot(f.normed_embedding, e_ref)), 3)
    bad = len(fs) != 1 or (row["sim"] < 0.80 if e_ref is not None else abs(eye) > 2)
    print(json.dumps(row) + ('  <-- check' if bad else ''), flush=True)
