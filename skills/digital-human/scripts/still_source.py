"""The 底稿: a calm "breathing" video made from one portrait, as the source for InfiniteTalk.

The portrait itself with a slow, small sway (sums of incommensurate sines: ~0.4 % shift, ~0.35 deg
roll, ~0.4 % zoom), a closed mouth and no speech. No model is involved. InfiniteTalk then keeps this
face and motion and only adds the talking.
A silent audio track is muxed in: VideoHelperSuite's LoadVideo on a file with no audio track can
crash ComfyUI's worker.

usage: still_source.py <portrait> <out.mp4> <seconds> [amp=1.0] [fps=25]
  seconds: the line's length + 0.5
Needs ffmpeg on PATH, or the imageio-ffmpeg package.
"""
import sys, shutil, subprocess, cv2, numpy as np

try:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FF = shutil.which('ffmpeg') or sys.exit('needs ffmpeg on PATH or pip install imageio-ffmpeg')
src, out, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
amp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
fps = int(sys.argv[5]) if len(sys.argv) > 5 else 25
im = cv2.imread(src)
H, W = im.shape[:2]
W, H = W // 2 * 2, H // 2 * 2
im = im[:H, :W]
n = int(round(secs * fps))
t = np.arange(n) / fps
s = lambda *pf: sum(np.sin(2 * np.pi * t / p + f) for p, f in pf) / len(pf)
dx = amp * 0.004 * W * s((6.1, 0.0), (3.7, 1.3))
dy = amp * 0.003 * H * s((4.3, 0.7), (2.9, 2.1))
rot = amp * 0.35 * s((7.9, 0.4), (5.3, 2.6))
zoom = 1.02 + amp * 0.004 * s((9.7, 1.1))            # 2 % base zoom hides the borders
enc = subprocess.Popen([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps),
                        '-i', '-', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-shortest', '-c:v', 'libx264',
                        '-crf', '14', '-pix_fmt', 'yuv420p', '-c:a', 'aac', out], stdin=subprocess.PIPE)
for i in range(n):
    M = cv2.getRotationMatrix2D((W / 2, H * 0.4), rot[i], zoom[i])
    M[:, 2] += (dx[i], dy[i])
    enc.stdin.write(cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).tobytes())
enc.stdin.close()
enc.wait()
print(f'{out}: {n} frames {W}x{H}, amp {amp}')
