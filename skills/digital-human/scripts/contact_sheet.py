"""Numbered grid of images for a person to pick from. A label after the path ("img.png=sim 0.86")
is printed at the bottom of the tile, so the check results sit on the picture.

usage: contact_sheet.py <out.jpg> <cols> <tile_w> <img1>[=label] [<img2>[=label] ...]
Tiles take the first image's aspect ratio; other shapes are letterboxed.
"""
import sys, cv2, numpy as np

out, cols, tw, items = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
tiles, th = [], None
for i, it in enumerate(items, 1):
    path, _, label = it.partition('=')
    im = cv2.imread(path)
    th = th or int(round(tw * im.shape[0] / im.shape[1]))
    s = min(tw / im.shape[1], th / im.shape[0])
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    t = np.zeros((th, tw, 3), np.uint8)
    y, x = (th - im.shape[0]) // 2, (tw - im.shape[1]) // 2
    t[y:y + im.shape[0], x:x + im.shape[1]] = im
    cv2.rectangle(t, (0, 0), (64, 56), (0, 0, 0), -1)
    cv2.putText(t, str(i), (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    if label:
        cv2.rectangle(t, (0, th - 40), (tw, th), (0, 0, 0), -1)
        cv2.putText(t, label, (10, th - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(t)
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(out)
