"""Render stills at given times in one process and tile them into a labelled grid."""
import sys
import cv2, numpy as np
import reel
ts = [float(x) for x in sys.argv[2:]]
tiles = []
for t in ts:
    im = cv2.cvtColor(reel.to8(reel.frame_at(int(round(t * reel.FPS)))), cv2.COLOR_RGB2BGR)
    im = cv2.resize(im, (270, 480), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f"{t:.1f}s", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(im)
cols = 6
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
