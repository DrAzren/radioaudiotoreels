"""Time-warp between the v1 timeline (which the scene/SFX code is written in) and the
current edit. Anchors = the same spoken word in both edits (matched by segment + source
time); the mapping is smoothed so visual motion speeds change gradually.

  to_new(t_old) -> t_new      (used for audio cue placement)
  to_old(t_new) -> t_old      (used to sample the visual timeline)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_old = json.load(open(os.path.join(HERE, "captions_v1.json")))["words"]
_new = json.load(open(os.path.join(HERE, "captions.json")))["words"]
_dur_new = json.load(open(os.path.join(HERE, "edit_map.json")))["duration"]

_idx = {(w["seg"], round(w["s"], 2)): w["t0"] for w in _old}
pairs = [(0.0, 0.0), (0.7, 0.7)]
for w in _new:
    k = (w["seg"], round(w["s"], 2))
    if k in _idx:
        pairs.append((_idx[k], w["t0"]))
pairs.sort(key=lambda p: p[1])
mono = [pairs[0]]
for o, n in pairs[1:]:
    if o > mono[-1][0] + 0.01 and n > mono[-1][1] + 0.01:
        mono.append((o, n))
last_o, last_n = mono[-1]
mono.append((last_o + 30, last_n + 30))  # slope 1 beyond the last word
A_old = np.array([p[0] for p in mono])
A_new = np.array([p[1] for p in mono])

_T = np.arange(-5, _dur_new + 35, 0.01)
_g = np.interp(_T, A_new, A_old)
_sig = 0.45 / 0.01
_k = np.exp(-0.5 * (np.arange(-4 * _sig, 4 * _sig + 1) / _sig) ** 2)
_k /= _k.sum()
_gs = np.convolve(np.pad(_g, len(_k) // 2, mode="edge"), _k, mode="valid")[:len(_T)]
_w = np.clip((_T - 0.6) / 1.5, 0, 1)        # keep the hook frame-exact (no smoothing at t<0.6)
_gs = _g * (1 - _w) + _gs * _w
_gs = np.maximum.accumulate(_gs)


def to_old(t_new):
    return float(np.interp(t_new, _T, _gs))


def to_new(t_old):
    return float(np.interp(t_old, _gs, _T))


if __name__ == "__main__":
    for tn in np.arange(0, _dur_new, 5):
        print(f"new {tn:6.1f} -> old {to_old(tn):6.2f}")
    print("anchors", len(mono))
