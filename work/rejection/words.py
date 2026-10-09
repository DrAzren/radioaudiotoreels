"""Print words with timings + the quietest 5 ms level in each inter-word gap (for cut planning)."""
import json, sys
import numpy as np, soundfile as sf
x, sr = sf.read("../src48m.wav"); HOP = sr // 200
db = 20*np.log10(np.sqrt(np.convolve(x**2, np.ones(HOP)/HOP, "same")[::HOP] + 1e-12))
T = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "transcript_large-v3.json"))
W = [w for s in T for w in s["words"]]
lo, hi = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0, 1e9)
for a, b in zip(W, W[1:] + [dict(s=W[-1]["e"] + .3, e=0, w="", p=1)]):
    if not (lo <= a["s"] < hi): continue
    g0, g1 = int(a["e"]*200) - 6, int(b["s"]*200) + 6
    seg = db[g0:max(g1, g0+1)]; k = int(np.argmin(seg))
    sil = (seg < -45).sum() * 5
    print(f'{a["s"]:7.2f}-{a["e"]:7.2f} p{a["p"]:.2f} {a["w"]:18s} | gap {b["s"]-a["e"]:5.2f}s min {seg[k]:6.1f}dB @{(g0+k)/200:7.3f} sil{sil:4d}ms')
