"""5 ms level contour of the source between two times: one char per 5 ms (#>-20, +>-30, .>-45, space<=-45 dB)."""
import sys
import numpy as np, soundfile as sf
x, sr = sf.read("../src48m.wav"); HOP = sr // 200
db = 20*np.log10(np.sqrt(np.convolve(x**2, np.ones(HOP)/HOP, "same")[::HOP] + 1e-12))
for arg in sys.argv[1:]:
    a, b = map(float, arg.split(":"))
    seg = db[int(a*200):int(b*200)]
    s = "".join("#" if v > -20 else "+" if v > -30 else "." if v > -45 else " " for v in seg)
    print(f"{a:7.2f} |" + "|".join(s[i:i+20] for i in range(0, len(s), 20)) + f"| {b:.2f}")
