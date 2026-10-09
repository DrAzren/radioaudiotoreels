"""Re-transcribe each edited segment on its own (and a +-3 s window around every inside cut)
from an edited/mixed file, to catch clipped first/last words or masked phrases."""
import json, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

path = sys.argv[1] if len(sys.argv) > 1 else "voice_edit_raw.wav"
E = json.load(open("edit_map.json"))
y, sr = sf.read(path)
if y.ndim > 1:
    y = y.mean(1)
y16 = resample_poly(y, 1, sr // 16000).astype(np.float32)
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
PROMPT = "Temu bual radio tentang rejection, cemburu, nilai diri, relationship, emosi."


def tr(a, b, tag):
    seg = y16[int(max(0, a) * 16000):int(b * 16000)]
    segs, _ = m.transcribe(seg, language="ms", word_timestamps=True, beam_size=5, initial_prompt=PROMPT,
                           condition_on_previous_text=False)
    words = [w for s in segs for w in s.words]
    txt = "".join(w.word for w in words).strip()
    first = f"{words[0].word.strip()}@{words[0].start:.2f}(p{words[0].probability:.2f})" if words else "-"
    last = f"{words[-1].word.strip()}@{words[-1].end:.2f}(p{words[-1].probability:.2f})" if words else "-"
    print(f"[{tag}] {a:6.2f}-{b:6.2f}  first {first}  last {last}\n    {txt}", flush=True)


for s in E["segments"]:
    tr(s["out_a"] - 0.05, s["out_b"] + 0.05, s["label"])
# windows around inside cuts (output time of the cut)
for p, q in zip(E["pieces"], E["pieces"][1:]):
    if p["seg"] == q["seg"] and abs(p["src_b"] - q["src_a"]) > 0.05:
        t = q["out_a"]
        tr(t - 3, t + 3, f"cut@{t:.2f}")
