"""Word-level transcript of the EDITED voice track, segment by segment (times = reel output time).
Captions are built from this, so removed audio can never leak into them."""
import json
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

E = json.load(open("edit_map.json"))
y, sr = sf.read("voice_edit_raw.wav")
y16 = resample_poly(y, 1, sr // 16000).astype(np.float32)
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
out = []
for s in E["segments"]:
    a, b = s["out_a"] - 0.05, s["out_b"] + 0.05
    segs, _ = m.transcribe(y16[int(a * 16000):int(b * 16000)], language="ms", word_timestamps=True, beam_size=5,
                           condition_on_previous_text=True, vad_filter=False,
                           initial_prompt="Temu bual radio tentang rejection, cemburu, nilai diri, relationship, emosi.")
    for sg in segs:
        for w in sg.words:
            out.append(dict(s=round(w.start + a, 3), e=round(w.end + a, 3), w=w.word.strip(), p=round(w.probability, 3),
                            seg=s["label"]))
json.dump(out, open("edit_transcript.json", "w"), ensure_ascii=False, indent=0)
for w in out:
    print(f"{w['s']:7.2f}-{w['e']:7.2f} p{w['p']:.2f} [{w['seg']}] {w['w']}")
