import json, sys, os
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[1], device="cpu", compute_type="int8", cpu_threads=4)
import soundfile as sf
audio, _sr = sf.read(os.environ.get("SRC", "src16k.wav"), dtype="float32")  # 16 kHz mono; avoids PyAV version issues
assert _sr == 16000
segs, info = m.transcribe(audio, language="ms", word_timestamps=True, beam_size=5, vad_filter=False,
    initial_prompt=os.environ.get("PROMPT", "Temu bual radio Kool FM tentang relationship, space, avoid conflict, confrontation, emotionally overwhelmed, stonewalling, silent treatment."))
out=[]
for s in segs:
    d={"start":s.start,"end":s.end,"text":s.text,"words":[{"s":w.start,"e":w.end,"w":w.word,"p":w.probability} for w in s.words]}
    out.append(d); print(f"[{s.start:7.2f}-{s.end:7.2f}] {s.text}", flush=True)
json.dump(out, open(os.environ.get("OUT", f"transcript_{sys.argv[1]}.json"),"w"), ensure_ascii=False, indent=1)
