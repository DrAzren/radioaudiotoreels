import json, sys
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[1], device="cpu", compute_type="int8", cpu_threads=4)
segs, info = m.transcribe("src16k.wav", language="ms", word_timestamps=True, beam_size=5, vad_filter=False,
    initial_prompt="Temu bual radio Kool FM tentang relationship, space, avoid conflict, confrontation, emotionally overwhelmed, stonewalling, silent treatment.")
out=[]
for s in segs:
    d={"start":s.start,"end":s.end,"text":s.text,"words":[{"s":w.start,"e":w.end,"w":w.word,"p":w.probability} for w in s.words]}
    out.append(d); print(f"[{s.start:7.2f}-{s.end:7.2f}] {s.text}", flush=True)
json.dump(out, open(f"transcript_{sys.argv[1]}.json","w"), ensure_ascii=False, indent=1)
