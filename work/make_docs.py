"""Write captions SRT and an edit decision list (EDL) for the reel."""
import json
C = json.load(open("captions.json"))
E = json.load(open("edit_map.json"))

def ts(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"

with open("../output/captions_ms.srt", "w") as f:
    prev_end = "."
    for i, c in enumerate(C["chunks"], 1):
        txt = " ".join(w["w"] for w in c["words"])
        if prev_end in ".?!":
            txt = txt[0].upper() + txt[1:]
        prev_end = txt[-1]
        f.write(f"{i}\n{ts(c['t0'])} --> {ts(c['t1'])}\n{txt}\n\n")

def mmss(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"

lines = ["# Edit decision list", "",
         "All audio is the original interview; segments are selected and pauses tightened only.", "",
         "| # | Reel time | Source time | Speaker | Content (first words) |", "|---|---|---|---|---|"]
for i, s in enumerate(E["segments"], 1):
    words = [w["w"] for w in C["words"] if s["out_a"] - 0.05 <= w["t0"] < s["out_b"]]
    lines.append(f"| {i} | {mmss(s['out_a'])}–{mmss(s['out_b'])} | {mmss(s['src_a'])}–{mmss(s['src_b'])} | "
                 f"{s['speaker']} | {' '.join(words[:9]).rstrip('.,?')}… |")
open("../output/EDL.md", "w").write("\n".join(lines) + "\n")
print(open("../output/EDL.md").read())
