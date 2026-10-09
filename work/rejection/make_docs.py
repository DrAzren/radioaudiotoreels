"""Write captions SRT and the edit decision list (EDL) for the REJECTION reel."""
import json
import os

OUT = "../../output/rejection_cemburu"
os.makedirs(OUT, exist_ok=True)
C = json.load(open("captions.json"))
E = json.load(open("edit_map.json"))


def ts(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


with open(f"{OUT}/captions_ms.srt", "w") as f:
    prev_end = "."
    for i, c in enumerate(C["chunks"], 1):
        txt = " ".join(w["w"] for w in c["words"])
        if prev_end in ".?!":
            txt = txt[0].upper() + txt[1:]
        prev_end = txt[-1]
        f.write(f"{i}\n{ts(c['t0'])} --> {ts(c['t1'])}\n{txt}\n\n")


def mmss(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


NAMES = {"DOKTOR": "Dr Fakrul Azren bin Azhar", "HOS": "Hos radio"}
lines = ["# Edit decision list — Rejection & cemburu", "",
         "Source: `Rejection.mp4` (Google Drive) → audio only, `work/extract_mp3.sh` → `work/source.mp3` "
         "(189.36 s, AAC 44.1 kHz stereo in the MP4). No video frames were used.", "",
         "All audio is the original interview. Segments are selected whole; no words are generated, re-ordered "
         "or cloned. Every cut point is the quietest 5 ms frame in a gap between words "
         "(level in dB at the cut shown below). Each segment was re-transcribed on its own and in the final mix.", "",
         "| # | Reel time | Source time | Speaker | Cut level in/out (dB) | Content |", "|---|---|---|---|---|---|"]
for i, s in enumerate(E["segments"], 1):
    words = [w["w"] for w in C["words"] if s["out_a"] - 0.12 <= w["t0"] < s["out_b"]]
    spk = " → ".join(NAMES[g[1]] for g in s["gains_db"])
    lines.append(f"| {i} | {mmss(s['out_a'])}–{mmss(s['out_b'])} | {mmss(s['src_a'])}–{mmss(s['src_b'])} | {spk} | "
                 f"{s['cut_db'][0]} / {s['cut_db'][1]} | {' '.join(words[:10]).rstrip('.,?')}… |")
lines += ["", "Segment 1 is a flash-forward (cold open) of the doctor's line that plays again in context in segment 5.",
          "In segment 3 the host's question runs straight into the doctor's answer in the source (no cut at the "
          "hand-over; only the level is matched with a 60 ms gain ramp in the gap).", "",
          "## Removed inside segment 5 (whole phrases, silence on both sides)", "",
          "| Source time | Removed | Cut levels (dB) |", "|---|---|---|"]
removed = {0: "“Sebab sebenarnya lagi yang daripada kita.” (unclear, garbled)",
           1: "filler “apa panggil” (“Kita ni mungkin, apa panggil, meletakkan…”)",
           2: "“Kita tak boleh expect yang kita boleh, apa mereka boleh jadi bahu untuk kita bersandar apabila "
              "emosi kita tengah tak stable.” (false start; shortens the reel)"}
import numpy as np, soundfile as sf
x, sr = sf.read("../src48m.wav")
H = sr // 200
db = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(H) / H, "same")[::H] + 1e-12))
for k, (a, b) in enumerate(E["inside_cuts"]):
    lines.append(f"| {mmss(a)}–{mmss(b)} | {removed[k]} | {db[int(a * 200)]:.1f} / {db[int(b * 200)]:.1f} |")
lines += ["", "## Not used from the source", "",
          "- 0:01–1:00 host monologue after “How to handle rejection?”, incl. a women-vs-men generalisation "
          "and the guest introduction.",
          "- 1:42.5–1:51.2 doctor: “Sebab itu kadang-kadang reaksi seseorang… sangat-sangat besar.” "
          "(kept the reel at ~108 s).",
          "- 1:51.6–2:03.6 host aside and “Dan doktor bagaimana ibu ayah ini?…” (no clean gap before it; only "
          "its clean tail “Nak ajar anak-anak untuk menerima rejection.” is used).", "",
          "## Uncaptioned on purpose", "",
          "- “untuk ___ nilai diri mereka” — decoded as mengalih / mengalir / mengenali; not confirmed.",
          "- “So apa yang [ibu bapa | kita] buat adalah” — subject not confirmed across decodes; the caption "
          "resumes at “Sebenarnya kita boleh ajar satu benda lah”."]
open(f"{OUT}/EDL.md", "w").write("\n".join(lines) + "\n")
print(open(f"{OUT}/EDL.md").read())
