"""Build caption chunks from the transcript of the EDITED voice track (edit_transcript.json, reel time).

Corrections fix obvious ASR mistakes only; words that could not be confirmed with repeated
decodes are left uncaptioned rather than guessed.
"""
import json
import re

T = json.load(open("edit_transcript.json"))
E = json.load(open("edit_map.json"))

words = [dict(w) for w in T]

# (approx reel time, old lower-case token, new token or None to drop)
FIX = [
    (7.39, "reject?", "rejection?"),         # "-tion" is present in the audio (/ʃən/ at src 1.19-1.38 s)
    (66.22, "mengalih", None),               # mengalih / mengalir / mengenali: unconfirmed -> uncaptioned
    (82.81, "so", None),                     # "So apa yang [ibu bapa | kita] buat adalah": the subject is
    (83.56, "apa", None),                    #   unconfirmed across decodes, so this lead-in is left
    (83.86, "yang", None),                   #   uncaptioned; the caption resumes at "sebenarnya kita ..."
    (84.14, "ibu", None),
    (84.88, "bapa", None),
    (85.30, "buat", None),
    (85.38, "adalah", None),
    (85.61, "sebenarnya", "Sebenarnya"),
    (101.61, "fokus", "Fokus"),
    (16.92, "okey,", "Okey,"),
    (0.35, "tapi", "Tapi"),
]
for t, old, new in FIX:
    cand = [w for w in words if abs(w["s"] - t) < 0.6 and w["w"].lower() == old]
    if not cand:
        raise SystemExit(f"no match for fix {t} {old}")
    w = min(cand, key=lambda w: abs(w["s"] - t))
    if new is None:
        words.remove(w)
    else:
        w["w"] = new

# merge hyphen continuation tokens ("dua" + "-dua")
merged = []
for w in words:
    if w["w"].startswith("-") and merged and w["s"] - merged[-1]["e"] < 0.3:
        merged[-1]["w"] += w["w"]
        merged[-1]["e"] = w["e"]
    else:
        merged.append(w)
words = merged


def speaker_at(t):
    for p in E["pieces"]:
        dur = p["src_b"] - p["src_a"]
        if p["out_a"] - 0.15 <= t < p["out_a"] + dur + 0.05:
            return p["speaker"]
    return "DOKTOR"


out = []
for w in words:
    mid = (w["s"] + min(w["e"], w["s"] + 0.5)) / 2
    out.append(dict(w=w["w"], s=w["s"], t0=round(w["s"], 3), t1=round(max(w["e"], w["s"] + 0.12), 3),
                    seg=w["seg"], speaker=speaker_at(mid)))
out.sort(key=lambda d: d["t0"])
for a, b in zip(out, out[1:]):
    if a["t1"] > b["t0"]:
        a["t1"] = b["t0"]

# chunking: <= 2 lines on screen (~26 chars), break at punctuation, pauses and speaker/segment changes
chunks, cur = [], []


def flush():
    if cur:
        chunks.append(dict(t0=cur[0]["t0"], t1=cur[-1]["t1"], speaker=cur[0]["speaker"], seg=cur[0]["seg"],
                           words=[dict(w=c["w"], t0=c["t0"], t1=c["t1"]) for c in cur]))
    cur.clear()


for w in out:
    if cur:
        prev = cur[-1]
        text_len = sum(len(c["w"]) + 1 for c in cur) + len(w["w"])
        if (w["t0"] - prev["t1"] > 0.35 or w["seg"] != prev["seg"] or w["speaker"] != prev["speaker"]
                or text_len > 26 or len(cur) >= 5 or re.search(r"[.?!,]$", prev["w"])):
            flush()
    cur.append(w)
flush()
# merge dangling 1-word chunks into the previous chunk
i = 1
while i < len(chunks):
    c, p = chunks[i], chunks[i - 1]
    if len(c["words"]) == 1 and len(c["words"][0]["w"]) <= 6 and c["seg"] == p["seg"] and \
            c["speaker"] == p["speaker"] and c["t0"] - p["t1"] < 0.4 and sum(len(w["w"]) + 1 for w in p["words"]) <= 26:
        p["words"] += c["words"]
        p["t1"] = c["t1"]
        chunks.pop(i)
    else:
        i += 1
# extend chunk end slightly for readability, never overlapping the next
for a, b in zip(chunks, chunks[1:]):
    a["t1"] = min(b["t0"], a["t1"] + 0.35)
chunks[-1]["t1"] += 0.6

json.dump(dict(words=out, chunks=chunks), open("captions.json", "w"), ensure_ascii=False, indent=1)
for c in chunks:
    print(f"{c['t0']:6.2f}-{c['t1']:6.2f} [{c['speaker'][:3]}] " + " ".join(w["w"] for w in c["words"]))
