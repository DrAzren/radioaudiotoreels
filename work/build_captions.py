"""Map Whisper word timings through the edit and build caption chunks.

Corrections fix obvious ASR mistakes only (checked against alternate decodes);
words whose content could not be confirmed are left uncaptioned rather than guessed.
"""
import json
import re

T = json.load(open("transcript_large-v3.json"))
E = json.load(open("edit_map.json"))

words = []
for s in T:
    for w in s["words"]:
        words.append(dict(s=w["s"], e=w["e"], w=w["w"].strip()))

# Drop Whisper's hallucinated repeat loop and the unconfirmed phrase (~145.8-147.7)
words = [w for w in words if not (145.80 <= w["s"] < 147.68)]

# time-anchored corrections: (approx_src_time, old_lower, new)  new=None -> drop
FIX = [
    (59.84, "konfrontasi.", "confrontation."),
    (61.28, "insting", "instinct"),
    (62.24, "mereka", "better"),
    (57.38, "sebut.", "tersebut."),
    (69.9, "disebutkan", "disebabkan"),
    (103.04, "makanan.", "masalah."),
    (148.26, "terlalu", "malam"),
    (149.22, "bintang", "bincang"),
    (160.12, "mata", "macam"),
    (165.44, "dia", None),
    (166.22, "tone", None),
    (166.88, "wall.", "stonewall."),
    (122.0, "okey.", None),
    (104.58, "doktor", "Doktor,"),
    (105.00, "faham", "faham"),
    (84.42, "contohnya", "Contohnya,"),
    (87.14, "cakap.", "cakap,"),
    (87.60, "ayah", "ayah"),
    (170.54, "dia", "Dia"),
    (50.28, "sebenarnya", "Sebenarnya,"),
    (68.0, "sikit.", None),
    (107.18, "dia", "Dia"),
    (125.2, "dia", "Dia"),
]
# unconfirmed words ("... minit je" count) are left uncaptioned rather than guessed
words = [w for w in words if not (128.0 <= w["s"] < 130.1)]
for t, old, new in FIX:
    cand = [w for w in words if abs(w["s"] - t) < 1.6 and w["w"].lower() == old]
    if not cand:
        print("WARN no match", t, old)
        continue
    w = min(cand, key=lambda w: abs(w["s"] - t))
    if new is None:
        words.remove(w)
    else:
        w["w"] = new

# merge hyphen continuation tokens ("Kadang" + "-kadang")
merged = []
for w in words:
    if w["w"].startswith("-") and merged and w["s"] - merged[-1]["e"] < 0.3:
        merged[-1]["w"] += w["w"]
        merged[-1]["e"] = w["e"]
    else:
        merged.append(w)
words = merged

# map to output time; a word whose (imprecise) timing falls in a removed pause
# is attached to the nearest kept piece of the same segment (within 0.45 s)
out = []
segs = {}
for p in E["pieces"]:
    segs.setdefault(p["seg"], []).append(p)
seg_ranges = [(s["label"], s["src_a"], s["src_b"]) for s in E["segments"]]
cut_ranges = [tuple(c) for c in E.get("inside_cuts", [])]
for label, sa, sb in seg_ranges:
    for w in words:
        mid = (w["s"] + min(w["e"], w["s"] + 0.6)) / 2
        if not (sa - 0.08 <= mid < sb + 0.05):
            continue
        if any(ca <= mid < cb for ca, cb in cut_ranges):
            continue
        best = None
        for p in segs[label]:
            dist = 0 if p["src_a"] <= mid < p["src_b"] else min(abs(mid - p["src_a"]), abs(mid - p["src_b"]))
            if best is None or dist < best[0]:
                best = (dist, p)
        if best[0] > 0.45:
            continue
        p = best[1]
        s0 = min(max(w["s"], p["src_a"]), p["src_b"] - 0.05)
        d = dict(w)
        d["t0"] = round(p["out_a"] + s0 - p["src_a"], 3)
        d["t1"] = round(p["out_a"] + min(p["src_b"], max(w["e"], s0 + 0.12)) - p["src_a"], 3)
        d["seg"] = label
        d["speaker"] = p["speaker"]
        out.append(d)
# de-dup (cold open and doc_key share source) is fine - different output times
out.sort(key=lambda d: d["t0"])
for a, b in zip(out, out[1:]):
    if a["t1"] > b["t0"]:
        a["t1"] = b["t0"]

# chunking
chunks, cur = [], []
def flush():
    if cur:
        chunks.append(dict(t0=cur[0]["t0"], t1=cur[-1]["t1"], speaker=cur[0]["speaker"], seg=cur[0]["seg"],
                           words=[dict(w=c["w"], t0=c["t0"], t1=c["t1"]) for c in cur]))
    cur.clear()

for i, w in enumerate(out):
    if cur:
        prev = cur[-1]
        text_len = sum(len(c["w"]) + 1 for c in cur) + len(w["w"])
        if (w["t0"] - prev["t1"] > 0.35 or w["seg"] != prev["seg"] or text_len > 26 or len(cur) >= 5
                or re.search(r"[.?!,]$", prev["w"])):
            flush()
    cur.append(w)
flush()
# merge dangling 1-word chunks into the previous chunk
i = 1
while i < len(chunks):
    c, p = chunks[i], chunks[i - 1]
    if len(c["words"]) == 1 and len(c["words"][0]["w"]) <= 6 and c["seg"] == p["seg"] and c["t0"] - p["t1"] < 0.4 \
            and sum(len(w["w"]) + 1 for w in p["words"]) <= 26:
        p["words"] += c["words"]; p["t1"] = c["t1"]; chunks.pop(i)
    else:
        i += 1
# extend chunk end slightly for readability, never overlapping the next
for a, b in zip(chunks, chunks[1:]):
    a["t1"] = min(b["t0"], a["t1"] + 0.35)
chunks[-1]["t1"] += 0.6

json.dump(dict(words=out, chunks=chunks), open("captions.json", "w"), ensure_ascii=False, indent=1)
for c in chunks:
    print(f"{c['t0']:6.2f}-{c['t1']:6.2f} [{c['speaker'][:3]}] " + " ".join(w["w"] for w in c["words"]))
