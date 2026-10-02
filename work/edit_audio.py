"""Build the edited voice track from the radio interview.

Each segment is a (src_start, src_end) range of the ORIGINAL audio. No words are
generated or re-ordered within a sentence; only whole phrases are selected and
pauses are tightened. Boundaries are snapped to the quietest 10 ms frame nearby
so cuts land in natural gaps. Output:
  voice_edit.wav       - edited mono voice (48 kHz)
  edit_map.json        - list of pieces mapping src time -> output time
"""
import json
import numpy as np
import soundfile as sf

SR = 48000
x, sr = sf.read("src48m.wav")
assert sr == SR
hop = SR // 100
rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same")[::hop] + 1e-12)
db = 20 * np.log10(rms)


def snap(t, win=0.12):
    i0, i1 = max(0, int((t - win) * 100)), int((t + win) * 100)
    i = i0 + int(np.argmin(db[i0:i1]))
    return i / 100


# (label, src_start, src_end, gap_before_seconds, speaker, cuts_inside)
# cuts_inside: list of (a, b) source ranges removed inside the segment
SEGMENTS = [
    ("cold_open", 137.16, 142.42, 0.70, "DOKTOR", []),
    ("host_q", 0.00, 4.95, 1.25, "HOS", []),
    ("host_mamak", 10.25, 18.95, 0.30, "HOS", []),
    ("doc_avoid", 50.22, 63.95, 0.50, "DOKTOR", [(50.66, 51.40), (55.45, 57.62)]),  # drop "adik dia", redundant "melarikan diri..."
    ("doc_shutdown", 68.40, 78.86, 0.35, "DOKTOR", []),
    ("doc_child", 78.92, 93.98, 0.40, "DOKTOR", [(84.30, 85.00), (86.18, 87.55)]),  # drop repeated "Contohnya", "ataupun cuba nak cakap"
    ("doc_two", 94.02, 103.50, 0.45, "DOKTOR", []),
    ("host_time", 105.45, 110.42, 0.45, "HOS", []),
    ("doc_noNumber", 125.62, 131.50, 0.40, "DOKTOR", [(127.90, 130.06)]),  # drop unclear "... minit je"
    ("doc_key", 137.16, 142.42, 0.95, "DOKTOR", []),
    ("doc_healthy", 142.95, 151.40, 0.55, "DOKTOR", []),  # ends on "healthy space"
    ("doc_avoidance", 153.20, 167.36, 0.40, "DOKTOR", [(161.22, 163.18)]),  # drop "Itu bukan cooling off lah kita panggil"
    ("doc_final", 167.25, 175.40, 0.70, "DOKTOR", []),
]
TAIL = 4.4           # music-only end card
MAX_PAUSE = 0.24     # internal pauses longer than this are tightened
MIN_SIL_DB = -42
FADE = 0.012

pieces = []          # dicts: src_a, src_b, out_a
out = [np.zeros(0)]
t_out = 0.0


def add_silence(d):
    global t_out
    n = int(round(d * SR))
    out.append(np.zeros(n))
    t_out += n / SR


def add_audio(a, b, seg_label, speaker):
    global t_out
    ia, ib = int(round(a * SR)), int(round(b * SR))
    chunk = x[ia:ib].copy()
    nf = int(FADE * SR)
    if len(chunk) > 2 * nf:
        chunk[:nf] *= np.linspace(0, 1, nf)
        chunk[-nf:] *= np.linspace(1, 0, nf)
    pieces.append(dict(src_a=a, src_b=b, out_a=t_out, seg=seg_label, speaker=speaker))
    out.append(chunk)
    t_out += len(chunk) / SR


def split_pauses(a, b):
    """Return sub-ranges of [a,b] with long silences removed (kept MAX_PAUSE)."""
    i0, i1 = int(a * 100), int(b * 100)
    quiet = db[i0:i1] < MIN_SIL_DB
    ranges, start, i = [], a, 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            dur = (j - i) / 100
            if dur > MAX_PAUSE and i > 0 and j < len(quiet):
                keep = MAX_PAUSE / 2
                ranges.append((start, a + i / 100 + keep))
                start = a + j / 100 - keep
            i = j
        else:
            i += 1
    ranges.append((start, b))
    return ranges


segments_meta = []
for label, a, b, gap, spk, cuts in SEGMENTS:
    a, b = snap(a), snap(b)
    add_silence(gap)
    seg_out_a = t_out
    # apply inside cuts
    spans, cur = [], a
    for ca, cb in cuts:
        ca, cb = snap(ca, 0.08), snap(cb, 0.08)
        spans.append((cur, ca))
        cur = cb
    spans.append((cur, b))
    for sa, sb in spans:
        for ra, rb in split_pauses(sa, sb):
            if rb - ra > 0.02:
                add_audio(ra, rb, label, spk)
    segments_meta.append(dict(label=label, speaker=spk, out_a=seg_out_a, out_b=t_out, src_a=a, src_b=b))

add_silence(TAIL)
y = np.concatenate(out)
sf.write("voice_edit_raw.wav", y, SR, subtype="PCM_24")
json.dump(dict(pieces=pieces, segments=segments_meta, duration=t_out), open("edit_map.json", "w"), indent=1)
print(f"duration {t_out:.2f}s")
for s in segments_meta:
    print(f"{s['label']:14s} {s['out_a']:7.2f}-{s['out_b']:7.2f}  ({s['out_b']-s['out_a']:.2f}s) src {s['src_a']:.2f}-{s['src_b']:.2f}")
