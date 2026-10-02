"""Build the edited voice track from the radio interview (v2: no clipped words).

Each segment is a range of the ORIGINAL audio. No words are generated or re-ordered.
v2 rules (after review: v1 sounded choppy):
  * every cut point is placed in the quietest 5 ms frame *between two words* inside an
    explicit search window, preferring real silence; fades scale with how loud the cut is
  * no mid-sentence trims unless both sides fall in true silence
  * only long pauses are tightened, and gently
  * each segment is level-matched (no aggressive frame-by-frame auto-gain)
Outputs voice_edit_raw.wav and edit_map.json (src time -> output time).
"""
import json

import numpy as np
import soundfile as sf

SR = 48000
x, sr = sf.read("src48m.wav")
assert sr == SR
HOP = SR // 200  # 5 ms
db = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(HOP) / HOP, "same")[::HOP] + 1e-12))


def cut_point(lo, hi, prefer="any"):
    """Quietest 5 ms frame in [lo, hi]. If a real silence (< -45 dB, >= 30 ms) exists, cut inside
    it: at its end for a segment start (prefer='late'), at its start for an end (prefer='early')."""
    i0, i1 = int(lo * 200), int(hi * 200)
    seg = db[i0:i1]
    q = seg < -45
    runs, i = [], 0
    while i < len(q):
        if q[i]:
            j = i
            while j < len(q) and q[j]:
                j += 1
            if j - i >= 6:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if runs:
        if prefer == "late":
            a, b = runs[-1]
            k = max(a, b - 4)          # 20 ms before speech resumes
        elif prefer == "early":
            a, b = runs[0]
            k = min(b - 1, a + 4)      # 20 ms after speech stops
        else:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            k = (a + b) // 2
    else:
        k = int(np.argmin(seg))
    return (i0 + k) / 200, float(seg[k])


# label, start window, end window, gap before (s), speaker, [inside cuts as (win_a, win_b)]
SEGMENTS = [
    ("cold_open", (137.10, 137.19), (142.20, 143.10), 0.70, "DOKTOR", []),
    ("host_q", (0.0, 0.0), (4.55, 5.55), 1.25, "HOS", []),
    ("host_mamak", (10.30, 10.42), (18.55, 19.55), 0.35, "HOS", []),
    ("doc_avoid", (49.98, 50.12), (63.75, 64.15), 0.50, "DOKTOR", []),
    ("doc_shutdown", (68.15, 68.50), (78.62, 79.12), 0.40, "DOKTOR", []),
    ("doc_child", (78.95, 79.13), (93.80, 94.02), 0.0, "DOKTOR", []),
    ("doc_two", (93.85, 94.02), (103.25, 103.72), 0.0, "DOKTOR", []),
    ("host_time", (104.40, 104.56), (110.60, 110.70), 0.45, "HOS", []),
    ("doc_noNumber", (124.85, 125.15), (131.60, 131.70), 0.40, "DOKTOR", []),
    ("doc_key", (137.10, 137.19), (142.20, 143.10), 0.95, "DOKTOR", []),
    ("doc_healthy", (142.55, 143.20), (153.00, 153.30), 0.55, "DOKTOR", []),  # ends after "Ruang yang sihat sebenarnya"
    ("doc_avoidance", (153.00, 153.35), (167.28, 167.42), 0.0, "DOKTOR",
     [((161.00, 161.40), (163.20, 163.35))]),  # "Itu bukan cooling off lah kita panggil" (clean gaps)
    ("doc_final", (167.42, 167.70), (175.10, 175.50), 0.70, "DOKTOR", []),
]
TAIL = 1.6            # short hold after the last word (no end card)
MIN_PAUSE = 0.42      # pauses longer than this...
KEEP_PAUSE = 0.30     # ...are shortened to this
SIL_DB = -48

pieces, out, t_out = [], [], 0.0
segments_meta, cut_log = [], []


def fade_len(level_db):
    return 0.006 if level_db < -45 else 0.018 if level_db < -30 else 0.035


def add_silence(d):
    global t_out
    n = int(round(d * SR))
    out.append(np.zeros(n))
    t_out += n / SR


def add_audio(a, b, la, lb, seg_label, speaker, gain):
    global t_out
    ia, ib = int(round(a * SR)), int(round(b * SR))
    chunk = x[ia:ib].copy() * gain
    fa, fb = int(fade_len(la) * SR), int(fade_len(lb) * SR)
    if len(chunk) > fa + fb:
        chunk[:fa] *= np.sin(np.linspace(0, np.pi / 2, fa)) ** 2
        chunk[-fb:] *= np.cos(np.linspace(0, np.pi / 2, fb)) ** 2
    pieces.append(dict(src_a=a, src_b=b, out_a=t_out, seg=seg_label, speaker=speaker))
    out.append(chunk)
    t_out += len(chunk) / SR


def split_pauses(a, b):
    """Shorten only long true-silence pauses inside [a,b]; return [(a,b,level_a,level_b)]."""
    i0, i1 = int(a * 200), int(b * 200)
    quiet = db[i0:i1] < SIL_DB
    ranges, start, i = [], a, 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            dur = (j - i) / 200
            if dur > MIN_PAUSE and i > 0 and j < len(quiet):
                keep = KEEP_PAUSE / 2
                ranges.append((start, a + i / 200 + keep))
                start = a + j / 200 - keep
            i = j
        else:
            i += 1
    ranges.append((start, b))
    return ranges


def level(t):
    return float(db[int(t * 200)])


def speech_rms(a, b):
    seg = x[int(a * SR):int(b * SR)]
    fr = seg[: len(seg) // HOP * HOP].reshape(-1, HOP)
    r = np.sqrt((fr ** 2).mean(1) + 1e-12)
    loud = r[20 * np.log10(r) > -32]
    return float(np.sqrt((loud ** 2).mean())) if len(loud) else 1e-3


# segment boundaries
bounds = []
for label, ws, we, gap, spk, cuts in SEGMENTS:
    a, la = (ws[0], level(ws[0])) if ws[1] == ws[0] else cut_point(*ws, prefer="late")
    b, lb = cut_point(*we, prefer="early")
    bounds.append((a, la, b, lb))
TARGET = np.median([speech_rms(a, b) for a, _, b, _ in bounds])

for (label, ws, we, gap, spk, cuts), (a, la, b, lb) in zip(SEGMENTS, bounds):
    gain = float(np.clip(TARGET / speech_rms(a, b), 10 ** (-5 / 20), 10 ** (5 / 20)))
    add_silence(gap)
    seg_out_a = t_out
    spans, cur, cur_l = [], a, la
    for wa, wb in cuts:
        ca, lca = cut_point(*wa, prefer="early")
        cb, lcb = cut_point(*wb, prefer="late")
        spans.append((cur, ca, cur_l, lca))
        cut_log.append((label, ca, lca, cb, lcb))
        cur, cur_l = cb, lcb
    spans.append((cur, b, cur_l, lb))
    for sa, sb, l_a, l_b in spans:
        sub = split_pauses(sa, sb)
        for k, (ra, rb) in enumerate(sub):
            if rb - ra > 0.02:
                add_audio(ra, rb, l_a if k == 0 else -90, l_b if k == len(sub) - 1 else -90, label, spk, gain)
    segments_meta.append(dict(label=label, speaker=spk, out_a=seg_out_a, out_b=t_out, src_a=a, src_b=b,
                              gain_db=round(20 * np.log10(gain), 2), cut_db=(round(la, 1), round(lb, 1))))

add_silence(TAIL)
y = np.concatenate(out)
sf.write("voice_edit_raw.wav", y, SR, subtype="PCM_24")
inside = [(ca, cb) for _, ca, _, cb, _ in cut_log]
json.dump(dict(pieces=pieces, segments=segments_meta, duration=t_out, inside_cuts=inside), open("edit_map.json", "w"),
          indent=1)
print(f"duration {t_out:.2f}s")
for s in segments_meta:
    print(f"{s['label']:14s} {s['out_a']:7.2f}-{s['out_b']:7.2f} src {s['src_a']:.3f}-{s['src_b']:.3f} "
          f"cut dB {s['cut_db']} gain {s['gain_db']:+.1f} dB")
for c in cut_log:
    print("inside cut", c)
