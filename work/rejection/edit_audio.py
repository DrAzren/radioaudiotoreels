"""Build the edited voice track for the REJECTION episode (same cut rules as the SPACE reel).

Each segment is a range of the ORIGINAL audio. No words are generated or re-ordered.
  * every cut point is placed in the quietest 5 ms frame *between two words* inside an
    explicit search window, preferring real silence; fades scale with how loud the cut is
  * no mid-sentence trims; the two inside cuts remove whole sentences at sentence gaps
  * only long pauses are tightened, and gently
  * each speaker run is level-matched (+-5 dB); where the host hands over to the doctor inside one
    contiguous source range the gain is ramped in the gap, so there is no cut there at all
Outputs voice_edit_raw.wav and edit_map.json (src time -> output time).
"""
import json

import numpy as np
import soundfile as sf

SR = 48000
x, sr = sf.read("../src48m.wav")
assert sr == SR
HOP = SR // 200  # 5 ms
db = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(HOP) / HOP, "same")[::HOP] + 1e-12))

# label, start window, end window, gap before (s), speaker, [inside cuts as (win_a, win_b)]
# Windows were chosen from the 5 ms level contour (contour.py); Whisper word times here run ~0.2-0.3 s early.
SEGMENTS = [
    # cold open (flash-forward, plays again in context): "Bila pasangan tinggalkan kita, timbullah pemikiran
    # apa yang kurang pada diri aku" - starts after "Tapi", ends before "ini kan"
    ("cold_open", (153.47, 153.62), (158.47, 158.67), 0.40, "DOKTOR", []),
    ("host_q1", (0.0, 0.0), (1.30, 1.42), 1.30, "HOS", []),                  # "How to handle rejection?"
    # "Doktor, kenapa rejection dan cemburu ... sangat kuat?" + the doctor's answer (contiguous in the source)
    ("host_q2_doc", (59.98, 60.38), (102.30, 102.60), 0.45, "HOS", []),   # ends "...dalam satu masa itu."
    ("host_kids", (123.58, 123.80), (126.18, 126.50), 0.55, "HOS", []),      # "Nak ajar anak-anak untuk menerima rejection."
    ("doc_parents", (126.96, 127.22), (189.17, 189.30), 0.45, "DOKTOR",
     [((140.82, 140.99), (142.25, 142.40)),    # drop unclear "Sebab sebenarnya lagi yang daripada kita."
      ((143.15, 143.62), (145.05, 145.20)),    # drop filler "apa panggil" ("Kita ni mungkin [..] meletakkan"); silence both sides
      ((176.15, 176.45), (182.74, 182.86))]),  # drop "Kita tak boleh expect yang kita boleh, apa mereka ... tak stable."
]
SPEAKER_AT = [(67.08, "DOKTOR")]  # speaker change inside a contiguous segment (source time)
TAIL = 1.6            # short hold after the last word (no end card)
MIN_PAUSE = 0.42      # pauses longer than this...
KEEP_PAUSE = 0.30     # ...are shortened to this
SIL_DB = -48


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


def fade_len(level_db):
    return 0.006 if level_db < -45 else 0.018 if level_db < -30 else 0.035


def level(t):
    return float(db[int(t * 200)])


def speech_rms(a, b):
    seg = x[int(a * SR):int(b * SR)]
    fr = seg[: len(seg) // HOP * HOP].reshape(-1, HOP)
    r = np.sqrt((fr ** 2).mean(1) + 1e-12)
    loud = r[20 * np.log10(r) > -32]
    return float(np.sqrt((loud ** 2).mean())) if len(loud) else 1e-3


def speaker_runs(label, spk, a, b):
    """[(a, b, speaker)] split at SPEAKER_AT changes."""
    runs, cur, cur_spk = [], a, spk
    for ts, s2 in SPEAKER_AT:
        if a < ts < b:
            runs.append((cur, ts, cur_spk))
            cur, cur_spk = ts, s2
    runs.append((cur, b, cur_spk))
    return runs


pieces, out, t_out = [], [], 0.0
segments_meta, cut_log = [], []


def add_silence(d):
    global t_out
    n = int(round(d * SR))
    out.append(np.zeros(n))
    t_out += n / SR


def add_audio(a, b, la, lb, seg_label, gain_fn, spk_fn):
    """gain_fn(src_times) -> gain per sample; pieces are split at speaker changes for the speaker map."""
    global t_out
    ia, ib = int(round(a * SR)), int(round(b * SR))
    ts = np.arange(ia, ib) / SR
    chunk = x[ia:ib] * gain_fn(ts)
    fa, fb = int(fade_len(la) * SR), int(fade_len(lb) * SR)
    if len(chunk) > fa + fb:
        chunk[:fa] *= np.sin(np.linspace(0, np.pi / 2, fa)) ** 2
        chunk[-fb:] *= np.cos(np.linspace(0, np.pi / 2, fb)) ** 2
    bounds = [a] + [t for t, _ in SPEAKER_AT if a < t < b] + [b]
    for p, q in zip(bounds, bounds[1:]):
        pieces.append(dict(src_a=p, src_b=q, out_a=t_out + (p - a), seg=seg_label, speaker=spk_fn((p + q) / 2)))
    out.append(chunk)
    t_out += len(chunk) / SR


def split_pauses(a, b):
    """Shorten only long true-silence pauses inside [a,b]; return [(a,b)]."""
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


# segment boundaries
bounds = []
for label, ws, we, gap, spk, cuts in SEGMENTS:
    a, la = (ws[0], level(ws[0]) if ws[0] > 0 else -90) if ws[1] == ws[0] else cut_point(*ws, prefer="late")
    b, lb = cut_point(*we, prefer="early")
    bounds.append((a, la, b, lb))
TARGET = np.median([speech_rms(r0, r1) for (lab, _, _, _, spk, _), (a, _, b, _) in zip(SEGMENTS, bounds)
                    for r0, r1, _ in speaker_runs(lab, spk, a, b)])

for (label, ws, we, gap, spk, cuts), (a, la, b, lb) in zip(SEGMENTS, bounds):
    runs = speaker_runs(label, spk, a, b)
    gains = [(r0, r1, s, float(np.clip(TARGET / speech_rms(r0, r1), 10 ** (-5 / 20), 10 ** (5 / 20))))
             for r0, r1, s in runs]

    def gain_fn(ts, gains=gains):
        g = np.full(len(ts), gains[0][3])
        for (_, r1, _, g0), (_, _, _, g1) in zip(gains, gains[1:]):
            g = np.where(ts >= r1, g1, g)
            ramp = (ts > r1 - 0.03) & (ts < r1 + 0.03)          # 60 ms gain ramp inside the hand-over gap
            g[ramp] = g0 + (g1 - g0) * (ts[ramp] - (r1 - 0.03)) / 0.06
        return g

    def spk_fn(t, runs=runs):
        return [s for r0, r1, s in runs if r0 <= t < r1 + 1e-6][0]

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
                add_audio(ra, rb, l_a if k == 0 else -90, l_b if k == len(sub) - 1 else -90, label, gain_fn, spk_fn)
    segments_meta.append(dict(label=label, speaker=spk, out_a=seg_out_a, out_b=t_out, src_a=a, src_b=b,
                              gains_db=[(round(r0, 2), s, round(20 * np.log10(g), 2)) for r0, _, s, g in gains],
                              cut_db=(round(la, 1), round(lb, 1))))

add_silence(TAIL)
y = np.concatenate(out)
sf.write("voice_edit_raw.wav", y, SR, subtype="PCM_24")
inside = [(ca, cb) for _, ca, _, cb, _ in cut_log]
json.dump(dict(pieces=pieces, segments=segments_meta, duration=t_out, inside_cuts=inside,
               speaker_at=SPEAKER_AT), open("edit_map.json", "w"), indent=1)
print(f"duration {t_out:.2f}s  pieces {len(pieces)}")
for s in segments_meta:
    print(f"{s['label']:12s} {s['out_a']:7.2f}-{s['out_b']:7.2f} src {s['src_a']:.3f}-{s['src_b']:.3f} "
          f"cut dB {s['cut_db']} gains {s['gains_db']}")
for c in cut_log:
    print("inside cut %s  out@%.3f (%.1f dB) -> in@%.3f (%.1f dB)" % c)
