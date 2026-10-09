"""Scene timeline + frame renderer for the REJECTION reel (same visual language as the SPACE reel).

All times are reel OUTPUT seconds (captions.json / edit_map.json; scene cues are written against the
word timings of the final edit, so no time-warp is needed).
Usage:  python3 reel.py <start_frame> <end_frame> <out.mp4>
        python3 reel.py --still <t> <out.png>
"""
import functools
import json
import math
import os
import subprocess
import sys

EP = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("BROLL_DIR", os.path.join(EP, "broll"))
sys.path.insert(0, os.path.dirname(EP))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from gfx import *  # noqa: E402,F401,F403

CAP = json.load(open(os.path.join(EP, "captions.json")))
DUR = json.load(open(os.path.join(EP, "edit_map.json")))["duration"]

# ------------------------------------------------------------- audio analysis (for waveforms)
_v, _sr = sf.read(os.path.join(EP, "voice_edit_raw.wav"))
if _v.ndim > 1:
    _v = _v.mean(1)
_hop = _sr // FPS
_nfr = len(_v) // _hop + 2
ENV = np.zeros(_nfr, np.float32)
BANDS = np.zeros((_nfr, 28), np.float32)
_win = np.hanning(2048)
_edges = np.geomspace(90, 5000, 29)
_freqs = np.fft.rfftfreq(2048, 1 / _sr)
_bidx = [(_freqs >= _edges[b]) & (_freqs < _edges[b + 1]) for b in range(28)]
for i in range(_nfr):
    seg = _v[i * _hop: i * _hop + 2048]
    if len(seg) < 2048:
        seg = np.pad(seg, (0, 2048 - len(seg)))
    ENV[i] = np.sqrt(np.mean(seg[:_hop] ** 2) + 1e-12)
    sp = np.abs(np.fft.rfft(seg * _win))
    for b in range(28):
        BANDS[i, b] = sp[_bidx[b]].mean() if _bidx[b].any() else 0
ENV = np.clip(ENV / (np.percentile(ENV, 98) + 1e-9), 0, 1.2)
BANDS = np.clip(BANDS / (np.percentile(BANDS, 98, axis=0, keepdims=True) + 1e-9), 0, 1.3)
for i in range(1, _nfr):
    BANDS[i] = np.maximum(BANDS[i], BANDS[i - 1] * 0.78)


def env_at(t):
    return float(ENV[min(len(ENV) - 1, max(0, int(t * FPS)))])


def bands_at(t):
    return BANDS[min(len(BANDS) - 1, max(0, int(t * FPS)))]


# ------------------------------------------------------------- cached graded sources + helpers
@functools.lru_cache(maxsize=None)
def G(name, look="neutral"):
    return grade(load(name), look)


def shot(name, t, t0, t1, z0, z1, c0, c1, look="neutral", rot0=0, rot1=0, ow=W, oh=H):
    p = ease_io(prog(t, t0, t1))
    return camera(G(name, look), lerp(z0, z1, p), lerp(c0[0], c1[0], p), lerp(c0[1], c1[1], p),
                  lerp(rot0, rot1, p), ow, oh)


def text(frame, s, fname, size, cx, cy, alpha=1.0, color=WHITE, scale=1.0, tracking=0, anchor="c", shadow=0.6,
         shadow_r=12):
    if alpha <= 0.004:
        return frame
    return blit(frame, text_sprite(s, fname, size, color, tracking, shadow, shadow_r), cx, cy, alpha, scale, anchor)


def slam(frame, s, fname, size, cx, cy, t, t0, color=WHITE, dur=0.22, out_t=None, out_d=0.25, s0=1.45, tracking=0):
    if t < t0:
        return frame
    p = prog(t, t0, t0 + dur)
    a = ease_out(p)
    if out_t is not None:
        a *= 1 - ease_in(prog(t, out_t, out_t + out_d))
    return text(frame, s, fname, size, cx, cy, a, color, lerp(s0, 1.0, ease_out(p)), tracking)


def rise(frame, s, fname, size, cx, cy, t, t0, color=WHITE, dur=0.35, out_t=None, out_d=0.3, dy=40, tracking=0,
         anchor="c", alpha=1.0):
    if t < t0:
        return frame
    p = ease_out(prog(t, t0, t0 + dur))
    a = p * alpha
    if out_t is not None:
        a *= 1 - ease_in(prog(t, out_t, out_t + out_d))
    return text(frame, s, fname, size, cx, cy + (1 - p) * dy, a, color, 1.0, tracking, anchor)


def pill(frame, s, cx, cy, t, t0, fg=WHITE, bg=INK, bga=0.75, size=30, fname="inter7", out_t=None, dot=None,
         border=None, anchor="c", dur=0.3, scale=1.0):
    if t < t0:
        return frame
    p = ease_out(prog(t, t0, t0 + dur))
    a = p
    if out_t is not None:
        a *= 1 - ease_in(prog(t, out_t, out_t + 0.3))
    spr = pill_sprite(s, fname, size, fg, bg, bga, dot=dot, border=border)
    return blit(frame, spr, cx, cy + (1 - p) * 16, a, scale * lerp(0.92, 1.0, p), anchor)


def gradient_v(frame, y0, y1, strength, top_dark=False):
    y0, y1 = int(max(0, y0)), int(min(H, y1))
    n = y1 - y0
    if n <= 0:
        return frame
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    if top_dark:
        ramp = ramp[::-1]
    frame[y0:y1] *= (1 - (ramp ** 1.3) * strength)[:, None, None]
    return frame


def bg_dark(name, look, t, t0, t1, sigma=16, dk=0.6, z=(1.1, 1.18), c=(0.5, 0.45)):
    f = shot(name, t, t0, t1, z[0], z[1], c, (c[0], c[1] - 0.03), look)
    small = blur(cv2.resize(f, (W // 4, H // 4)), sigma / 4)
    return darken(cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR), dk)


def fit_card(name, look, w, h_box, t, t0, t1, z0=1.0, z1=1.06, cx=0.5):
    """Fit-to-width image with blurred cover fill, for split panels (w x h_box)."""
    src = G(name, look)
    z = lerp(z0, z1, ease_io(prog(t, t0, t1)))
    bg = camera(src, 1.0, cx, 0.5, 0, w, h_box)
    bg = cv2.resize(darken(blur(cv2.resize(bg, (w // 4, h_box // 4)), 6), 0.45), (w, h_box))
    ch = int(w * 1920 / 1080)
    card = camera(src, z, cx, 0.5, 0, w, ch)
    y0 = (h_box - ch) // 2
    bg[y0:y0 + ch] = card
    for k in range(18):
        a = (18 - k) / 18 * 0.5
        bg[y0 + k] *= (1 - a)
        bg[y0 + ch - 1 - k] *= (1 - a)
    return bg


def heart_rings(f, t, t0, t1, cx, cy, period=0.8, alpha=0.13):
    for b in np.arange(t0, t1, period):
        if b <= t < b + period:
            p = (t - b) / period
            ov = f.copy()
            cv2.circle(ov, (int(cx), int(cy)), int(140 + 420 * p), (1, 1, 1), 3, cv2.LINE_AA)
            f = over(f, ov, alpha * (1 - p))
    return f


# ------------------------------------------------------------- radio graphics + speaker tags
def onair(frame, t, t0, t1):
    a = window(t, t0, t1, 0.3, 0.3)
    if a <= 0:
        return frame
    spr = pill_sprite("ON AIR", "inter8", 32, WHITE, (0.75, 0.08, 0.08), 0.92, dot=(1, 1, 1))
    return blit(frame, spr, 60, 176, a, 1.0, "l")


DOCTOR = "DR FAKRUL AZREN BIN AZHAR"
SPEAKER_TAGS = [  # (t0, t1, label, dot) -- small tags
    (0.40, 3.10, DOCTOR, TEAL),
    (6.55, 15.20, "HOS RADIO", CORAL),
    (51.10, 53.70, "HOS RADIO", CORAL),
    (54.25, 57.90, DOCTOR, TEAL),
    (72.75, 76.40, DOCTOR, TEAL),
]
NAME_CARD = (15.30, 19.30)  # full lower-third when the doctor is first heard in context
ONAIR = [(6.55, 9.30), (51.10, 53.70)]


def speaker_tags(frame, t):
    for t0, t1, lab, dot in SPEAKER_TAGS:
        a = window(t, t0, t1, 0.3, 0.35)
        if a > 0:
            x = 60 + (240 if any(a0 <= t <= a1 for a0, a1 in ONAIR) else 0)
            frame = blit(frame, pill_sprite(lab, "inter7", 32, WHITE, INK, 0.62, dot=dot), x, 176, a, 1.0, "l")
    for a0, a1 in ONAIR:
        frame = onair(frame, t, a0, a1)
    a = window(t, NAME_CARD[0], NAME_CARD[1], 0.35, 0.4)
    if a > 0:
        p = ease_out(prog(t, NAME_CARD[0], NAME_CARD[0] + 0.45))
        x = 60 - (1 - p) * 40
        frame = gradient_v(frame, 0, 420, 0.55 * a, top_dark=True)
        frame = blit(frame, rrect_sprite(8, 104, 4, TEAL, 1.0), x, 150, a, 1.0, "tl")
        frame = text(frame, DOCTOR, "mont8", 40, x + 26, 180, a, WHITE, anchor="l", shadow=0.7)
        frame = text(frame, "Tetamu · Temu bual radio", "inter5", 30, x + 26, 230, a, DIM, anchor="l", shadow=0.7)
    return frame


def waveform_bars(frame, t, cx, cy, width, height, alpha, color=WHITE, n=28):
    if alpha <= 0:
        return frame
    b = bands_at(t)
    ov = frame.copy()
    gap = width / n
    bw = max(3, int(gap * 0.5))
    for i in range(n):
        hh = max(4, float(b[i % len(b)]) * height * (0.55 + 0.45 * math.sin(math.pi * (i + 0.5) / n)))
        x = int(cx - width / 2 + i * gap + gap / 2)
        cv2.line(ov, (x, int(cy - hh / 2)), (x, int(cy + hh / 2)), tuple(float(c) for c in color), bw, cv2.LINE_AA)
    return over(frame, ov, alpha)


def waveform_line(frame, t, y, alpha, color=WHITE, amp=90, synth_amt=1.0):
    if alpha <= 0:
        return frame
    ov = frame.copy()
    xs = np.arange(0, W + 1, 6)
    e = env_at(t) + 0.25 * synth_amt
    ph = t * 7
    win = np.sin(np.pi * xs / W) ** 2
    ys = y + amp * e * win * (np.sin(xs * 0.035 + ph) * 0.6 + np.sin(xs * 0.011 - ph * 0.7) * 0.4)
    cv2.polylines(ov, [np.stack([xs, ys], 1).astype(np.int32)], False, tuple(float(c) for c in color), 4,
                  cv2.LINE_AA)
    glow = blur(ov - frame, 8)
    frame = over(frame, ov, alpha)
    return np.clip(frame + glow * 0.8 * alpha, 0, 1)


def check_row(f, t, t0, s, cx, cy, ok=True, size=60):
    """'✓ text' / '✗ text' row that pops in."""
    if t < t0:
        return f
    p = ease_out(prog(t, t0, t0 + 0.3))
    mark = text_sprite("✓" if ok else "✗", "dejavu", int(size * 0.9), TEAL if ok else CORAL, shadow=0.6,
                       shadow_r=8)
    lab = text_sprite(s, "anton", size, WHITE, shadow=0.6, shadow_r=8)
    spr = hstack([mark, lab], -18)
    box = rrect_sprite(spr.shape[1] + 10, int(size * 1.55), 22, INK, 0.62)
    f = blit(f, box, cx, cy + (1 - p) * 20, p)
    return blit(f, spr, cx, cy + (1 - p) * 20, p, lerp(0.9, 1.0, p))


# ------------------------------------------------------------- scenes
def sc_hook_bed(t):
    f = shot("bed", t, 0.0, 3.0, 1.2, 1.32, (0.42, 0.36), (0.44, 0.38), "cool")
    f = desat(f, 0.35)
    f = darken(f, 0.18)
    f = gradient_v(f, 0, 800, 0.6, top_dark=True)
    f = slam(f, "BILA PASANGAN", "mont8", 64, W / 2, 470, t, 0.47, WHITE, s0=1.25, tracking=6, out_t=2.75)
    f = slam(f, "TINGGALKAN", "anton", 170, W / 2, 610, t, 0.93, WHITE, s0=1.35, out_t=2.75)
    f = slam(f, "KITA…", "anton", 170, W / 2, 770, t, 1.35, CORAL, s0=1.35, out_t=2.75)
    return gradient_v(f, 1150, H, 0.55)


def sc_hook_mirror(t):
    f = shot("mirror", t, 2.9, 5.8, 1.25, 1.12, (0.58, 0.36), (0.56, 0.38), "cool")
    f = desat(f, 0.3)
    f = darken(f, 0.2)
    f = gradient_v(f, 0, 900, 0.65, top_dark=True)
    f = rise(f, "TIMBUL SATU PERSOALAN", "mont8", 38, W / 2, 330, t, 3.05, DIM, tracking=8)
    f = slam(f, "APA YANG KURANG", "anton", 128, W / 2, 500, t, 4.37, WHITE, s0=1.3)
    f = slam(f, "PADA DIRI AKU?", "anton", 150, W / 2, 650, t, 4.89, CORAL, s0=1.35)
    return gradient_v(f, 1150, H, 0.55)


def sc_title(t):
    f = fill(INK)
    f += radial_glow(CORAL, W / 2, 960, 560, 0.09)
    f = waveform_line(f, t, 1010, window(t, 5.5, 6.75, 0.15, 0.2), CORAL, amp=70, synth_amt=1.4)
    f = rise(f, "BILA DITOLAK…", "anton", 150, W / 2, 720, t, 5.6, dy=30)
    f = rise(f, "KENAPA EMOSINYA", "anton", 108, W / 2, 1130, t, 5.85, AMBER, dy=30)
    f = rise(f, "SANGAT KUAT?", "anton", 108, W / 2, 1250, t, 5.97, AMBER, dy=30)
    return f


def sc_radio(t):
    f = shot("mic", t, 6.55, 9.35, 1.02, 1.12, (0.42, 0.5), (0.45, 0.48), "neutral")
    f = darken(f, 0.1)
    f = waveform_bars(f, t, W / 2, 1150, 760, 150, 0.85 * window(t, 6.55, 9.35, 0.4, 0.3), WHITE)
    f = rise(f, "TEMU BUAL RADIO", "mont8", 34, W / 2, 1255, t, 6.8, DIM, tracking=10, out_t=9.0)
    return gradient_v(f, 1200, H, 0.55)


def sc_kombo(t):
    f = bg_dark("car", "cool", t, 9.3, 15.3, dk=0.62)
    f = rise(f, "SOALAN HOS", "mono7", 32, W / 2, 330, t, 9.35, DIM, tracking=10)
    f = pill(f, "REJECTION", W / 2, 470, t, 9.45, WHITE, CORAL, 0.95, 64, "anton", dur=0.25)
    f = rise(f, "+", "anton", 110, W / 2, 610, t, 10.2, DIM)
    f = pill(f, "CEMBURU", W / 2, 750, t, 10.38, INK, AMBER, 0.95, 64, "anton", dur=0.25)
    f = rise(f, "=", "anton", 110, W / 2, 890, t, 12.6, DIM)
    if t >= 12.82:
        p = ease_out(prog(t, 12.82, 13.1))
        glow = 0.12 * p * (1 + 0.3 * math.sin(t * 6))
        f += radial_glow(CORAL, W / 2, 1060, 420, glow)
        f = text(f, "KOMBO EMOSI", "anton", 150, W / 2, 1050, p, WHITE, lerp(1.3, 1.0, p))
    f = rise(f, "YANG SANGAT KUAT?", "mont8", 46, W / 2, 1180, t, 14.64, CORAL, tracking=6)
    return f


def sc_doc_intro(t):
    f = shot("car", t, 15.3, 19.4, 1.08, 1.18, (0.4, 0.42), (0.42, 0.4), "cool")
    f = darken(f, 0.12)
    return gradient_v(f, 1150, H, 0.55)


def sc_core(t):
    f = fill(INK)
    f += radial_glow(TEAL, W / 2, 690, 600, 0.07)
    cx, cy = W // 2, 690
    ov = f.copy()
    breathe = 1 + 0.03 * math.sin(t * 2.4)
    for k, (r, t0) in enumerate([(380, 19.35), (285, 19.55), (190, 19.75)]):
        p = ease_out(prog(t, t0, t0 + 0.5))
        if p > 0:
            cv2.circle(ov, (cx, cy), int(r * p * breathe), (0.45, 0.48, 0.52), 3, cv2.LINE_AA)
    f = over(f, ov, 0.9)
    pc = ease_back(prog(t, 20.3, 20.65))
    if pc > 0:
        core = rrect_sprite(200, 200, 100, TEAL, 0.95)
        f = blit(f, core, cx, cy, clamp(pc), max(0.01, pc))
        f = text(f, "DIRI", "anton", 64, cx, cy - 26, clamp(pc), INK, max(0.01, pc), shadow=0)
        f = text(f, "KITA", "anton", 64, cx, cy + 34, clamp(pc), INK, max(0.01, pc), shadow=0)
    # two emotions reaching in toward the core
    for lab, col, side, t0 in [("REJECTION", CORAL, -1, 18.0), ("CEMBURU", AMBER, 1, 18.4)]:
        p = ease_io(prog(t, t0 + 1.4, t0 + 3.4))
        a = window(t, t0 + 1.3, 23.9, 0.3, 0.3)
        x = cx + side * lerp(380, 230, p)
        y = cy + side * lerp(-470, -330, p)
        f = pill(f, lab, x, y, t, t0 + 1.3, WHITE if col == CORAL else INK, col, 0.92 * a, 40, "anton")
    f = rise(f, "SANGAT DEKAT", "anton", 110, W / 2, 1130, t, 21.98, WHITE, dy=30)
    f = rise(f, "DENGAN DIRI KITA", "mont8", 50, W / 2, 1235, t, 22.6, TEAL, tracking=6)
    return f


def sc_needs(t):
    f = shot("couple", t, 23.85, 29.2, 1.05, 1.16, (0.5, 0.4), (0.52, 0.38), "warm")
    f = gradient_v(f, 0, 760, 0.55, top_dark=True)
    f = rise(f, "KEPERLUAN ASAS DALAM HUBUNGAN", "mont8", 30, W / 2, 300, t, 24.0, DIM, tracking=6)
    f = pill(f, "RASA DITERIMA", W / 2, 420, t, 25.0, INK, WHITE, 0.94, 48, "anton")
    f = pill(f, "RASA DISAYANGI", W / 2, 540, t, 26.12, INK, AMBER, 0.94, 48, "anton")
    f = pill(f, "RASA SELAMAT", W / 2, 660, t, 27.48, INK, TEAL, 0.94, 48, "anton")
    return gradient_v(f, 1150, H, 0.55)


def sc_loss(t):
    f = shot("platform", t, 29.15, 35.3, 1.04, 1.16, (0.55, 0.45), (0.6, 0.42), "cool")
    f = desat(f, 0.2 * ease_out(prog(t, 30.7, 31.5)))
    f = gradient_v(f, 0, 820, 0.6, top_dark=True)
    f = slam(f, "REJECTION", "anton", 170, W / 2, 470, t, 30.72, CORAL, s0=1.3, out_t=31.9)
    f = rise(f, "RASA", "mont8", 54, W / 2, 400, t, 32.1, DIM, tracking=10)
    f = rise(f, "KEHILANGAN", "anton", 140, W / 2, 520, t, 32.8, WHITE, dy=30)
    f = rise(f, "ATAUPUN", "mont8", 40, W / 2, 640, t, 33.5, DIM, tracking=10)
    f = slam(f, "DITOLAK.", "anton", 150, W / 2, 790, t, 34.34, CORAL, s0=1.25)
    return gradient_v(f, 1150, H, 0.55)


def sc_jealous(t):
    punch = ease_io(prog(t, 39.9, 40.3))
    f = shot("cafe", t, 35.25, 43.95, lerp(1.05, 1.18, punch), lerp(1.1, 1.24, punch), (0.35, 0.42), (0.33, 0.4),
             "neutral")
    f = gradient_v(f, 0, 900, 0.62, top_dark=True)
    f = slam(f, "CEMBURU", "anton", 170, W / 2, 330, t, 36.7, AMBER, s0=1.3)
    items = [("TAKUT DITINGGALKAN", 38.22), ("RASA DIBANDINGKAN", 40.5), ("TAK CUKUP PENTING", 43.0)]
    for i, (s, t0) in enumerate(items):
        if t >= t0:
            p = ease_out(prog(t, t0, t0 + 0.3))
            spr = text_sprite(s, "anton", 64, WHITE, shadow=0.6, shadow_r=8)
            box = rrect_sprite(spr.shape[1] + 6, 100, 20, INK, 0.62)
            y = 500 + i * 125
            f = blit(f, box, W / 2 + (1 - p) * 60, y, p)
            f = blit(f, spr, W / 2 + (1 - p) * 60, y, p)
    return gradient_v(f, 1150, H, 0.55)


def sc_collide(t):
    f = fill(INK)
    p = ease_in(prog(t, 43.95, 45.0))
    hit = t >= 45.0
    xL, xR = lerp(250, W / 2 - 120, p), lerp(W - 250, W / 2 + 120, p)
    shake = 8 * math.exp(-max(0, t - 45.0) * 12) * math.sin(t * 95) if hit else 0
    f += radial_glow(CORAL, W / 2, 800, 500, 0.18 * ease_out(prog(t, 45.0, 45.15)) * (1 - prog(t, 45.2, 46.2)) + 0.04)
    f = pill(f, "REJECTION", xL + shake, 800, t, 43.95, WHITE, CORAL, 0.95, 52, "anton", dur=0.2)
    f = pill(f, "CEMBURU", xR + shake, 800, t, 43.95, INK, AMBER, 0.95, 52, "anton", dur=0.2)
    f = slam(f, "SERENTAK", "anton", 180, W / 2 + shake, 1050, t, 45.46, WHITE, s0=1.3)
    return f


EMO = [("SEDIH", 47.52, 250, 470, WHITE), ("MARAH", 48.2, 800, 600, CORAL), ("MALU", 48.76, 300, 760, WHITE),
       ("TAKUT", 49.24, 780, 900, CORAL)]


def sc_overload(t):
    f = shot("floor", t, 46.0, 50.95, 1.06, 1.18, (0.55, 0.42), (0.56, 0.4), "cool")
    load_ = ease_io(prog(t, 47.4, 49.6))
    f = desat(f, 0.2 + 0.4 * load_)
    f = darken(f, 0.15 + 0.2 * load_)
    f = gradient_v(f, 0, 1000, 0.5, top_dark=True)
    f = heart_rings(f, t, 46.2, 50.9, 560, 760, 0.8 - 0.25 * load_, 0.14)
    for s, t0, x, y, col in EMO:
        if t >= t0:
            jitter = 6 * math.sin(t * 13 + x) * load_
            f = slam(f, s, "anton", 150, x + jitter, y, t, t0, col, s0=1.35)
    f = rise(f, "DALAM SATU MASA", "mont8", 50, W / 2, 1060, t, 49.74, AMBER, tracking=8)
    return gradient_v(f, 1150, H, 0.55)


def sc_host_kids(t):
    f = shot("mother", t, 50.9, 54.15, 1.12, 1.22, (0.6, 0.48), (0.62, 0.46), "warm")
    f = gradient_v(f, 0, 600, 0.5, top_dark=True)
    f = rise(f, "MENERIMA REJECTION", "anton", 96, W / 2, 470, t, 52.5, WHITE, dy=26)
    return gradient_v(f, 1150, H, 0.55)


def sc_doc_mic(t):
    f = shot("mic", t, 54.1, 58.0, 1.15, 1.25, (0.45, 0.5), (0.48, 0.48), "neutral")
    f = darken(f, 0.25)
    f = waveform_bars(f, t, W / 2, 1150, 760, 150, 0.8 * window(t, 54.1, 58.0, 0.3, 0.3), TEAL)
    f = rise(f, "DARI MANA DATANGNYA?", "mont8", 40, W / 2, 1260, t, 54.4, DIM, tracking=8)
    return gradient_v(f, 1200, H, 0.55)


def sc_raised(t):
    f = shot("father", t, 58.0, 62.3, 1.04, 1.14, (0.5, 0.45), (0.5, 0.42), "warm")
    f = gradient_v(f, 0, 820, 0.75, top_dark=True)
    f = rise(f, "CARA KITA", "mont8", 56, W / 2, 380, t, 59.66, WHITE, tracking=10)
    f = rise(f, "DIBESARKAN", "anton", 160, W / 2, 520, t, 60.3, WHITE, dy=30)
    return gradient_v(f, 1150, H, 0.55)


def sc_nilai(t):
    f = shot("mother", t, 62.25, 68.0, 1.3, 1.45, (0.66, 0.5), (0.68, 0.48), "warm")
    f = gradient_v(f, 0, 760, 0.6, top_dark=True)
    f = rise(f, "APA YANG IBU BAPA TERAPKAN", "mont8", 34, W / 2, 330, t, 63.9, DIM, tracking=6)
    if t >= 66.8:
        p = ease_out(prog(t, 66.8, 67.1))
        f += radial_glow(AMBER, W / 2, 500, 380, 0.12 * p)
        f = text(f, "NILAI DIRI", "anton", 190, W / 2, 500, p, AMBER, lerp(1.3, 1.0, p))
    return gradient_v(f, 1150, H, 0.55)


def sc_worth(t):
    f = bg_dark("bed", "cool", t, 67.95, 72.75, dk=0.7)
    f = rise(f, "NILAI DIRI KITA", "mont8", 44, W / 2, 360, t, 68.0, WHITE, tracking=8)
    x0, x1, y0, y1 = 140, 940, 470, 1150
    # container
    ov = f.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (0.5, 0.52, 0.56), 3, cv2.LINE_AA)
    f = over(f, ov, window(t, 68.0, 72.75, 0.3, 0.3))
    # fill from the bottom: relationship share grows to ~80 %
    share = 0.8 * ease_io(prog(t, 69.7, 71.6))
    h = y1 - y0 - 12
    yr = int(y1 - 6 - h * share)
    if share > 0.01:
        f[yr:y1 - 6, x0 + 6:x1 - 6] = f[yr:y1 - 6, x0 + 6:x1 - 6] * 0.15 + np.array(CORAL, np.float32) * 0.85
        f = text(f, "RELATIONSHIP", "anton", 110, W / 2, (yr + y1) / 2 - 20, clamp(share * 3), WHITE)
        f = text(f, f"{int(share * 100)}%", "mono7", 54, W / 2, (yr + y1) / 2 + 80, clamp(share * 3), WHITE)
    rest = y1 - 6 - h * 0.8
    a = clamp((share - 0.6) * 5)
    if a > 0:
        f[y0 + 6:int(rest), x0 + 6:x1 - 6] = f[y0 + 6:int(rest), x0 + 6:x1 - 6] * (1 - 0.6 * a) + \
            np.array(TEAL, np.float32) * 0.6 * a
        f = text(f, "DIRI SENDIRI", "mont8", 40, W / 2, (y0 + rest) / 2, a, INK, shadow=0)
    f = rise(f, "SEBAHAGIAN BESAR", "anton", 80, W / 2, 1270, t, 69.72, AMBER, dy=24)
    return f


def sc_loved(t):
    f = shot("couple", t, 72.7, 77.3, 1.2, 1.3, (0.42, 0.36), (0.44, 0.34), "warm")
    f = gradient_v(f, 0, 900, 0.62, top_dark=True)
    f = rise(f, "BILA PASANGAN SAYANG", "mont8", 44, W / 2, 330, t, 73.45, AMBER, tracking=6)
    f = check_row(f, t, 75.42, "CUKUP BAIK", W / 2, 470)
    f = check_row(f, t, 76.28, "SANGAT BAIK", W / 2, 590)
    f = check_row(f, t, 76.78, "SANGAT SELAMAT", W / 2, 710)
    return gradient_v(f, 1150, H, 0.55)


def sc_left(t):
    f = shot("mirror", t, 77.3, 83.3, 1.05, 1.15, (0.5, 0.42), (0.52, 0.4), "cool")
    f = desat(f, 0.45)
    f = darken(f, 0.22)
    f = gradient_v(f, 0, 900, 0.65, top_dark=True)
    f = rise(f, "BILA PASANGAN PERGI", "mont8", 44, W / 2, 330, t, 77.73, CORAL, tracking=6)
    f = slam(f, "APA YANG KURANG", "anton", 128, W / 2, 500, t, 81.16, WHITE, s0=1.3)
    f = slam(f, "PADA DIRI AKU?", "anton", 150, W / 2, 650, t, 82.02, CORAL, s0=1.35)
    return gradient_v(f, 1150, H, 0.55)


def sc_teach(t):
    f = shot("window", t, 83.3, 89.65, 1.04, 1.14, (0.4, 0.42), (0.42, 0.4), "warm")
    f = gradient_v(f, 0, 760, 0.6, top_dark=True)
    f = rise(f, "SATU BENDA", "anton", 130, W / 2, 430, t, 88.61, WHITE, dy=28)
    f = rise(f, "YANG BOLEH KITA AJAR:", "mont8", 42, W / 2, 540, t, 88.9, AMBER, tracking=6)
    return gradient_v(f, 1150, H, 0.55)


def sc_key(t):
    f = fill(INK)
    pulse = 0.08 + 0.06 * ease_out(prog(t, 93.41, 93.6)) * (1 - prog(t, 93.6, 94.6))
    f += radial_glow(AMBER, W / 2, 1100, 650, pulse)
    shake = 6 * math.exp(-max(0, t - 93.41) * 14) * math.sin(t * 90) if t >= 93.41 else 0
    f = rise(f, "BUKAN SEMUA BENDA", "anton", 120, W / 2, 640, t, 89.72, WHITE, dy=30)
    f = rise(f, "KITA BOLEH", "anton", 120, W / 2, 790, t, 91.33, WHITE, dy=30)
    f = slam(f, "KAWAL.", "anton", 280, W / 2 + shake, 1010, t, 92.56, AMBER, s0=1.35)
    return f


PW = 534
SPLIT_T = 93.8


def panel_L(t):
    p = fit_card("phone_down", "cool", PW, H, t, SPLIT_T, 100.3, cx=0.4)
    p = desat(p, 0.3)
    p = gradient_v(p, 900, 1350, 0.75)
    p = rise(p, "EXPECT", "anton", 84, PW / 2, 1010, t, 94.72, CORAL, dy=20)
    p = rise(p, "PASANGAN AKAN", "mont8", 34, PW / 2, 1095, t, 96.45, WHITE, dy=16)
    p = rise(p, "KEKAL", "anton", 76, PW / 2, 1165, t, 99.11, WHITE, dy=16)
    return rise(p, "SELAMA-LAMANYA", "mont8", 30, PW / 2, 1235, t, 99.3, WHITE, dy=16)


def panel_R(t):
    p = fit_card("journal", "warm", PW, H, t, SPLIT_T, 101.5, cx=0.55)
    p = gradient_v(p, 900, 1350, 0.75)
    p = rise(p, "FOKUS", "anton", 84, PW / 2, 1010, t, 95.4, TEAL, dy=20)
    p = rise(p, "PADA DIRI KITA", "mont8", 34, PW / 2, 1095, t, 95.7, WHITE, dy=16)
    return p


def sc_split(t):
    f = fill(INK)
    po = ease_out(prog(t, SPLIT_T, SPLIT_T + 0.45))
    grow = ease_io(prog(t, 100.17, 100.75))          # "It's all about kita": healthy side takes over
    xl = int(-PW + po * PW - grow * PW)
    xr = int(W - po * PW - grow * (W - PW))
    L, R = panel_L(t), panel_R(t)
    actL = 1.0 if t < 99.9 else 1.0 - 0.6 * ease_out(prog(t, 99.9, 100.3))
    f = paste_rgb(f, L * actL, xl, 0)
    if grow > 0:
        R = cv2.resize(R, (int(PW + grow * (W - PW)), H), interpolation=cv2.INTER_LINEAR)
    f = paste_rgb(f, R, xr, 0)
    ov = f.copy()
    cv2.line(ov, (int(xr), 0), (int(xr), H), (1, 1, 1), 4, cv2.LINE_AA)
    f = over(f, ov, 0.5 * po * (1 - grow))
    f = gradient_v(f, 0, 440, 0.75, top_dark=True)
    a = po * (1 - grow)
    f = text(f, "✗ TAK SIHAT", "dejavu", 40, W / 4, 250, a, CORAL)
    f = text(f, "BERGANTUNG", "anton", 66, W / 4, 320, a, WHITE)
    f = text(f, "PADA PASANGAN", "anton", 50, W / 4, 382, a, WHITE)
    f = text(f, "✓ SIHAT", "dejavu", 40, 3 * W / 4, 250, a, TEAL)
    f = text(f, "DALAM", "anton", 66, 3 * W / 4, 320, a, WHITE)
    f = text(f, "KAWALAN KITA", "anton", 50, 3 * W / 4, 382, a, WHITE)
    return f


def sc_final(t):
    f = shot("park", t, 100.3, DUR, 1.0, 1.1, (0.5, 0.42), (0.5, 0.4), "warm")
    f = gradient_v(f, 0, 900, 0.72, top_dark=True)
    f = gradient_v(f, 1100, H, 0.4)
    o = 101.55
    f = rise(f, "IT'S ALL ABOUT", "mont8", 64, W / 2, 400, t, 100.3, WHITE, out_t=o, tracking=6)
    if t >= 101.17:
        p = ease_out(prog(t, 101.17, 101.5))
        f += radial_glow(TEAL, W / 2, 560, 380, 0.12 * p * (1 - prog(t, o, o + 0.3)))
        f = text(f, "KITA.", "anton", 260, W / 2, 560, p * (1 - ease_in(prog(t, o, o + 0.3))), TEAL, lerp(1.3, 1, p))
    f = rise(f, "FOKUS KEPADA", "mont8", 46, W / 2, 225, t, 101.65, WHITE, tracking=8, out_t=105.45)
    f = rise(f, "EMOSI", "anton", 124, W / 2, 330, t, 103.14, WHITE, out_t=105.45)
    f = rise(f, "PEMIKIRAN", "anton", 124, W / 2, 450, t, 103.69, WHITE, out_t=105.45)
    f = rise(f, "TINGKAH LAKU", "anton", 124, W / 2, 570, t, 104.64, WHITE, out_t=105.45)
    if t >= 105.61:
        p = ease_out(prog(t, 105.61, 106.2))
        y = lerp(640, 430, p)
        f += radial_glow(AMBER, W / 2, y, 440, 0.15 * (1 - prog(t, 106.2, 107.4)))
        f = text(f, "ITU YANG", "anton", 150, W / 2, y - 120, clamp(p * 1.4), WHITE, lerp(0.7, 1.0, p))
        f = text(f, "PENTING.", "anton", 210, W / 2, y + 50, clamp(p * 1.4), AMBER, lerp(0.6, 1.0, p))
    return f


SCENES = [
    (0.00, 2.90, sc_hook_bed),
    (2.90, 5.50, sc_hook_mirror),
    (5.50, 6.55, sc_title),
    (6.55, 9.30, sc_radio),
    (9.30, 15.30, sc_kombo),
    (15.30, 19.30, sc_doc_intro),
    (19.30, 23.85, sc_core),
    (23.85, 29.15, sc_needs),
    (29.15, 35.25, sc_loss),
    (35.25, 43.95, sc_jealous),
    (43.95, 46.00, sc_collide),
    (46.00, 50.90, sc_overload),
    (50.90, 54.10, sc_host_kids),
    (54.10, 58.00, sc_doc_mic),
    (58.00, 62.25, sc_raised),
    (62.25, 67.95, sc_nilai),
    (67.95, 72.70, sc_worth),
    (72.70, 77.30, sc_loved),
    (77.30, 83.30, sc_left),
    (83.30, 89.60, sc_teach),
    (89.60, 93.80, sc_key),
    (93.80, 100.80, sc_split),
    (100.80, 1e9, sc_final),
]
TRANS = {
    2.90: ("fade", 0.35), 5.50: ("black", 0.24), 6.55: ("fade", 0.35), 9.30: ("whip", 0.25), 15.30: ("fade", 0.35),
    19.30: ("zoom", 0.3), 23.85: ("fade", 0.35), 29.15: ("whip", 0.25), 35.25: ("wipe", 0.45),
    43.95: ("flash", 0.18), 46.00: ("fade", 0.25), 50.90: ("fade", 0.45), 54.10: ("whip", 0.25),
    58.00: ("fade", 0.35), 62.25: ("whip", 0.25), 67.95: ("fade", 0.35), 72.70: ("wipe", 0.45),
    77.30: ("fade", 0.4), 83.30: ("fade", 0.45), 89.60: ("black", 0.45), 93.80: ("fade", 0.3),
    100.80: ("fade", 0.3),
}
_SC_BY_END = {b: fn for a, b, fn in SCENES}
_SC_BY_START = {a: fn for a, b, fn in SCENES}


def scene_at(t):
    for a, b, fn in SCENES:
        if a <= t < b:
            return fn
    return SCENES[-1][2]


def render_raw(t):
    for b, (kind, d) in TRANS.items():
        if b - d / 2 <= t < b + d / 2:
            p = (t - (b - d / 2)) / d
            A, B = _SC_BY_END[b](t), _SC_BY_START[b](t)
            if kind == "fade":
                return lerp(A, B, ease_io(p))
            if kind == "black":
                return A * (1 - ease_in(clamp(p * 2))) if p < 0.5 else B * ease_out(clamp(p * 2 - 1))
            if kind == "flash":
                return np.clip(lerp(A, B, ease_io(p)) + 0.55 * math.sin(math.pi * p), 0, 1)
            if kind == "whip":
                e = ease_io(p)
                A2 = np.roll(A, -int(e * W * 0.35), axis=1)
                B2 = np.roll(B, int((1 - e) * W * 0.35), axis=1)
                return motion_blur_h(lerp(A2, B2, clamp((p - 0.3) / 0.4)), 160 * math.sin(math.pi * p))
            if kind == "zoom":
                e = ease_io(p)
                return lerp(camera(A, 1 + 0.25 * e), camera(B, 1.18 - 0.18 * e), e)
            if kind == "wipe":
                x = int(-300 + p * (W + 600))
                xs = np.arange(W)[None, :, None].astype(np.float32)
                m = np.clip((x - xs) / 160 + 0.5, 0, 1)
                bar = np.exp(-((xs - x) / 140) ** 2)
                return (A * (1 - m) + B * m) * (1 - 0.92 * bar)
    return scene_at(t)(t)


# ------------------------------------------------------------- captions
KEYWORDS = {"rejection": CORAL, "rejection,": CORAL, "rejection.": CORAL, "rejection?": CORAL,
            "cemburu": AMBER, "kombo": AMBER, "emosi": TEAL, "emosi,": TEAL, "diterima,": TEAL,
            "disayangi,": TEAL, "selamat": TEAL, "selamat.": TEAL, "kehilangan": CORAL, "ditolak.": CORAL,
            "ditinggalkan,": CORAL, "dibandingkan": CORAL, "penting.": CORAL, "serentak,": AMBER,
            "sedih,": WHITE, "marah,": CORAL, "malu,": WHITE, "takut": CORAL, "dibesarkan": AMBER,
            "nilai": AMBER, "relationship": CORAL, "kawal": AMBER, "expect": CORAL, "selama-lamanya.": CORAL,
            "pemikiran,": TEAL, "tingkah": TEAL, "laku": TEAL, "kita.": WHITE, "kurang": CORAL}
BIG = {"rejection", "rejection,", "rejection.", "rejection?", "cemburu", "kombo", "kehilangan", "ditolak.",
       "nilai", "kawal", "relationship", "emosi,"}
CAP_HIDE = [(89.55, 93.85), (100.15, 1e9)]   # key line + final takeaway carry their own kinetic type
CAP_Y = 1395
CAP_SIZE = 64
CAP_MAXW = 900


def cap_words(chunk, idx_active):
    pieces = []
    for i, w in enumerate(chunk["words"]):
        key = w["w"].lower()
        col = KEYWORDS.get(key, WHITE)
        size = int(CAP_SIZE * (1.18 if key in BIG else 1.0))
        spr = text_sprite(w["w"], "mont8", size, col, 0, 0.75, 9)
        if i > idx_active:
            spr = spr.copy()
            spr[..., 3] = (spr[..., 3].astype(np.float32) * 0.55).astype(np.uint8)
        if i == idx_active:
            spr = cv2.resize(spr, None, fx=1.06, fy=1.06, interpolation=cv2.INTER_LINEAR)
        pieces.append(spr)
    lines, cur, cw = [], [], 0
    for s in pieces:
        wv = s.shape[1] - 36
        if cur and cw + wv > CAP_MAXW:
            lines.append(cur)
            cur, cw = [], 0
        cur.append(s)
        cw += wv + 16
    if cur:
        lines.append(cur)
    return lines


@functools.lru_cache(maxsize=2048)
def cap_sprite(ci, idx_active):
    rows = [hstack(l, 16 - 36) for l in cap_words(CAP["chunks"][ci], idx_active)]
    lh = int(CAP_SIZE * 1.2)
    w = max(r.shape[1] for r in rows)
    h = lh * (len(rows) - 1) + max(r.shape[0] for r in rows)
    out = np.zeros((h, w, 4), np.uint8)
    for i, r in enumerate(rows):
        x, y = (w - r.shape[1]) // 2, i * lh
        reg = out[y:y + r.shape[0], x:x + r.shape[1]]
        a = r[..., 3:4].astype(np.float32) / 255
        reg[..., :3] = (reg[..., :3] * (1 - a) + r[..., :3] * a).astype(np.uint8)
        reg[..., 3] = np.maximum(reg[..., 3], r[..., 3])
    return out


def captions(f, t):
    if any(a <= t < b for a, b in CAP_HIDE):
        return f
    for ci, ch in enumerate(CAP["chunks"]):
        if ch["t0"] - 0.05 <= t < ch["t1"]:
            idx = max([i for i, w in enumerate(ch["words"]) if t >= w["t0"] - 0.03] or [0])
            pin = ease_out(prog(t, ch["t0"] - 0.05, ch["t0"] + 0.1))
            pout = 1 - prog(t, ch["t1"] - 0.08, ch["t1"])
            return blit(f, cap_sprite(ci, idx), W / 2, CAP_Y + (1 - pin) * 10, min(pin, pout))
    return f


# ------------------------------------------------------------- main
def frame_at(fi):
    t = fi / FPS
    f = render_raw(t).astype(np.float32)
    f = speaker_tags(f, t)
    f = captions(f, t)
    f = finish(f, fi, 0.032, 0.85)
    return f * (1 - ease_in(prog(t, DUR - 0.9, DUR)))


def to8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    if sys.argv[1] == "--still":
        img = to8(frame_at(int(round(float(sys.argv[2]) * FPS))))
        cv2.imwrite(sys.argv[3], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sys.exit(0)
    a, b, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                          "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(to8(frame_at(fi)).tobytes())
        if (fi - a) % 60 == 0:
            print(f"[{a}-{b}] frame {fi}", flush=True)
    p.stdin.close()
    p.wait()
