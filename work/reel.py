"""Scene timeline + frame renderer for the 'SPACE atau AVOIDANCE?' reel.

All times are OUTPUT seconds (see edit_map.json / captions.json).
Usage:  python3 reel.py <start_frame> <end_frame> <out.mp4>
        python3 reel.py --still <t> <out.png>
"""
import functools
import json
import math
import subprocess
import sys

import cv2
import numpy as np
import soundfile as sf

from gfx import *  # noqa

CAP = json.load(open(os.path.join(HERE, "captions.json")))
DUR = json.load(open(os.path.join(HERE, "edit_map.json")))["duration"]

# ------------------------------------------------------------- audio analysis (for waveforms)
_v, _sr = sf.read(os.path.join(HERE, "voice_edit_raw.wav"))
if _v.ndim > 1:
    _v = _v.mean(1)
_hop = _sr // FPS
_nfr = len(_v) // _hop + 2
ENV = np.zeros(_nfr, np.float32)
BANDS = np.zeros((_nfr, 28), np.float32)
_win = np.hanning(2048)
_edges = np.geomspace(90, 5000, 29)
_freqs = np.fft.rfftfreq(2048, 1 / _sr)
for i in range(_nfr):
    seg = _v[i * _hop: i * _hop + 2048]
    if len(seg) < 2048:
        seg = np.pad(seg, (0, 2048 - len(seg)))
    ENV[i] = np.sqrt(np.mean(seg[:_hop] ** 2) + 1e-12)
    sp = np.abs(np.fft.rfft(seg * _win))
    for b in range(28):
        m = (_freqs >= _edges[b]) & (_freqs < _edges[b + 1])
        BANDS[i, b] = sp[m].mean() if m.any() else 0
ENV = np.clip(ENV / (np.percentile(ENV, 98) + 1e-9), 0, 1.2)
BANDS = BANDS / (np.percentile(BANDS, 98, axis=0, keepdims=True) + 1e-9)
BANDS = np.clip(BANDS, 0, 1.3)
for i in range(1, _nfr):  # smooth decay
    BANDS[i] = np.maximum(BANDS[i], BANDS[i - 1] * 0.78)


def env_at(t):
    return float(ENV[min(len(ENV) - 1, max(0, int(t * FPS)))])


def bands_at(t):
    return BANDS[min(len(BANDS) - 1, max(0, int(t * FPS)))]


# ------------------------------------------------------------- cached graded sources
@functools.lru_cache(maxsize=None)
def G(name, look="neutral"):
    return grade(load(name), look)


def shot(name, t, t0, t1, z0, z1, c0, c1, look="neutral", rot0=0, rot1=0, ow=W, oh=H):
    p = ease_io(prog(t, t0, t1))
    return camera(G(name, look), lerp(z0, z1, p), lerp(c0[0], c1[0], p), lerp(c0[1], c1[1], p),
                  lerp(rot0, rot1, p), ow, oh)


def fit_card(name, look, w, h_box, t, t0, t1, z0=1.0, z1=1.06):
    """Fit-to-width image with blurred cover fill, for split panels (w x h_box)."""
    src = G(name, look)
    p = ease_io(prog(t, t0, t1))
    z = lerp(z0, z1, p)
    bg = camera(src, 1.0, 0.5, 0.5, 0, w, h_box)
    bg = darken(blur(cv2.resize(bg, (w // 4, h_box // 4)), 6), 0.45)
    bg = cv2.resize(bg, (w, h_box), interpolation=cv2.INTER_LINEAR)
    ch = int(w * 1920 / 1080)
    card = camera(src, z, 0.5, 0.5, 0, w, ch)
    y0 = (h_box - ch) // 2
    bg[y0:y0 + ch] = card
    # soft edge between card and fill
    for k in range(18):
        a = (18 - k) / 18 * 0.5
        bg[y0 + k] *= (1 - a)
        bg[y0 + ch - 1 - k] *= (1 - a)
    return bg


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
    sc = lerp(s0, 1.0, ease_out(p))
    return text(frame, s, fname, size, cx, cy, a, color, sc, tracking)


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
         border=None, anchor="c", dur=0.3):
    if t < t0:
        return frame
    p = ease_out(prog(t, t0, t0 + dur))
    a = p
    if out_t is not None:
        a *= 1 - ease_in(prog(t, out_t, out_t + 0.3))
    spr = pill_sprite(s, fname, size, fg, bg, bga, dot=dot, border=border)
    return blit(frame, spr, cx, cy + (1 - p) * 16, a, 1.0, anchor)


def gradient_v(frame, y0, y1, strength, top_dark=False):
    """darken a vertical band with a smooth ramp (for text legibility)."""
    y0, y1 = int(max(0, y0)), int(min(H, y1))
    n = y1 - y0
    if n <= 0:
        return frame
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    if top_dark:
        ramp = ramp[::-1]
    ramp = (ramp ** 1.3) * strength
    frame[y0:y1] *= (1 - ramp)[:, None, None]
    return frame


# ------------------------------------------------------------- reusable graphics
def onair(frame, t, t0, t1):
    a = window(t, t0, t1, 0.3, 0.3)
    if a <= 0:
        return frame
    spr = pill_sprite("ON AIR", "inter8", 32, WHITE, (0.75, 0.08, 0.08), 0.92, dot=(1, 1, 1))
    return blit(frame, spr, 60, 176, a, 1.0, "l")


SPEAKER_TAGS = [  # (t0, t1, label, dot)
    (0.75, 3.5, "DOKTOR", TEAL),
    (7.35, 12.3, "HOS RADIO", CORAL),
    (21.8, 25.2, "DOKTOR", TEAL),
    (67.0, 71.9, "HOS RADIO", CORAL),
    (72.2, 75.6, "DOKTOR", TEAL),
]
ONAIR = [(7.35, 12.3), (67.0, 71.9)]


def speaker_tags(frame, t):
    for t0, t1, lab, dot in SPEAKER_TAGS:
        a = window(t, t0, t1, 0.3, 0.35)
        if a > 0:
            onair_on = any(a0 <= t <= a1 for a0, a1 in ONAIR)
            x = 60 + (240 if onair_on else 0)
            spr = pill_sprite(lab, "inter7", 32, WHITE, INK, 0.62, dot=dot)
            frame = blit(frame, spr, x, 176, a, 1.0, "l")
    for a0, a1 in ONAIR:
        frame = onair(frame, t, a0, a1)
    return frame


def waveform_bars(frame, t, cx, cy, width, height, alpha, color=WHITE, n=28, synth=False):
    if alpha <= 0:
        return frame
    b = bands_at(t)
    if synth:
        b = np.array([0.35 + 0.3 * math.sin(t * 9 + i * 0.7) * math.sin(t * 3.1 + i * 0.3) for i in range(n)])
    ov = frame.copy()
    gap = width / n
    bw = max(3, int(gap * 0.5))
    for i in range(n):
        v = float(b[i % len(b)])
        # mirror for symmetry
        hh = max(4, v * height * (0.55 + 0.45 * math.sin(math.pi * (i + 0.5) / n)))
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
    pts = np.stack([xs, ys], 1).astype(np.int32)
    cv2.polylines(ov, [pts], False, tuple(float(c) for c in color), 4, cv2.LINE_AA)
    glow = blur(ov - frame, 8)
    frame = over(frame, ov, alpha)
    return np.clip(frame + glow * 0.8 * alpha, 0, 1)


def phone_ui(frame, t, cx, cy, sc, events):
    """Chat UI card. events: dict with times: sent, reply_none, blocked, clock (list of (t,label))."""
    w, h = int(760 * sc), int(1000 * sc)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    a_in = ease_out(prog(t, events["in"], events["in"] + 0.35))
    if a_in <= 0:
        return frame
    yoff = int((1 - a_in) * 60)
    card = rrect_sprite(w, h, int(56 * sc), (0.07, 0.08, 0.1), 0.94, border=(0.25, 0.27, 0.3), border_w=2,
                        shadow=24)
    frame = blit(frame, card, cx, cy + yoff, a_in)
    # header
    blocked = t >= events.get("blocked", 1e9)
    hy = y0 + int(110 * sc) + yoff
    av_col = (0.35, 0.37, 0.42) if blocked else (0.55, 0.42, 0.33)
    cv2.circle(frame, (x0 + int(90 * sc), hy), int(38 * sc), tuple(float(c) * a_in for c in av_col), -1, cv2.LINE_AA)
    frame = text(frame, "Dia", "inter7", int(40 * sc), x0 + int(150 * sc), hy - int(14 * sc), a_in, WHITE,
                 anchor="l", shadow=0)
    status = "tidak dapat dihubungi" if blocked else "dalam talian 2 jam lalu"
    frame = text(frame, status, "inter5", int(26 * sc), x0 + int(150 * sc), hy + int(26 * sc), a_in,
                 CORAL if blocked else DIM, anchor="l", shadow=0)
    cv2.line(frame, (x0 + 20, hy + int(70 * sc)), (x0 + w - 20, hy + int(70 * sc)),
             (0.2 * a_in, 0.21 * a_in, 0.24 * a_in), 2, cv2.LINE_AA)
    # sent bubble
    if t >= events["sent"]:
        p = ease_back(prog(t, events["sent"], events["sent"] + 0.3))
        msg = events.get("msg", "Kita bincang elok-elok, boleh?")
        spr = text_sprite(msg, "inter6", int(34 * sc), INK, shadow=0)
        bw, bh = spr.shape[1] + int(30 * sc), spr.shape[0] + int(4 * sc)
        bub = rrect_sprite(bw, bh, int(28 * sc), (0.85, 0.94, 0.86), 1.0)
        bx = x0 + w - int(40 * sc) - bw / 2
        by = y0 + int(330 * sc) + yoff
        frame = blit(frame, bub, bx, by, a_in, max(0.01, p))
        frame = blit(frame, spr, bx, by, a_in, max(0.01, p))
        # time + ticks
        tl = events.get("time", "11:47 PM")
        frame = text(frame, tl, "inter5", int(24 * sc), bx + bw / 2 - int(70 * sc), by + bh / 2 + int(30 * sc),
                     a_in * p, DIM, anchor="r", shadow=0)
        tx = int(bx + bw / 2 - int(52 * sc))
        ty = int(by + bh / 2 + int(32 * sc))
        col = (0.6, 0.62, 0.66)
        for k in (0, 1):
            ox = tx + k * int(14 * sc)
            cv2.line(frame, (ox, ty), (ox + int(8 * sc), ty + int(8 * sc)), col, 3, cv2.LINE_AA)
            cv2.line(frame, (ox + int(8 * sc), ty + int(8 * sc)), (ox + int(24 * sc), ty - int(10 * sc)), col, 3,
                     cv2.LINE_AA)
    # elapsed time stamps (no reply)
    for (tt, lab) in events.get("clock", []):
        if t >= tt:
            p = ease_out(prog(t, tt, tt + 0.25))
            frame = text(frame, lab, "mono4", int(26 * sc), cx, y0 + int(520 * sc) + yoff + (1 - p) * 12, a_in * p * 0.9,
                         DIM, shadow=0)
            break
    if t >= events.get("noreply", 1e9):
        p = ease_out(prog(t, events["noreply"], events["noreply"] + 0.35))
        frame = text(frame, "Tiada balasan", "inter6", int(30 * sc), cx, y0 + int(640 * sc) + yoff, a_in * p, CORAL,
                     shadow=0)
    if blocked:
        p = ease_back(prog(t, events["blocked"], events["blocked"] + 0.3))
        spr = pill_sprite("BLOCKED", "inter8", int(36 * sc), WHITE, CORAL, 0.95)
        frame = blit(frame, spr, cx, y0 + int(800 * sc) + yoff, a_in, max(0.01, p))
    return frame


def notif_ding(frame, t, t0, cx, cy):
    """small notification banner at top"""
    a = window(t, t0, t0 + 1.6, 0.18, 0.3)
    if a <= 0:
        return frame
    p = ease_out(prog(t, t0, t0 + 0.25))
    card = rrect_sprite(940, 160, 38, (0.12, 0.13, 0.16), 0.92, shadow=16)
    y = cy - (1 - p) * 160
    frame = blit(frame, card, cx, y, a)
    cv2.circle(frame, (int(cx - 380), int(y)), 40, (0.37 * a, 0.83 * a, 0.55 * a), -1, cv2.LINE_AA)
    frame = text(frame, "Mesej dihantar \u00b7 11:47 PM", "inter7", 40, cx - 315, y - 22, a, WHITE, anchor="l", shadow=0)
    frame = text(frame, "Kita bincang elok-elok, boleh?", "inter5", 36, cx - 315, y + 28, a, DIM, anchor="l", shadow=0)
    return frame


# ------------------------------------------------------------- scenes
def sc_hook_kitchen(t):
    f = shot("kitchen", t, 0, 0.55, 1.12, 1.32, (0.6, 0.36), (0.62, 0.34), "cool")
    f = desat(f, ease_out(prog(t, 0.25, 0.5)) * 0.65)
    f = darken(f, 0.15)
    f = slam(f, "SPACE?", "anton", 300, W / 2, 560, t, 0.05)
    return f


def sc_hook_split(t):
    base = G("sofa", "cool")
    z = lerp(1.08, 1.14, ease_io(prog(t, 0.55, 2.1)))
    img = camera(base, z, 0.5, 0.45)
    img = desat(img, 0.5 * (1 - prog(t, 0.55, 1.4)))
    d = int(ease_out(prog(t, 0.62, 1.35)) * 70)
    f = fill(INK)
    L, R = img[:, :W // 2], img[:, W // 2:]
    f[:, max(0, -d): W // 2 - d] = L[:, d:] if d > 0 else L
    f[:, W // 2 + d:] = R[:, :W // 2 - d] if d > 0 else R
    # thin cold seam light in the gap
    if d > 2:
        f[:, W // 2 - d:W // 2 + d] = np.array(INK) * 0.8
        g = np.exp(-((np.arange(2 * d) - d) / (d * 0.35)) ** 2)[None, :, None] * np.array(TEAL) * 0.18
        f[:, W // 2 - d:W // 2 + d] += g
    f = darken(f, 0.2)
    f = text(f, "SPACE?", "anton", 300, W / 2, 560, 1.0 - ease_in(prog(t, 1.85, 2.1)))
    f = rise(f, "ATAU", "mont8", 60, W / 2, 770, t, 1.05, out_t=1.85, out_d=0.25, tracking=8)
    f = slam(f, "AVOIDANCE?", "anton", 172, W / 2, 890, t, 1.2, CORAL, out_t=1.85, out_d=0.25)
    return f


def sc_cold_walkout(t):
    f = shot("walkout", t, 2.1, 3.7, 1.04, 1.16, (0.42, 0.45), (0.42, 0.42), "cool")
    return gradient_v(f, 1100, H, 0.6)


def sc_cold_phone(t):
    f = shot("bed", t, 3.7, 6.2, 1.1, 1.2, (0.5, 0.42), (0.52, 0.4), "cool")
    f = darken(blur(f, 2.5 * ease_out(prog(t, 3.9, 4.6))), 0.25)
    f = notif_ding(f, t, 3.78, W / 2, 360)
    return gradient_v(f, 1100, H, 0.6)


def sc_title(t):
    f = fill(INK)
    f += radial_glow(AMBER, W / 2, 960, 520, 0.10)
    f = waveform_line(f, t, 1000, window(t, 6.2, 7.6, 0.15, 0.2), AMBER, amp=70, synth_amt=1.4)
    f = rise(f, "BERI SPACE…", "anton", 150, W / 2, 720, t, 6.24, dy=30)
    f = rise(f, "ATAU LARI DARI", "anton", 112, W / 2, 1240, t, 6.5, AMBER, dy=30)
    f = rise(f, "KONFLIK?", "anton", 112, W / 2, 1360, t, 6.62, AMBER, dy=30)
    return f


def sc_radio(t):
    f = shot("mic", t, 7.3, 12.55, 1.02, 1.12, (0.42, 0.5), (0.45, 0.48), "neutral")
    f = darken(f, 0.1)
    a = window(t, 7.3, 12.55, 0.4, 0.3)
    f = waveform_bars(f, t, W / 2, 1150, 760, 150, 0.85 * a, WHITE)
    f = rise(f, "TEMU BUAL RADIO", "mont8", 34, W / 2, 1255, t, 7.6, DIM, tracking=10, out_t=12.2)
    return gradient_v(f, 1200, H, 0.55)


def sc_host_kitchen(t):
    f = shot("kitchen", t, 12.55, 14.7, 1.08, 1.2, (0.5, 0.4), (0.58, 0.38), "cool")
    return gradient_v(f, 1150, H, 0.55)


def sc_host_walkout(t):
    f = shot("walkout", t, 14.7, 16.55, 1.18, 1.04, (0.4, 0.45), (0.45, 0.5), "cool")
    return gradient_v(f, 1150, H, 0.55)


def sc_host_mamak(t):
    f = shot("mamak", t, 16.55, 18.55, 1.04, 1.15, (0.45, 0.5), (0.42, 0.45), "warm")
    f = rise(f, "KEDAI MAMAK", "mont8", 34, W / 2, 330, t, 16.75, WHITE, tracking=10, out_t=18.2, alpha=0.9)
    return gradient_v(f, 1150, H, 0.55)


def sc_host_jog(t):
    bob = 0.004 * math.sin(t * 9)
    f = shot("jogging", t, 18.55, 21.7, 1.1, 1.2, (0.5, 0.42 + bob), (0.5, 0.38 + bob), "neutral")
    f = rise(f, "JOGGING", "mont8", 34, W / 2, 330, t, 20.2, WHITE, tracking=10, out_t=21.3, alpha=0.9)
    return gradient_v(f, 1150, H, 0.55)


def bg_dark(name, look, t, t0, t1, sigma=16, dk=0.6, z=(1.1, 1.18)):
    f = shot(name, t, t0, t1, z[0], z[1], (0.5, 0.45), (0.5, 0.42), look)
    small = cv2.resize(f, (W // 4, H // 4))
    small = blur(small, sigma / 4)
    f = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return darken(f, dk)


def sc_response(t):
    f = bg_dark("sofa", "cool", t, 21.7, 26.25)
    f = rise(f, "CONFLICT RESPONSE", "mont8", 36, W / 2, 470, t, 21.85, DIM, tracking=12)
    words = [("FIGHT", 22.0), ("FLIGHT", 22.3), ("FREEZE", 22.6), ("WITHDRAW", 22.9)]
    hi = ease_out(prog(t, 25.15, 25.5))
    for i, (w_, t0) in enumerate(words):
        y = 610 + i * 150
        isk = w_ in ("FLIGHT", "WITHDRAW")
        col = AMBER if (isk and hi > 0.5) else WHITE
        alpha = 1.0 if isk else 1 - 0.7 * hi
        sc = 1 + (0.08 * hi if isk else 0)
        if t >= t0:
            p = ease_out(prog(t, t0, t0 + 0.35))
            f = text(f, w_, "anton", 140, W / 2, y + (1 - p) * 30, p * alpha, col, sc)
    if t >= 25.3:
        f = pill(f, "AVOID CONFLICT", W / 2, 1210, t, 25.3, WHITE, CORAL, 0.92, 34, "inter8")
    return f


def sc_confront(t):
    beat = sum(math.exp(-((t - b) / 0.06) ** 2) for b in (26.55, 27.35, 28.15)) + \
        0.6 * sum(math.exp(-((t - b - 0.22) / 0.06) ** 2) for b in (26.55, 27.35, 28.15))
    f = shot("kitchen", t, 26.25, 29.0, 1.85 + 0.015 * beat, 2.0 + 0.015 * beat, (0.66, 0.33), (0.66, 0.32), "cool")
    f = desat(f, 0.3)
    f = f * (1 + 0.05 * beat)
    f = rise(f, "CONFRONTATION", "mont8", 36, W / 2, 330, t, 28.35, CORAL, tracking=12, alpha=0.95)
    return gradient_v(f, 1150, H, 0.55)


def sc_flow(t):
    f = bg_dark("walkout", "cool", t, 29.0, 33.1, dk=0.66)
    items = [("CONFLICT", 29.05, CORAL), ("EMOTIONAL OVERLOAD", 29.85, WHITE), ("WITHDRAW", 30.84, WHITE),
             ("TEMPORARY RELIEF", 31.75, TEAL)]
    for i, (s, t0, col) in enumerate(items):
        y = 420 + i * 190
        f = rise(f, s, "anton", 96 if len(s) < 12 else 84, W / 2, y, t, t0, col, dy=26)
        if i and t >= t0 - 0.15:
            a = ease_out(prog(t, t0 - 0.15, t0 + 0.1))
            f = draw_arrow_down(f, W / 2, y - 135, y - 62, DIM, a, 4)
    if t >= 32.25:
        p = ease_out(prog(t, 32.25, 32.6))
        spr = hstack([text_sprite("Temporary relief", "mont8", 50, WHITE, shadow=0.6),
                      text_sprite("≠", "dejavu", 56, CORAL, shadow=0.6),
                      text_sprite("problem solved", "mont8", 50, WHITE, shadow=0.6)], -18)
        box = rrect_sprite(spr.shape[1] + 10, 104, 22, INK, 0.75)
        f = blit(f, box, W / 2, 1215, p)
        f = blit(f, spr, W / 2, 1215, p)
    return f


def sc_overwhelm(t):
    t0, t1 = 33.1, 39.6
    f = shot("overwhelm", t, t0, t1, 1.06, 1.2, (0.42, 0.3), (0.42, 0.28), "cool")
    # surroundings muffle: blur grows outside the face
    s = 2 + 9 * ease_io(prog(t, 33.4, 35.6))
    if s > 0.5:
        bl = blur(cv2.resize(f, (W // 2, H // 2)), s / 2)
        bl = cv2.resize(bl, (W, H))
        m = MASK_FACE
        f = f * m[..., None] + bl * (1 - m[..., None])
    shut = ease_out(prog(t, 35.5, 35.9))
    f = desat(f, 0.25 + 0.45 * shut)
    f = darken(f, 0.12 + 0.3 * shut)
    # floating 'voices'
    va = window(t, 33.2, 36.0, 0.5, 0.6) * (1 - 0.6 * shut)
    if va > 0:
        for k, (bx, by, ph) in enumerate([(170, 520, 0), (880, 430, 1.3), (900, 820, 2.1), (150, 900, 3.0),
                                          (820, 1150, 4.2), (230, 1220, 5.1)]):
            dx = 14 * math.sin(t * 1.3 + ph)
            dy = 10 * math.cos(t * 1.1 + ph)
            ap = va * (0.35 + 0.25 * math.sin(t * 2 + ph))
            spr = pill_sprite("• • •", "dejavu", 34, WHITE, (0.3, 0.32, 0.36), 0.7)
            f = blit(f, spr, bx + dx, by + dy, ap, 1.15)
    # heartbeat rings
    for b in np.arange(33.5, 39.5, 0.8):
        if b <= t < b + 0.8:
            p = (t - b) / 0.8
            ov = f.copy()
            cv2.circle(ov, (500, 1080), int(140 + 420 * p), (1, 1, 1), 3, cv2.LINE_AA)
            f = over(f, ov, 0.13 * (1 - p))
    # meter
    ma = window(t, 33.3, 37.7, 0.3, 0.4)
    if ma > 0:
        f = text(f, "EMOTIONAL LOAD", "mono7", 30, 80, 300, ma, WHITE, anchor="l", shadow=0.5)
        fillp = ease_io(prog(t, 33.5, 35.45)) * 5
        for i in range(5):
            x = 80 + i * 92
            lit = clamp(fillp - i)
            col = CORAL if i >= 3 else AMBER
            seg = rrect_sprite(80, 26, 6, (0.25, 0.26, 0.3), 0.9)
            f = blit(f, seg, x, 350, ma, 1.0, "tl")
            if lit > 0:
                seg2 = rrect_sprite(80, 26, 6, col, 1.0)
                f = blit(f, seg2, x, 350, ma * lit, 1.0, "tl")
    f = slam(f, "SHUT DOWN", "anton", 200, W / 2, 760, t, 35.5, WHITE, dur=0.3, out_t=37.6, s0=1.2, tracking=6)
    f = rise(f, "EMOTIONALLY", "mont9", 64, W / 2, 700, t, 37.95, TEAL, tracking=4)
    f = rise(f, "OVERWHELMED", "mont9", 64, W / 2, 790, t, 38.6, TEAL, tracking=4)
    return gradient_v(f, 1150, H, 0.55)


def _face_mask():
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (470, 700), (300, 420), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 90)


MASK_FACE = _face_mask()


def sc_regulate(t):
    f = shot("walk", t, 39.6, 43.9, 1.0, 1.1, (0.5, 0.48), (0.5, 0.44), "neutral")
    f = darken(f, 0.15)
    f = gradient_v(f, 0, 760, 0.65, top_dark=True)
    # arousal curve
    a = window(t, 39.8, 43.9, 0.3, 0.3)
    if a > 0:
        x0, x1, yb = 110, 970, 650
        ov = f.copy()
        cv2.line(ov, (x0, yb), (x1, yb), (0.5, 0.52, 0.56), 2, cv2.LINE_AA)
        xs = np.linspace(0, 1, 160)
        ys = np.where(xs < 0.25, 0.25 + 3.0 * xs, 1.0 - 0.0 * xs)
        ys = 0.2 + 0.75 * np.exp(-((xs - 0.25) / 0.16) ** 2) * (xs <= 0.25) + \
            (0.2 + 0.75 * np.exp(-((xs - 0.25) / 0.33) ** 2)) * (xs > 0.25)
        ys = ys * (1 - 0.15 * (xs > 0.7) * (xs - 0.7))
        pr = ease_io(prog(t, 40.0, 42.9))
        n = max(2, int(160 * pr))
        pts = np.stack([x0 + xs[:n] * (x1 - x0), yb - ys[:n] * 300], 1).astype(np.int32)
        # gradient color along curve: coral -> teal
        for k in range(n - 1):
            c = np.array(CORAL) * (1 - xs[k]) + np.array(TEAL) * xs[k]
            cv2.line(ov, tuple(pts[k]), tuple(pts[k + 1]), tuple(float(v) for v in c), 7, cv2.LINE_AA)
        f = over(f, ov, a)
        if n > 2:
            cv2.circle(f, tuple(pts[n - 1]), 11, (1, 1, 1), -1, cv2.LINE_AA)
        f = rise(f, "HIGH AROUSAL", "mono7", 34, x0 + 0.25 * (x1 - x0), yb - 345, t, 40.6, CORAL, alpha=a)
        f = rise(f, "SPACE", "mono7", 34, x0 + 0.55 * (x1 - x0), yb - 205, t, 41.4, WHITE, alpha=a)
        f = rise(f, "CALM", "mono7", 34, x0 + 0.92 * (x1 - x0), yb - 120, t, 42.5, TEAL, alpha=a)
    f = rise(f, "REGULATE EMOSI", "anton", 120, W / 2, 860, t, 42.05, WHITE, dy=30)
    return gradient_v(f, 1150, H, 0.55)


def flashback(f, t, amt=1.0):
    flick = 1 + 0.025 * math.sin(t * 37) * math.sin(t * 11)
    f = f * flick
    # warm light leak drifting on the right edge
    leak = radial_glow((1.0, 0.55, 0.25), W + 80 - 120 * math.sin(t * 0.7), 500 + 200 * math.sin(t * 0.5), 520, 0.22)
    f = f + leak * amt
    return np.clip(f, 0, 1)


def weave(t):
    return 0.0015 * math.sin(t * 23), 0.0012 * math.cos(t * 17)


def sc_child(t):
    dx, dy = weave(t)
    f = shot("child", t, 43.9, 49.4, 1.05, 1.16, (0.45 + dx, 0.48 + dy), (0.45 + dx, 0.45 + dy), "sepia")
    f = flashback(f, t)
    f = rise(f, "MASA KECIL", "mono7", 34, W / 2, 330, t, 47.7, AMBER, tracking=14)
    return gradient_v(f, 1150, H, 0.55)


def chips(f, t):
    f = pill(f, "SPEAK UP", W / 2 - 170, 420, t, 49.9, INK, WHITE, 0.95, 36, "inter8")
    f = pill(f, "GET SCOLDED", W / 2 + 170, 420, t, 51.15, WHITE, CORAL, 0.95, 36, "inter8")
    return f


def sc_child2(t):
    dx, dy = weave(t)
    f = shot("child", t, 49.4, 50.55, 1.55, 1.62, (0.45 + dx, 0.48 + dy), (0.45 + dx, 0.47 + dy), "sepia")
    f = flashback(f, t)
    f = chips(f, t)
    return gradient_v(f, 1150, H, 0.55)


def sc_parent(t):
    dx, dy = weave(t)
    f = shot("parent", t, 50.55, 51.8, 1.06, 1.14, (0.5 + dx, 0.42 + dy), (0.5 + dx, 0.4 + dy), "sepia")
    f = darken(flashback(f, t), 0.1)
    f = chips(f, t)
    return gradient_v(f, 1150, H, 0.55)


def sc_learned(t):
    f = bg_dark("child", "sepia", t, 51.8, 56.9, sigma=14, dk=0.55)
    f = flashback(f, t, 0.6)
    items = [("SPEAK UP", 51.8, WHITE), ("GET SCOLDED", 51.95, CORAL), ("STAY QUIET", 54.95, WHITE),
             ("FEELS SAFER", 56.1, AMBER)]
    for i, (s, t0, col) in enumerate(items):
        y = 400 + i * 200
        f = rise(f, s, "anton", 110, W / 2, y, t, t0, col, dy=24)
        if i and t >= t0 - 0.15:
            a = ease_out(prog(t, t0 - 0.15, t0 + 0.1))
            f = draw_arrow_down(f, W / 2, y - 145, y - 70, DIM, a, 4)
    return f


def sc_adult(t):
    f = shot("adult_pattern", t, 56.9, 59.1, 1.15, 1.24, (0.42, 0.36), (0.42, 0.35), "cool")
    f = darken(f, 0.15)
    f = gradient_v(f, 0, 900, 0.6, top_dark=True)
    f = rise(f, "Sometimes avoidance is", "mont7", 50, W / 2, 360, t, 57.05, WHITE)
    f = slam(f, "LEARNED.", "anton", 170, W / 2, 500, t, 57.5, AMBER, s0=1.25)
    return gradient_v(f, 1150, H, 0.55)


def card_row(f, t, t0, y, num, title, sub, img, col, dim=0.0):
    if t < t0:
        return f
    p = ease_out(prog(t, t0, t0 + 0.4))
    a = p * (1 - dim)
    x = 80 + (1 - p) * 60
    cw, ch = 920, 330
    card = rrect_sprite(cw, ch, 34, (0.1, 0.11, 0.14), 0.92, border=col, border_w=2, shadow=20)
    f = blit(f, card, x, y, a, 1.0, "tl")
    th = camera(G(img, "neutral"), 1.15, 0.45, 0.42, 0, 230, 290)
    f = paste_rgb(f, th, int(x + 20), int(y + 20), a, rrect_mask(230, 290, 24))
    f = text(f, num, "anton", 90, x + 300, y + 80, a, col, anchor="l", shadow=0.3)
    f = text(f, title, "anton", 70, x + 300, y + 175, a, WHITE, anchor="l", shadow=0.3)
    f = text(f, sub, "mont7", 38, x + 300, y + 255, a, DIM, anchor="l", shadow=0)
    return f


def sc_two(t):
    f = fill(INK)
    f += radial_glow(TEAL, 300, 600, 600, 0.06) + radial_glow(CORAL, 800, 1100, 600, 0.06)
    f = rise(f, "KENA BEZA DUA BENDA", "mont8", 40, W / 2, 340, t, 59.15, DIM, tracking=10)
    d1 = 0.45 * ease_out(prog(t, 63.0, 63.4))
    f = card_row(f, t, 59.2, 440, "1", "PERLUKAN MASA", "untuk regulate emosi", "breathing", TEAL, d1)
    f = card_row(f, t, 63.0, 820, "2", "MEMANG MENGELAK", "semua masalah", "car", CORAL)
    return f


def clock_face(f, t, cx, cy, r, a, spin_until, speed=1.0):
    if a <= 0:
        return f
    ov = f.copy()
    cv2.circle(ov, (cx, cy), r, (0.9, 0.9, 0.92), 5, cv2.LINE_AA)
    for k in range(12):
        ang = k * math.pi / 6
        r0 = r - (34 if k % 3 == 0 else 20)
        cv2.line(ov, (int(cx + r0 * math.sin(ang)), int(cy - r0 * math.cos(ang))),
                 (int(cx + (r - 8) * math.sin(ang)), int(cy - (r - 8) * math.cos(ang))),
                 (0.85, 0.85, 0.88), 5 if k % 3 == 0 else 3, cv2.LINE_AA)
    tt = min(t, spin_until) + 0.3 * max(0, t - spin_until) * math.exp(-max(0, t - spin_until))
    mang = tt * 2 * math.pi * 0.9 * speed
    hang = mang / 12 + 1.2
    cv2.line(ov, (cx, cy), (int(cx + 0.55 * r * math.sin(hang)), int(cy - 0.55 * r * math.cos(hang))),
             (1, 1, 1), 10, cv2.LINE_AA)
    cv2.line(ov, (cx, cy), (int(cx + 0.85 * r * math.sin(mang)), int(cy - 0.85 * r * math.cos(mang))),
             tuple(float(c) for c in AMBER), 6, cv2.LINE_AA)
    cv2.circle(ov, (cx, cy), 14, (1, 1, 1), -1, cv2.LINE_AA)
    return over(f, ov, a)


NUMS = [("30 MIN", 70.33, 200), ("2 JAM", 70.77, 540), ("1 HARI", 71.41, 880)]


def sc_clock(t):
    f = bg_dark("mic", "neutral", t, 66.95, 76.4, sigma=20, dk=0.72)
    fade = 1 - 0.7 * ease_io(prog(t, 74.2, 76.2))
    f = clock_face(f, t, W // 2, 690, 250, window(t, 66.95, 76.4, 0.35, 0.3) * fade, 74.0, 1.0)
    for i, (s, t0, x) in enumerate(NUMS):
        if t < t0:
            continue
        p = ease_back(prog(t, t0, t0 + 0.28))
        fall = ease_in(prog(t, 73.9 + i * 0.12, 74.7 + i * 0.12))
        a = clamp(p) * (1 - fall)
        y = 1100 + fall * 120
        f = text(f, s, "anton", 110, x, y, a, WHITE, max(0.2, p))
        st = 72.7 + i * 0.22
        if t >= st:
            sp = ease_out(prog(t, st, st + 0.18))
            sw = text_sprite(s, "anton", 110).shape[1] - 40
            ov = f.copy()
            cv2.line(ov, (int(x - sw / 2), int(y + 8)), (int(x - sw / 2 + sw * sp), int(y - 8)),
                     tuple(float(c) for c in CORAL), 10, cv2.LINE_AA)
            f = over(f, ov, a)
    f = rise(f, "TIADA ANGKA KHUSUS", "mont8", 40, W / 2, 1250, t, 73.0, CORAL, tracking=8, out_t=75.9)
    return f


def sc_key(t):
    f = fill(INK)
    pulse = 0.08 + 0.05 * ease_out(prog(t, 80.3, 80.5)) * (1 - prog(t, 80.5, 82))
    f += radial_glow(AMBER, W / 2, 1150, 650, pulse)
    shake = 6 * math.exp(-max(0, t - 80.3) * 14) * math.sin(t * 90) if t >= 80.3 else 0
    up = ease_io(prog(t, 78.7, 79.1))
    dim = 1 - 0.6 * up
    f = rise(f, "BUKAN SEKADAR", "mont8", 64, W / 2, 560 - up * 120, t, 76.95, WHITE, tracking=6, alpha=dim)
    if t >= 77.75:
        p = ease_out(prog(t, 77.75, 77.97))
        f = text(f, "BERAPA LAMA.", "anton", 170, W / 2, 700 - up * 140, p * dim, WHITE, lerp(1.15, 1.0, p))
    f = rise(f, "YANG PENTING:", "mont8", 56, W / 2, 880, t, 78.95, AMBER, tracking=6)
    f = slam(f, "APA JADI", "anton", 200, W / 2 + shake, 1060, t, 79.3, WHITE, s0=1.2)
    f = slam(f, "SELEPAS ITU?", "anton", 200, W / 2 + shake, 1250, t, 80.28, AMBER, s0=1.35)
    return f


# ---- split screen (healthy vs avoidance)
PW = 534


def panel_L(t):
    seq = [("kitchen", 82.6, 85.3, "MARAH", CORAL), ("walk", 85.3, 86.6, "AMBIL MASA", WHITE),
           ("breathing", 86.6, 87.9, "REGULATE", TEAL), ("reconnect", 87.9, 106.0, "KEMBALI BINCANG", AMBER)]
    for name, a, b, lab, col in seq:
        if a <= t < b:
            p = fit_card(name, "warm" if name in ("reconnect", "breathing") else "neutral", PW, H, t, a, b)
            if t - a < 0.25 and name != "kitchen":
                prev = seq[[s[0] for s in seq].index(name) - 1]
                q = fit_card(prev[0], "neutral", PW, H, t, prev[1], prev[2])
                p = lerp(q, p, ease_io((t - a) / 0.25))
            p = gradient_v(p, 1000, 1280, 0.7)
            p = rise(p, lab, "anton", 76, PW / 2, 1150, t, a + 0.08, col, dy=20)
            return p
    return fill(INK, PW, H)


def panel_R(t):
    if t < 91.4:
        p = fill((0.05, 0.055, 0.07), PW, H)
        return text(p, "?", "anton", 260, PW / 2, 760, 0.12)
    if t < 93.9:
        p = fit_card("walkout", "cool", PW, H, t, 91.4, 93.9)
        p = gradient_v(p, 1000, 1280, 0.7)
        return rise(p, "HILANG", "anton", 76, PW / 2, 1150, t, 91.5, CORAL, dy=20)
    if t < 96.44:
        p = fit_card("bed", "cool", PW, H, t, 93.9, 96.44)
        p = darken(blur(p, 3), 0.35)
        ev = dict(**{"in": 93.9}, sent=94.0, noreply=94.5, blocked=94.95,
                  clock=[(95.6, "1 jam kemudian…"), (94.4, "10 minit kemudian…")][::-1],
                  msg="Boleh kita bincang?")
        ev["clock"] = sorted(ev["clock"], key=lambda x: -x[0])
        return phone_ui(p, t, PW // 2, 760, 0.66, ev)
    if t < 97.8:
        k = (t - 96.44) / (97.8 - 96.44)
        day = int(min(2, k * 3))
        sky_n = np.array([0.04, 0.05, 0.12])
        sky_d = np.array([0.55, 0.45, 0.35])
        ph = (math.sin(k * 3 * math.pi * 2 - math.pi / 2) + 1) / 2
        top = sky_n * (1 - ph) + sky_d * ph
        p = np.empty((H, PW, 3), np.float32)
        p[:] = (top[None, None, :] * np.linspace(1, 0.35, H)[:, None, None]).astype(np.float32)
        p = clock_face(p, 70 + t * 4, PW // 2, 600, 170, 0.9, 1e9, 6.0)
        p = text(p, f"HARI {day + 1}", "anton", 140, PW / 2, 1000, 1.0, WHITE)
        return rise(p, "TIGA HARI KEMUDIAN", "mont8", 30, PW / 2, 1130, t, 96.5, DIM, tracking=6)
    p = fit_card("returns", "neutral", PW, H, t, 97.8, 100.0)
    p = gradient_v(p, 1000, 1280, 0.7)
    p = rise(p, "MUNCUL SEMULA", "anton", 70, PW / 2, 1120, t, 97.85, WHITE, dy=20)
    return rise(p, "macam tiada apa-apa", "mont7", 34, PW / 2, 1195, t, 98.25, DIM)


def compose_split(t, open_t=82.6, lr_alpha=(1, 1), close=0.0):
    f = fill(INK)
    po = ease_out(prog(t, open_t, open_t + 0.45))
    xl = int(-PW + po * PW) - int(close * PW * 0.0)
    xr = int(W - po * PW)
    L = panel_L(t)
    R = panel_R(t)
    actL = 1.0 if t < 91.4 else 0.45 + 0.55 * ease_out(prog(t, 103.9, 104.3))
    actR = 0.45 if t < 91.4 else 1.0
    f = paste_rgb(f, L * actL, xl, 0)
    f = paste_rgb(f, R * actR, xr, 0)
    # divider
    ov = f.copy()
    cv2.line(ov, (W // 2, 0), (W // 2, H), (1, 1, 1), 4, cv2.LINE_AA)
    f = over(f, ov, 0.5 * po)
    # headers
    f = gradient_v(f, 0, 420, 0.75, top_dark=True)
    la = 1.0 if (t < 91.4 or t >= 103.9) else 0.6
    f = text(f, "HEALTHY", "anton", 70, W / 4, 300, po * la, AMBER)
    f = text(f, "SPACE", "anton", 70, W / 4, 372, po * la, AMBER)
    f = text(f, "AVOIDANCE /", "anton", 70, 3 * W / 4, 300, po * (0.35 if t < 91.4 else 1.0), CORAL)
    f = text(f, "STONEWALLING", "anton", 70, 3 * W / 4, 372, po * (0.35 if t < 91.4 else 1.0), CORAL)
    # formulas
    if t >= 89.75:
        p = ease_out(prog(t, 89.75, 90.2))
        spr = flow_sprite(("SPACE", "REGULATE", "RETURN"), "mono7", 26, (WHITE, WHITE, AMBER))
        f = blit(f, spr, W / 4, 1268, p * (1 if t < 91.4 else 0.7), min(1.0, (PW - 20) / spr.shape[1]))
    if t >= 98.6:
        p = ease_out(prog(t, 98.6, 99.0))
        spr = flow_sprite(("CONFLICT", "DISAPPEAR", "NO RESOLUTION"), "mono7", 26, (WHITE, WHITE, CORAL))
        f = blit(f, spr, 3 * W / 4, 1268, p, min(1.0, (PW - 20) / spr.shape[1]))
    return f


def sc_split(t):
    return compose_split(t)


def brick_wall(f, t, t0, t1, cx, width, alpha=1.0):
    rows = 48
    bh = H // rows
    built = ease_io(prog(t, t0, t1)) * rows
    ov = f.copy()
    bw = width // 3
    for r in range(rows):
        k = clamp(built - r)
        if k <= 0:
            break
        y = H - (r + 1) * bh
        y_drop = int((1 - ease_out(k)) * -60)
        off = (bw // 2) if r % 2 else 0
        x = cx - width // 2 - off
        while x < cx + width // 2:
            xa, xb = max(cx - width // 2, x + 3), min(cx + width // 2, x + bw - 3)
            if xb > xa:
                shade = 0.32 + 0.05 * ((r * 7 + x // bw) % 3)
                col = (shade * 0.95, shade * 0.97, shade * 1.05)
                cv2.rectangle(ov, (int(xa), int(y + 3 + y_drop)), (int(xb), int(y + bh - 3 + y_drop)), col, -1,
                              cv2.LINE_AA)
            x += bw
    return over(f, ov, alpha)


def sc_avoid_full(t):
    f = shot("returns", t, 99.6, 103.9, 1.0, 1.07, (0.4, 0.42), (0.42, 0.4), "neutral")
    f = darken(desat(f, 0.35), 0.45)
    f = brick_wall(f, t, 101.55, 102.75, W // 2 + 30, 300)
    o1 = 100.75
    f = rise(f, "INI BUKAN", "mont8", 70, W / 2, 560, t, 99.7, WHITE, out_t=o1, tracking=6)
    f = slam(f, "‘SPACE’.", "anton", 200, W / 2, 720, t, 99.95, WHITE, out_t=o1, s0=1.2)
    f = rise(f, "INI", "mont8", 70, W / 2, 560, t, o1 + 0.1, WHITE, out_t=101.5, tracking=6)
    f = slam(f, "AVOIDANCE.", "anton", 190, W / 2, 720, t, 100.84, CORAL, out_t=101.5, s0=1.25)
    f = slam(f, "STONEWALLING", "anton", 150, W / 2, 760, t, 102.7, WHITE, s0=1.3, tracking=4)
    return gradient_v(f, 1150, H, 0.5)


def sc_recap(t):
    f = compose_split(t, open_t=103.5)
    close = ease_io(prog(t, 104.9, 105.5))
    if close > 0:
        fin = shot("final", t, 104.9, 111.6, 1.0, 1.1, (0.5, 0.42), (0.5, 0.4), "warm")
        f = lerp(f, fin, close)
    return f


def sc_final(t):
    f = shot("final", t, 104.9, 111.6, 1.0, 1.1, (0.5, 0.42), (0.5, 0.4), "warm")
    f = gradient_v(f, 0, 1000, 0.62, top_dark=True)
    f = gradient_v(f, 1100, H, 0.4)
    o = 107.15
    f = rise(f, "SPACE YANG SIHAT", "anton", 118, W / 2, 470, t, 105.55, WHITE, out_t=o)
    f = rise(f, "BUKAN SEKADAR MENJAUH.", "mont8", 50, W / 2, 580, t, 106.1, DIM, out_t=o, tracking=2)
    f = rise(f, "ADA TEMPOH.", "anton", 110, W / 2, 420, t, 107.6, WHITE)
    f = rise(f, "ADA JALAN", "anton", 110, W / 2, 545, t, 108.95, WHITE)
    f = rise(f, "UNTUK", "mont8", 54, W / 2, 650, t, 109.55, DIM, tracking=8)
    if t >= 109.95:
        p = ease_out(prog(t, 109.95, 110.6))
        y = lerp(1150, 830, p)
        s = lerp(0.55, 1.0, p)
        glow = 0.16 * (1 - prog(t, 110.4, 111.4))
        f += radial_glow(AMBER, W / 2, y, 420, glow)
        f = text(f, "KEMBALI.", "anton", 230, W / 2, y, clamp(p * 1.4), AMBER, s)
    f = rise(f, "KEPADA CONVERSATION", "mont8", 40, W / 2, 990, t, 110.85, WHITE, tracking=8)
    return f


def sc_end(t):
    f = shot("final", t, 104.9, 116, 1.1, 1.16, (0.5, 0.4), (0.5, 0.39), "warm")
    f = darken(blur(cv2.resize(f, (W // 4, H // 4)), 5), 0.62)
    f = cv2.resize(f, (W, H))
    f += radial_glow(AMBER, W / 2, 860, 600, 0.07)
    f = rise(f, "REGULATE.", "anton", 150, W / 2, 680, t, 111.8, WHITE)
    f = rise(f, "RETURN.", "anton", 150, W / 2, 840, t, 112.3, AMBER)
    f = rise(f, "COMMUNICATE.", "anton", 150, W / 2, 1000, t, 112.8, WHITE)
    f = pill(f, "Save untuk rujukan", W / 2, 1210, t, 113.7, WHITE, INK, 0.55, 34, "inter6",
             border=(0.5, 0.5, 0.55))
    f = rise(f, "Audio: temu bual radio Kool FM  ·  Visual ilustrasi dijana AI", "inter5", 24, W / 2, 1490, t,
             114.0, DIM)
    return f * (1 - ease_in(prog(t, 115.45, DUR)))


SCENES = [
    (0.00, 0.55, sc_hook_kitchen),
    (0.55, 2.10, sc_hook_split),
    (2.10, 3.70, sc_cold_walkout),
    (3.70, 6.20, sc_cold_phone),
    (6.20, 7.30, sc_title),
    (7.30, 12.55, sc_radio),
    (12.55, 14.70, sc_host_kitchen),
    (14.70, 16.55, sc_host_walkout),
    (16.55, 18.55, sc_host_mamak),
    (18.55, 21.70, sc_host_jog),
    (21.70, 26.25, sc_response),
    (26.25, 29.00, sc_confront),
    (29.00, 33.10, sc_flow),
    (33.10, 39.60, sc_overwhelm),
    (39.60, 43.90, sc_regulate),
    (43.90, 49.40, sc_child),
    (49.40, 50.55, sc_child2),
    (50.55, 51.80, sc_parent),
    (51.80, 56.90, sc_learned),
    (56.90, 59.10, sc_adult),
    (59.10, 66.95, sc_two),
    (66.95, 76.40, sc_clock),
    (76.40, 82.60, sc_key),
    (82.60, 99.60, sc_split),
    (99.60, 103.90, sc_avoid_full),
    (103.90, 105.50, sc_recap),
    (105.50, 111.60, sc_final),
    (111.60, DUR + 1, sc_end),
]
TRANS = {
    0.55: ("whip", 0.2), 2.10: ("fade", 0.3), 3.70: ("whip", 0.25), 6.20: ("black", 0.24), 7.30: ("fade", 0.4),
    12.55: ("fade", 0.3), 14.70: ("wipe", 0.45), 16.55: ("whip", 0.25), 18.55: ("whip", 0.3), 21.70: ("fade", 0.35),
    26.25: ("zoom", 0.3), 29.00: ("fade", 0.3), 33.10: ("fade", 0.4), 39.60: ("fade", 0.45), 43.90: ("black", 0.6),
    49.40: ("fade", 0.2), 50.55: ("flash", 0.18), 51.80: ("fade", 0.35), 56.90: ("fade", 0.7), 59.10: ("fade", 0.35),
    66.95: ("whip", 0.25), 76.40: ("black", 0.45), 82.60: ("fade", 0.3), 99.60: ("whip", 0.3), 103.90: ("fade", 0.3),
    111.60: ("fade", 0.7),
}


def scene_at(t):
    for a, b, fn in SCENES:
        if a <= t < b:
            return fn
    return SCENES[-1][2]


def render_raw(t):
    fn = scene_at(t)
    for b, (kind, d) in TRANS.items():
        if b - d / 2 <= t < b + d / 2:
            p = (t - (b - d / 2)) / d
            prev = [s for s in SCENES if s[1] == b][0][2]
            nxt = [s for s in SCENES if s[0] == b][0][2]
            A, B = prev(t), nxt(t)
            if kind == "fade":
                return lerp(A, B, ease_io(p))
            if kind == "black":
                return A * (1 - ease_in(clamp(p * 2))) if p < 0.5 else B * ease_out(clamp(p * 2 - 1))
            if kind == "flash":
                m = lerp(A, B, ease_io(p))
                return np.clip(m + 0.55 * math.sin(math.pi * p), 0, 1)
            if kind == "whip":
                e = ease_io(p)
                shift = int(e * W * 0.35)
                A2 = np.roll(A, -shift, axis=1)
                B2 = np.roll(B, int((1 - e) * W * 0.35), axis=1)
                m = lerp(A2, B2, clamp((p - 0.3) / 0.4))
                return motion_blur_h(m, 160 * math.sin(math.pi * p))
            if kind == "zoom":
                e = ease_io(p)
                A2 = camera(A, 1 + 0.25 * e)
                B2 = camera(B, 1.18 - 0.18 * e)
                return lerp(A2, B2, e)
            if kind == "wipe":
                x = int(-300 + p * (W + 600))
                xs = np.arange(W)[None, :, None].astype(np.float32)
                m = np.clip((x - xs) / 160 + 0.5, 0, 1)
                bar = np.exp(-((xs - x) / 140) ** 2)
                out = A * (1 - m) + B * m
                return out * (1 - 0.92 * bar)
    return fn(t)


# ------------------------------------------------------------- captions
KEYWORDS = {"space": AMBER, "space.": AMBER, "ruang": AMBER, "konflik": CORAL, "konflik.": CORAL,
            "konflik?": CORAL, "avoid": CORAL, "confrontation.": CORAL, "shutdown": CORAL, "emosi": TEAL,
            "emosi.": TEAL, "regulate": TEAL, "overwhelmed": TEAL, "avoidance": CORAL, "stonewall.": CORAL,
            "kembali": AMBER, "healthy": AMBER, "diam.": WHITE, "selamat": WHITE, "conversation": AMBER,
            "hilang.": CORAL, "block": CORAL, "mamak": WHITE, "jogging": WHITE, "dibesarkan.": AMBER,
            "masalah.": CORAL, "lama": AMBER, "selepas": AMBER, "sihat": AMBER, "tempoh": AMBER, "jalan": AMBER}
BIG = {"space", "space.", "konflik", "konflik.", "konflik?", "shutdown", "emosi", "emosi.", "avoidance", "stonewall.",
       "kembali", "confrontation."}
CAP_HIDE = [(76.4, 82.6), (105.4, 200)]
CAP_Y = 1395
CAP_SIZE = 64
CAP_MAXW = 900


def cap_words(chunk, idx_active):
    """Return layout: list of lines, each list of (sprite) for state with active word index."""
    pieces = []
    for i, w in enumerate(chunk["words"]):
        key = w["w"].lower()
        base_col = KEYWORDS.get(key, WHITE)
        is_big = key in BIG
        col = base_col if i <= idx_active else (WHITE if base_col == WHITE else base_col)
        alpha_dim = i > idx_active
        size = int(CAP_SIZE * (1.18 if is_big else 1.0))
        spr = text_sprite(w["w"], "mont8", size, col, 0, 0.75, 9)
        if alpha_dim:
            spr = spr.copy()
            spr[..., 3] = (spr[..., 3].astype(np.float32) * 0.55).astype(np.uint8)
        if i == idx_active:
            spr = cv2.resize(spr, None, fx=1.06, fy=1.06, interpolation=cv2.INTER_LINEAR)
        pieces.append(spr)
    # wrap into <= 2 lines
    space = 16
    lines, cur, cw = [], [], 0
    for s in pieces:
        wv = s.shape[1] - 36
        if cur and cw + wv > CAP_MAXW:
            lines.append(cur)
            cur, cw = [], 0
        cur.append(s)
        cw += wv + space
    if cur:
        lines.append(cur)
    return lines


@functools.lru_cache(maxsize=2048)
def cap_sprite(ci, idx_active):
    ch = CAP["chunks"][ci]
    lines = cap_words(ch, idx_active)
    rows = [hstack(l, 16 - 36) for l in lines]
    lh = int(CAP_SIZE * 1.2)
    w = max(r.shape[1] for r in rows)
    h = lh * (len(rows) - 1) + max(r.shape[0] for r in rows)
    out = np.zeros((h, w, 4), np.uint8)
    for i, r in enumerate(rows):
        x = (w - r.shape[1]) // 2
        y = i * lh
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
            idx = -1
            for i, w in enumerate(ch["words"]):
                if t >= w["t0"] - 0.03:
                    idx = i
            idx = max(idx, 0)
            pin = ease_out(prog(t, ch["t0"] - 0.05, ch["t0"] + 0.1))
            pout = 1 - prog(t, ch["t1"] - 0.08, ch["t1"])
            spr = cap_sprite(ci, idx)
            return blit(f, spr, W / 2, CAP_Y + (1 - pin) * 10, min(pin, pout))
    return f


# ------------------------------------------------------------- main
def frame_at(fi):
    t = fi / FPS
    f = render_raw(t).astype(np.float32)
    f = speaker_tags(f, t)
    f = captions(f, t)
    vig = 0.85
    grain = 0.05 if 43.9 <= t < 56.9 else 0.032
    return finish(f, fi, grain, vig)


def to8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    if sys.argv[1] == "--still":
        t = float(sys.argv[2])
        img = to8(frame_at(int(round(t * FPS))))
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
