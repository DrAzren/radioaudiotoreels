"""Voice clean-up + synthesized score + sound design + ducking -> mix.wav

Everything except the voices is synthesized here (no third-party audio).
"""
import json
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000
DUR = json.load(open("edit_map.json"))["duration"]
N = int(DUR * SR)
rng = np.random.default_rng(3)

# ---------------------------------------------------------------- 1. voice clean-up (ffmpeg chain)
subprocess.run([
    "ffmpeg", "-v", "error", "-y", "-i", "voice_edit_raw.wav", "-af",
    "highpass=f=75,lowpass=f=14500,"
    "afftdn=nr=8:nf=-50:tn=1,"
    "equalizer=f=250:t=q:w=1.0:g=-2.5,equalizer=f=3200:t=q:w=1.2:g=2.5,equalizer=f=7500:t=q:w=1.5:g=-1.5,"
    "dynaudnorm=f=300:g=15:p=0.9:m=6:s=8,"
    "acompressor=threshold=-20dB:ratio=3:attack=6:release=120:makeup=2,"
    "alimiter=limit=0.89:level=false",
    "-ar", str(SR), "-ac", "1", "voice_clean.wav"], check=True)
voice, _ = sf.read("voice_clean.wav")
voice = np.pad(voice, (0, max(0, N - len(voice))))[:N]

# ---------------------------------------------------------------- helpers
def t2i(t):
    return int(t * SR)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "highpass", fs=SR, output="sos"), x)


def bp(x, f1, f2, order=2):
    return sosfilt(butter(order, [f1, f2], "bandpass", fs=SR, output="sos"), x)


def env_adsr(n, a, d, s, r):
    e = np.ones(n) * s
    ia, idd, ir = int(a * SR), int(d * SR), int(r * SR)
    ia = min(ia, n)
    e[:ia] = np.linspace(0, 1, ia)
    e[ia:ia + idd] = np.linspace(1, s, len(e[ia:ia + idd]))
    if ir > 0:
        e[-ir:] *= np.linspace(1, 0, min(ir, n))
    return e


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def ir_reverb(seconds=2.6, decay=3.2):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((2, n)) * np.exp(-decay * t)
    ir[:, :int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
    ir = np.stack([lp(ir[0], 6000), lp(ir[1], 6000)])
    return ir / np.sqrt((ir ** 2).sum(1, keepdims=True))


IR = ir_reverb()


def reverb(st, wet=0.3):
    out = np.stack([fftconvolve(st[c], IR[c])[:st.shape[1]] for c in range(2)])
    return st * (1 - wet) + out * wet


def place(buf, sig, t, gain=1.0, pan=0.0):
    """add mono/stereo sig to stereo buf at time t"""
    i = t2i(t)
    if sig.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        sig = np.stack([sig * l * 1.414, sig * r * 1.414])
    n = min(sig.shape[1], buf.shape[1] - i)
    if n > 0 and i >= 0:
        buf[:, i:i + n] += sig[:, :n] * gain


# ---------------------------------------------------------------- 2. score
music = np.zeros((2, N))


def pad_chord(notes, t0, t1, gain=0.06, bright=1200, att=1.2, rel=1.5, detune=0.12):
    n = t2i(t1 - t0 + rel)
    t = np.arange(n) / SR
    sig = np.zeros((2, n))
    for m in notes:
        for k, d in enumerate((-detune, 0, detune)):
            f = midi(m + d * 0.1)
            ph = rng.uniform(0, 2 * np.pi)
            # soft saw via few harmonics
            w = sum(np.sin(2 * np.pi * f * h * t + ph * h) / h ** 1.3 for h in range(1, 7))
            sig[k % 2] += w
    sig = np.stack([lp(sig[0], bright), lp(sig[1], bright)])
    e = env_adsr(n, att, 0.5, 0.9, rel)
    lfo = 1 + 0.08 * np.sin(2 * np.pi * 0.17 * t)
    place(music, sig * e * lfo / (len(notes) * 3), t0, gain)


def pluck(m, t0, gain=0.05, dec=1.4, pan=0.0, bright=1.0):
    n = t2i(dec * 2)
    t = np.arange(n) / SR
    f = midi(m)
    w = np.sin(2 * np.pi * f * t) + 0.35 * bright * np.sin(4 * np.pi * f * t) * np.exp(-t * 4) + \
        0.12 * bright * np.sin(6 * np.pi * f * t) * np.exp(-t * 7)
    e = np.exp(-t / dec * 3) * np.minimum(1, t / 0.004)
    place(music, w * e, t0, gain, pan)


def sub_pulse(t0, t1, bpm, gain=0.12, f=48):
    step = 60 / bpm
    t = t0
    while t < t1:
        n = t2i(0.5)
        tt = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * tt * (1 + 0.6 * np.exp(-tt * 30))) * np.exp(-tt * 9)
        place(music, s, t, gain)
        t += step


def drone(m, t0, t1, gain=0.05, f_lp=500):
    n = t2i(t1 - t0)
    t = np.arange(n) / SR
    f = midi(m)
    s = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.003 * t) + 0.25 * np.sin(2 * np.pi * f * 3.01 * t)
    s = lp(s, f_lp) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * t))
    e = np.minimum(1, t / 1.5) * np.minimum(1, (t[-1] - t) / 1.5 + 1e-3)
    place(music, s * e, t0, gain)


A, C, D, E, F, G = 57, 60, 62, 64, 65, 67  # A3 etc.

# HOOK + cold open: tense drone, A minor colour
drone(33, 0.0, 6.4, 0.07, 300)          # A1
drone(40, 0.0, 6.4, 0.035, 400)         # E2
pad_chord([A, C, E, 71], 0.3, 6.1, 0.05, 900, att=2.0)
# radio / host section: gentle movement Am - F - C - G with plucks @ 92bpm
prog_h = [([A, C, E], 6.4), ([F - 12 + 12, A, C + 12], 10.0), ([C, E, G], 13.6), ([G - 12 + 12, 59, D + 12], 17.2)]
for notes, t0 in prog_h:
    pad_chord(notes, t0, t0 + 3.6, 0.045, 1400)
beat = 60 / 92
arp = [A + 12, E + 12, C + 12, E + 12, 69 + 12, E + 12, C + 12, E + 12]
k = 0
tt = 7.3
while tt < 21.6:
    root = [A, F, C, G][min(3, int((tt - 6.4) / 3.6))]
    shape = {A: [69, 76, 72, 76], F: [65, 72, 69, 72], C: [67, 72, 76, 72], G: [67, 71, 74, 71]}[root]
    pluck(shape[k % 4], tt, 0.022, 0.9, pan=0.3 * np.sin(k))
    tt += beat / 2
    k += 1
# psychology: minimal atmospheric pulse (heartbeat-like) + dark pad
drone(38, 21.6, 44.2, 0.05, 350)        # D2
pad_chord([D, F, A], 21.8, 29.0, 0.04, 800, att=2.5)
pad_chord([A - 12, C, E], 29.0, 35.4, 0.04, 700, att=1.5)
pad_chord([D - 12, F, A], 35.4, 44.0, 0.035, 600, att=1.0)
sub_pulse(21.8, 35.4, 72, 0.08)
# regulate: lift slightly
pad_chord([F, A, C + 12], 39.8, 44.0, 0.03, 1600, att=2.0)
# flashback: nostalgic music-box plucks + warm pad
for notes, t0 in [([A, C, E], 44.2), ([F, A, C + 12], 48.0), ([D, F, A], 51.8), ([E, 68, 71], 54.6)]:
    pad_chord(notes, t0, t0 + 3.8, 0.035, 1100, att=1.5)
mb = [81, 76, 72, 76, 79, 76, 72, 74]
tt, k = 44.4, 0
while tt < 56.8:
    pluck(mb[k % len(mb)], tt, 0.016, 1.8, pan=-0.4 + 0.8 * (k % 2), bright=0.4)
    tt += 0.62
    k += 1
# adult / two things: neutral, slight unease
pad_chord([A, C, E, 71], 56.9, 63.0, 0.04, 1200, att=1.5)
pad_chord([D, F, A, 64], 63.0, 67.2, 0.04, 1000)
drone(33, 56.9, 75.8, 0.04, 300)
# clock: sparse
pad_chord([A, D + 12, E + 12], 67.0, 75.6, 0.03, 900, att=2)
# 75.9-76.9 dip (nothing)
# key line swell from 77.0, impact at 80.28
pad_chord([F, A, C + 12, E + 12], 77.0, 82.6, 0.05, 1500, att=2.5)
drone(29, 80.28, 84.0, 0.06, 250)       # F1 under impact
# healthy: warm F - C - Am - G
for notes, t0 in [([F, A, C + 12], 82.6), ([C, E, G], 85.0), ([A, C, E], 87.4), ([G - 12 + 12, 59, D + 12], 89.6)]:
    pad_chord(notes, t0, t0 + 2.6, 0.05, 2200, att=0.8)
tt, k = 82.8, 0
warm = [72, 77, 81, 77, 72, 76, 79, 76, 69, 72, 76, 72, 67, 71, 74, 71]
while tt < 91.3:
    pluck(warm[k % len(warm)], tt, 0.026, 1.2, pan=0.35 * np.sin(k * 1.3))
    tt += beat / 2
    k += 1
# avoidance: tension Dm - Bb, low pulse, drop at 99.6
pad_chord([D, F, A], 91.4, 95.6, 0.045, 700)
pad_chord([58, D, F], 95.6, 99.6, 0.045, 600)
drone(26, 91.4, 103.8, 0.06, 260)        # D1
sub_pulse(91.6, 99.4, 96, 0.07)
pad_chord([D - 12, A - 12, D], 99.6, 103.8, 0.04, 400, att=0.2)
# resolution: F - G - C (major), swell to the end
pad_chord([F, A, C + 12], 103.9, 106.6, 0.05, 2000, att=1.0)
pad_chord([G, 71, D + 12], 106.6, 109.3, 0.055, 2400, att=0.8)
pad_chord([C, E, G, C + 12], 109.3, DUR + 0.5, 0.065, 2800, att=0.6, rel=3.0)
drone(36, 109.3, DUR, 0.05, 400)          # C2
tt, k = 104.0, 0
res = [72, 77, 81, 77, 74, 79, 83, 79, 76, 79, 84, 79]
while tt < 113.5:
    pluck(res[k % len(res)], tt, 0.022 * (0.6 if tt > 111.6 else 1), 1.6, pan=0.35 * np.sin(k * 1.1))
    tt += beat / 2 if tt < 109.3 else beat
    k += 1
music = reverb(music, 0.35)
# fade out end
fe = np.ones(N)
fe[-t2i(1.4):] = np.linspace(1, 0, t2i(1.4)) ** 1.5
music *= fe

# ---------------------------------------------------------------- 3. sound design
sfx = np.zeros((2, N))


def s_bass_hit(dur=1.4):
    n = t2i(dur)
    t = np.arange(n) / SR
    f = 34 + 40 * np.exp(-t * 9)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6)
    s += 0.5 * lp(rng.standard_normal(n), 900) * np.exp(-t * 40)
    return np.tanh(s * 1.6) * 0.8


def s_whoosh(dur=0.5, f0=300, f1=3000, rev=False):
    n = t2i(dur)
    t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    blocks = 24
    for b in range(blocks):
        a, z = b * n // blocks, (b + 1) * n // blocks
        fc = f0 * (f1 / f0) ** (b / blocks)
        seg = bp(noise, max(60, fc * 0.6), min(18000, fc * 1.6))[a:z]
        out[a:z] = seg
    e = np.sin(np.pi * t) ** 2
    out = out * e
    return out[::-1] if rev else out


def s_ding():
    n = t2i(1.2)
    t = np.arange(n) / SR
    s = (np.sin(2 * np.pi * 1318.5 * t) * np.exp(-t * 6) +
         0.6 * np.sin(2 * np.pi * 1760 * t) * np.exp(-t * 5) * (t > 0.09))
    return s * 0.5


def s_tuning():
    n = t2i(0.75)
    t = np.arange(n) / SR
    st = bp(rng.standard_normal(n), 800, 5000) * 0.35
    chirp = np.sin(2 * np.pi * np.cumsum(900 + 700 * np.sin(2 * np.pi * 3 * t)) / SR) * 0.25
    e = np.minimum(1, t / 0.03) * np.exp(-t * 2.5)
    s = (st + chirp) * e
    click = np.zeros(n)
    click[:int(0.004 * SR)] = np.hanning(int(0.004 * SR)) * 0.9
    return s + click


def s_pop(f=1100):
    n = t2i(0.09)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t * (1 + 0.5 * np.exp(-t * 80))) * np.exp(-t * 60) * 0.5


def s_heart():
    n = t2i(0.6)
    t = np.arange(n) / SR
    thump = lambda d: np.sin(2 * np.pi * 52 * (t - d)) * np.exp(-np.clip(t - d, 0, None) * 22) * (t >= d)
    return lp(thump(0) + 0.7 * thump(0.22), 200)


def s_tick(f=3800):
    n = t2i(0.03)
    return bp(rng.standard_normal(n), f * 0.7, f * 1.3) * np.exp(-np.arange(n) / SR * 260) * 0.8


def s_riser(dur=1.6):
    n = t2i(dur)
    t = np.linspace(0, 1, n)
    nz = hp(rng.standard_normal(n), 2000) * t ** 2 * 0.4
    tone = np.sin(2 * np.pi * np.cumsum(200 + 600 * t ** 2) / SR) * t ** 2 * 0.3
    return nz + tone


def s_swell(dur=0.9):
    n = t2i(dur)
    t = np.linspace(0, 1, n)
    x = lp(rng.standard_normal(n), 2500) * np.exp(-(1 - t) * 6)
    x[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return x * 0.6


def s_impact():
    n = t2i(2.2)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * 58 * t) * np.exp(-t * 3) + 0.35 * lp(rng.standard_normal(n), 1500) * np.exp(-t * 12)
    return s * 0.7


def s_thud():
    n = t2i(0.25)
    t = np.arange(n) / SR
    return lp(rng.standard_normal(n), 400) * np.exp(-t * 30) * 0.9 + np.sin(2 * np.pi * 80 * t) * np.exp(-t * 25) * 0.5


def s_shimmer():
    n = t2i(2.0)
    t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f * t + i) * np.exp(-t * (2 + i)) for i, f in enumerate([1046.5, 1568, 2093]))
    return s * np.minimum(1, t / 0.02) * 0.25


def s_send():
    n = t2i(0.25)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * np.cumsum(700 + 1600 * t / 0.25) / SR) * np.exp(-t * 14) * 0.4


def s_door():
    n = t2i(0.8)
    t = np.arange(n) / SR
    knock = lp(rng.standard_normal(n), 300) * np.exp(-t * 18)
    latch = bp(rng.standard_normal(n), 1500, 4000) * np.exp(-np.clip(t - 0.05, 0, None) * 80) * (t > 0.05) * 0.3
    return (knock + latch) * 0.9


def s_hiss(dur):
    n = t2i(dur)
    x = bp(rng.standard_normal(n), 1500, 9000) * 0.05
    t = np.arange(n) / SR
    flutter = 1 + 0.3 * np.sin(2 * np.pi * 18 * t)
    e = np.minimum(1, t / 0.6) * np.minimum(1, (t[-1] - t) / 0.6)
    return x * flutter * e


P = place
P(sfx, s_bass_hit(), 0.02, 0.55)
P(sfx, s_impact(), 0.02, 0.25)
P(sfx, s_whoosh(0.45, 400, 4000), 0.42, 0.22, -0.3)
P(sfx, s_bass_hit(1.0), 1.2, 0.3)
P(sfx, s_whoosh(0.4, 300, 2500), 1.95, 0.12, 0.3)
P(sfx, s_whoosh(0.4, 500, 5000), 3.55, 0.16, -0.4)
P(sfx, s_ding(), 3.78, 0.2, 0.2)
P(sfx, s_riser(1.6), 4.6, 0.12)
P(sfx, s_tuning(), 6.12, 0.28)
P(sfx, s_whoosh(0.5, 200, 2000), 7.05, 0.14)
P(sfx, s_whoosh(0.55, 300, 3000), 14.45, 0.16, 0.5)
P(sfx, s_whoosh(0.4, 400, 4000), 16.4, 0.13, -0.5)
P(sfx, s_whoosh(0.45, 400, 4500), 18.35, 0.15, 0.5)
for tt in (22.0, 22.3, 22.6, 22.9):
    P(sfx, s_pop(900), tt, 0.12)
P(sfx, s_impact(), 25.18, 0.12)
for tt in (26.55, 27.35, 28.15):
    P(sfx, s_heart(), tt, 0.35)
for tt in (29.05, 29.85, 30.84, 31.75):
    P(sfx, s_pop(1000), tt, 0.12)
P(sfx, s_pop(700), 32.25, 0.12)
for k, tt in enumerate(np.arange(33.5, 39.4, 0.8)):
    P(sfx, s_heart(), tt, 0.18 + 0.12 * min(1, k / 3))
P(sfx, s_impact(), 35.5, 0.22)
P(sfx, s_swell(1.0), 38.65, 0.2)
P(sfx, s_door(), 43.75, 0.35)
P(sfx, s_hiss(13.0), 43.9, 0.6)
P(sfx, s_pop(1200), 49.9, 0.12)
P(sfx, s_bass_hit(0.8), 51.15, 0.18)
P(sfx, s_pop(800), 54.95, 0.12)
P(sfx, s_pop(1300), 56.1, 0.12)
P(sfx, s_swell(1.0), 56.5, 0.2)
P(sfx, s_impact(), 57.5, 0.14)
P(sfx, s_pop(1000), 59.2, 0.1)
P(sfx, s_pop(800), 63.0, 0.1)
P(sfx, s_whoosh(0.4, 400, 4000), 66.8, 0.14, 0.4)
for k, tt in enumerate(np.arange(67.1, 74.0, 0.5)):
    P(sfx, s_tick(3800 if k % 2 else 2900), tt, 0.16)
for tt in (70.33, 70.77, 71.41):
    P(sfx, s_pop(1200), tt, 0.14)
for tt in (72.7, 72.92, 73.14):
    P(sfx, s_whoosh(0.15, 2000, 8000), tt, 0.08)
P(sfx, s_swell(0.8), 76.15, 0.18)
P(sfx, s_impact(), 80.28, 0.32)
P(sfx, s_bass_hit(1.2), 80.28, 0.22)
P(sfx, s_whoosh(0.5, 300, 3500), 82.4, 0.13)
for tt in (85.3, 86.6, 87.9):
    P(sfx, s_whoosh(0.25, 800, 5000), tt - 0.1, 0.06)
P(sfx, s_pop(1300), 89.75, 0.1)
P(sfx, s_impact(), 91.4, 0.12)
P(sfx, s_send(), 94.0, 0.14)
P(sfx, s_ding(), 94.5, 0.0)
P(sfx, s_thud(), 94.95, 0.3)
for k, tt in enumerate(np.arange(96.45, 97.8, 0.17)):
    P(sfx, s_tick(3400), tt, 0.13)
P(sfx, s_pop(900), 98.6, 0.1)
P(sfx, s_whoosh(0.5, 300, 4000), 99.4, 0.16)
P(sfx, s_bass_hit(1.2), 99.62, 0.28)
P(sfx, s_impact(), 100.84, 0.18)
for tt in np.linspace(101.6, 102.65, 8):
    P(sfx, s_thud(), tt, 0.18, rng.uniform(-0.3, 0.3))
P(sfx, s_bass_hit(1.4), 102.7, 0.26)
P(sfx, s_swell(1.0), 103.0, 0.15)
P(sfx, s_whoosh(0.6, 2000, 300), 104.9, 0.1)
P(sfx, s_shimmer(), 105.55, 0.12)
P(sfx, s_impact(), 109.95, 0.2)
P(sfx, s_shimmer(), 109.95, 0.2)
for tt in (111.8, 112.3, 112.8):
    P(sfx, s_pop(900), tt, 0.08)
sfx = reverb(sfx, 0.18)

# ---------------------------------------------------------------- 4. ducking + mix
ve = np.abs(voice)
w = int(0.03 * SR)
ve = np.convolve(ve, np.ones(w) / w, "same")
act = (ve > 0.01).astype(float)
# attack 60ms, release 450ms smoothing
sm = np.zeros_like(act)
a_up, a_dn = 1 - np.exp(-1 / (0.06 * SR)), 1 - np.exp(-1 / (0.45 * SR))
prev = 0.0
for i in range(0, N, 48):  # control-rate 1 kHz
    x = act[i]
    c = a_up * 48 if x > prev else a_dn * 48
    prev = prev + (x - prev) * min(1, c)
    sm[i:i + 48] = prev
duck = 1 - 0.62 * sm
# music dip before key statement
dip = np.ones(N)
i0, i1, i2 = t2i(75.7), t2i(76.2), t2i(77.0)
dip[i0:i1] = np.linspace(1, 0.08, i1 - i0)
dip[i1:i2] = np.linspace(0.08, 1, i2 - i1) ** 3 * 0.92 + 0.08
MUSIC_GAIN, SFX_GAIN = 2.4, 0.8
mix = voice[None, :] * 1.0 + music * MUSIC_GAIN * duck * dip + sfx * SFX_GAIN * (1 - 0.3 * sm)
peak = np.abs(mix).max()
mix = mix / peak * 0.9
sf.write("mix_pre.wav", mix.T, SR, subtype="PCM_24")
sf.write("music_only.wav", (music * MUSIC_GAIN * duck * dip).T / peak, SR, subtype="PCM_16")
sf.write("sfx_only.wav", (sfx * SFX_GAIN).T / peak, SR, subtype="PCM_16")
print("mixed", mix.shape, "peak", peak)
