"""REJECTION episode: voice clean-up + synthesized score + sound design + ducking -> mix_pre.wav

Everything except the voices is synthesized here (no third-party audio).
"""
import json
import subprocess

import numpy as np
import soundfile as sf

M = lambda t: t  # cue times are written directly on the final edit
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000
DUR = json.load(open("edit_map.json"))["duration"]
N = int(DUR * SR)
rng = np.random.default_rng(3)

# ---------------------------------------------------------------- 1. voice clean-up (ffmpeg chain)
subprocess.run([
    "ffmpeg", "-v", "error", "-y", "-i", "voice_edit_raw.wav", "-af",
    "highpass=f=75,lowpass=f=14500,"
    "equalizer=f=250:t=q:w=1.0:g=-2.5,equalizer=f=3200:t=q:w=1.2:g=2.5,equalizer=f=7500:t=q:w=1.5:g=-1.5,"
    "acompressor=threshold=-22dB:ratio=2.5:attack=12:release=180:knee=6:makeup=2,"
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
    t0, t1 = M(t0), M(t1)
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
    t0 = M(t0)
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
        place(music, s, M(t), gain)
        t += step


def drone(m, t0, t1, gain=0.05, f_lp=500):
    t0, t1 = M(t0), M(t1)
    n = t2i(t1 - t0)
    t = np.arange(n) / SR
    f = midi(m)
    s = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.003 * t) + 0.25 * np.sin(2 * np.pi * f * 3.01 * t)
    s = lp(s, f_lp) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * t))
    e = np.minimum(1, t / 1.5) * np.minimum(1, (t[-1] - t) / 1.5 + 1e-3)
    place(music, s * e, t0, gain)


A, C, D, E, F, G = 57, 60, 62, 64, 65, 67  # A3 etc.
CAPW = json.load(open("captions.json"))["words"]
WORDS = [(w["t0"], w["t1"]) for w in CAPW]

# HOOK: tense, dark (D minor colour) -> title
drone(26, 0.0, 6.6, 0.07, 280)          # D1
drone(33, 0.0, 6.6, 0.035, 380)         # A1
pad_chord([D, F, A, 64], 0.2, 5.4, 0.05, 900, att=1.8)
# radio / host question: light movement Dm - Bb - F - C, plucks @ 90 bpm
beat = 60 / 90
for notes, t0 in [([D, F, A], 6.5), ([58, D, F], 8.7), ([F, A, C + 12], 10.9), ([C, E, G], 13.1)]:
    pad_chord(notes, t0, t0 + 2.2, 0.042, 1400)
tt, k = 6.6, 0
while tt < 15.2:
    shape = [[74, 69, 65, 69], [70, 65, 62, 65], [72, 69, 65, 69], [72, 67, 64, 67]][min(3, int((tt - 6.5) / 2.2))]
    pluck(shape[k % 4], tt, 0.02, 0.9, pan=0.3 * np.sin(k))
    tt += beat / 2
    k += 1
# psychology: minimal pulse + dark pad
drone(26, 15.2, 43.9, 0.045, 320)
pad_chord([D, F, A], 15.4, 23.8, 0.038, 800, att=2.5)
pad_chord([F, A, C + 12], 23.8, 29.2, 0.04, 1500, att=1.0)          # needs: briefly warmer
pad_chord([A - 12, C, E], 29.2, 35.3, 0.04, 700, att=1.0)           # loss
pad_chord([58 - 12, D, F], 35.3, 43.9, 0.038, 700, att=1.2)         # jealousy
sub_pulse(15.6, 29.0, 70, 0.07)
sub_pulse(29.2, 43.8, 80, 0.075)
# overload: tension builds into "serentak", dark cluster under the four emotions
drone(25, 43.9, 51.0, 0.06, 260)
pad_chord([D, 63, A], 44.0, 51.0, 0.045, 600, att=0.6)
sub_pulse(44.0, 50.8, 104, 0.08)
# parenting: warmer, music-box plucks (F - C - Dm - Bb)
for notes, t0 in [([F, A, C + 12], 51.0), ([C, E, G], 56.4), ([D, F, A], 61.8), ([58, D, F], 67.2)]:
    pad_chord(notes, t0, t0 + 5.4, 0.036, 1300, att=1.5)
mb = [77, 72, 69, 72, 76, 72, 69, 74]
tt, k = 51.2, 0
while tt < 72.6:
    pluck(mb[k % len(mb)], tt, 0.015, 1.8, pan=-0.4 + 0.8 * (k % 2), bright=0.4)
    tt += 0.66
    k += 1
# loved (warm major) -> left (cold minor)
pad_chord([F, A, C + 12, E + 12], 72.7, 77.3, 0.045, 2200, att=0.8)
pad_chord([D, F, A], 77.3, 83.3, 0.04, 600, att=0.5)
drone(26, 77.3, 83.4, 0.05, 260)
# teaching: hopeful lift, then DIP before the key line (88.9 - 89.7)
pad_chord([58, D, F], 83.3, 86.2, 0.04, 1600, att=1.2)
pad_chord([C, E, G], 86.2, 88.9, 0.042, 1800, att=0.8, rel=0.6)
# key line swell, impact after "kawal"
pad_chord([F, A, C + 12, E + 12], 89.7, 93.8, 0.05, 1500, att=2.2)
drone(29, 92.98, 95.0, 0.06, 250)        # F1 under the impact
# split: unhealthy (left) tension under a hopeful top line
pad_chord([D, F, A], 93.8, 100.2, 0.038, 800, att=0.8)
sub_pulse(94.0, 99.9, 84, 0.06)
# resolution: Bb - C - F (major), swell to the end
pad_chord([58, D, F], 100.2, 102.8, 0.05, 2000, att=0.8)
pad_chord([C, E, G], 102.8, 105.5, 0.055, 2400, att=0.8)
pad_chord([F, A, C + 12, F + 12], 105.5, DUR + 0.5, 0.065, 2800, att=0.6, rel=2.5)
drone(29, 105.5, DUR, 0.05, 400)          # F1
tt, k = 100.3, 0
res = [77, 81, 84, 81, 76, 79, 84, 79, 77, 81, 84, 81]
while tt < DUR - 0.3:
    pluck(res[k % len(res)], tt, 0.022 * (0.6 if tt > 106.6 else 1), 1.6, pan=0.35 * np.sin(k * 1.1))
    tt += beat / 2 if tt < 105.5 else beat
    k += 1
music = reverb(music, 0.35)
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



SFX_LOG = []
_vr, _ = sf.read("voice_edit_raw.wav")
_h = SR // 200
VDB = 20 * np.log10(np.sqrt(np.convolve(_vr ** 2, np.ones(_h) / _h, "same")[::_h]) + 1e-9)   # 5 ms voice level


def P(buf, sig, t, gain=1.0, pan=0.0, head=0.12, search=0.6, label=""):
    """Place an SFX only where its loud head [t, t+head] does not overlap a spoken word.
    Searches up to `search` s later/earlier for the nearest clear gap; drops the SFX if none."""
    head = min(head, 0.06)   # only the attack must sit in a pause; tails decay under speech (ducked mix)

    def clear(x):
        i0, i1 = int(x * 200), int((x + head) * 200) + 1
        return VDB[max(0, i0 - 2):i1].max() < -30 if i1 < len(VDB) else True
    cands = sorted(np.arange(t - search, t + search + 1e-6, 0.01), key=lambda x: abs(x - t))
    for x in cands:
        if x >= 0 and clear(x):
            place(buf, sig, x, gain, pan)
            SFX_LOG.append((label, round(t, 2), round(float(x), 2)))
            return
    SFX_LOG.append((label, round(t, 2), None))


P(sfx, s_bass_hit(), 0.0, 0.55, label="hook bass hit")
P(sfx, s_impact(), 0.0, 0.25, label="hook impact")
P(sfx, s_whoosh(0.4, 400, 4000), 5.35, 0.18, -0.3, head=0.4, label="title whoosh")
P(sfx, s_tuning(), 6.0, 0.26, head=0.5, label="radio tuning")
P(sfx, s_whoosh(0.35, 500, 5000), 8.0, 0.12, 0.3, head=0.3, label="to question")
P(sfx, s_swell(0.8), 12.0, 0.12, head=0.15, label="kombo swell")
P(sfx, s_whoosh(0.4, 300, 3000), 15.2, 0.12, -0.4, head=0.3, label="to doctor")
for tt in (25.0, 26.12, 27.48):
    P(sfx, s_pop(1100), tt - 0.08, 0.08, head=0.06, search=0.25, label="needs pop")
P(sfx, s_whoosh(0.4, 400, 4000), 29.05, 0.14, 0.4, head=0.25, label="to loss")
P(sfx, s_bass_hit(0.8), 30.95, 0.16, head=0.1, search=0.5, label="rejection hit")
P(sfx, s_whoosh(0.45, 300, 3500), 35.1, 0.12, -0.4, head=0.25, label="wipe")
P(sfx, s_impact(), 45.0, 0.2, head=0.12, search=0.4, label="collide")
for k, tt in enumerate(np.arange(46.2, 50.8, 0.8)):
    P(sfx, s_heart(), tt, 0.16 + 0.1 * min(1, k / 3), head=0.08, search=0.3, label="heartbeat")
P(sfx, s_swell(1.0), 50.6, 0.15, head=0.15, label="to host")
P(sfx, s_tuning(), 50.75, 0.12, head=0.4, label="radio")
P(sfx, s_whoosh(0.35, 400, 4000), 53.9, 0.12, head=0.25, label="to doctor 2")
P(sfx, s_shimmer(), 66.9, 0.1, head=0.1, search=0.5, label="nilai diri")
P(sfx, s_riser(1.4), 69.7, 0.08, head=0.05, label="worth fill")
P(sfx, s_whoosh(0.5, 300, 3000), 72.55, 0.12, head=0.25, label="wipe 2")
for tt in (75.42, 76.28, 76.78):
    P(sfx, s_pop(1300), tt - 0.06, 0.08, head=0.06, search=0.25, label="tick")
P(sfx, s_whoosh(0.6, 2000, 300), 77.2, 0.1, head=0.2, label="to left")
P(sfx, s_bass_hit(1.0), 82.9, 0.14, head=0.1, search=0.4, label="aku? hit")
P(sfx, s_swell(0.8), 88.9, 0.15, head=0.15, search=0.4, label="pre-key swell")
P(sfx, s_impact(), 92.97, 0.3, head=0.12, search=0.9, label="KAWAL impact")
P(sfx, s_bass_hit(1.2), 92.97, 0.22, head=0.12, search=0.9, label="KAWAL bass")
P(sfx, s_whoosh(0.5, 300, 3500), 93.75, 0.12, head=0.25, label="split open")
P(sfx, s_whoosh(0.6, 2000, 300), 100.05, 0.1, head=0.15, label="healthy takes over")
P(sfx, s_shimmer(), 101.1, 0.12, head=0.08, search=0.4, label="KITA shimmer")
P(sfx, s_impact(), 105.55, 0.16, head=0.08, search=0.3, label="PENTING impact")
P(sfx, s_shimmer(), 106.55, 0.18, head=0.08, search=0.3, label="final shimmer")
sfx = reverb(sfx, 0.18)
for row in SFX_LOG:
    print("sfx", row)

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
i0, i1, i2 = t2i(88.9), t2i(89.3), t2i(89.9)   # dip right before "bukan semua benda kita boleh kawal"
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
