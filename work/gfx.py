"""Low-level drawing helpers for the reel renderer (numpy/OpenCV/PIL).

Frames are float32 RGB arrays in [0,1], shape (H, W, 3).
"""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# palette (sRGB 0-1)
def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

WHITE = hexc("F5F3EE")
DIM = hexc("A9AEB6")
AMBER = hexc("F4B860")
CORAL = hexc("FF5E57")
TEAL = hexc("5FD3C6")
INK = hexc("0B0D12")

FONTS = {
    "anton": "anton-400.ttf",
    "bebas": "bebas-neue-400.ttf",
    "mont9": "montserrat-900.ttf",
    "mont8": "montserrat-800.ttf",
    "mont7": "montserrat-700.ttf",
    "mont5": "montserrat-500.ttf",
    "inter4": "inter-400.ttf",
    "inter5": "inter-500.ttf",
    "inter6": "inter-600.ttf",
    "inter7": "inter-700.ttf",
    "inter8": "inter-800.ttf",
    "mono4": "jetbrains-mono-400.ttf",
    "mono7": "jetbrains-mono-700.ttf",
    "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


@functools.lru_cache(maxsize=None)
def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, FONTS[name]), size)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, a, b):
    if b <= a:
        return 1.0 if t >= a else 0.0
    return clamp((t - a) / (b - a))


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_in(x):
    return x ** 3


def ease_io(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_back(x, s=1.70158):
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def lerp(a, b, x):
    return a + (b - a) * x


def window(t, a, b, fi=0.25, fo=0.25):
    """1 inside [a,b] with fade in/out ramps."""
    if t < a or t > b:
        return 0.0
    return min(ease_out(prog(t, a, a + fi)) if fi > 0 else 1.0,
               1 - ease_in(prog(t, b - fo, b)) if fo > 0 else 1.0)


# ---------------------------------------------------------------- images
_IMG = {}


def load(name):
    if name not in _IMG:
        im = cv2.imread(os.path.join(HERE, "broll", name + ".png"), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)
        _IMG[name] = im
    return _IMG[name]


def camera(img, zoom=1.0, cx=0.5, cy=0.5, rot=0.0, out_w=W, out_h=H):
    """Crop/zoom into img. (cx,cy) = focal point in normalized image coords
    placed at the centre of the output. zoom>=1 relative to cover-fit."""
    ih, iw = img.shape[:2]
    base = max(out_w / iw, out_h / ih)
    s = base * zoom
    # keep focal point inside so we never reveal borders
    half_w, half_h = out_w / (2 * s), out_h / (2 * s)
    fx = clamp(cx * iw, half_w, iw - half_w)
    fy = clamp(cy * ih, half_h, ih - half_h)
    M = cv2.getRotationMatrix2D((fx, fy), rot, s)
    M[0, 2] += out_w / 2 - fx
    M[1, 2] += out_h / 2 - fy
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def kb(name, t, t0, t1, z0, z1, c0, c1, rot0=0.0, rot1=0.0, out_w=W, out_h=H, ease=ease_io):
    p = ease(prog(t, t0, t1))
    return camera(load(name), lerp(z0, z1, p), lerp(c0[0], c1[0], p), lerp(c0[1], c1[1], p),
                  lerp(rot0, rot1, p), out_w, out_h)


def blur(img, sigma):
    if sigma < 0.3:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def motion_blur_h(img, length):
    length = int(abs(length))
    if length < 2:
        return img
    k = np.zeros((1, length), np.float32)
    k[0, :] = 1 / length
    return cv2.filter2D(img, -1, k)


def desat(img, amt):
    g = img @ np.array([0.299, 0.587, 0.114], np.float32)
    return img * (1 - amt) + g[..., None] * amt


def grade(img, look="neutral", amount=1.0):
    """Cinematic grade: soft contrast + split-tone."""
    x = img
    if look == "sepia":
        g = x @ np.array([0.33, 0.5, 0.17], np.float32)
        tone = np.stack([g * 1.08 + 0.04, g * 0.95 + 0.02, g * 0.72], -1)
        x = x * 0.25 + tone * 0.75
        x = 0.06 + x * 0.88  # lifted blacks, faded
    elif look == "warm":
        x = x * np.array([1.06, 1.0, 0.9], np.float32)
    elif look == "cool":
        x = x * np.array([0.92, 0.98, 1.08], np.float32)
    elif look == "cold_desat":
        x = desat(x, 0.45) * np.array([0.9, 0.97, 1.08], np.float32)
    # gentle S-curve
    x = np.clip(x, 0, 1)
    x = x + 0.12 * amount * (x - 0.5) * (1 - np.abs(2 * x - 1))
    return np.clip(x, 0, 1)


def darken(img, amt):
    return img * (1 - amt)


# ---------------------------------------------------------------- compositing
def over(dst, rgb, alpha):
    """alpha-composite: alpha (H,W) or scalar."""
    if np.isscalar(alpha):
        return dst * (1 - alpha) + rgb * alpha
    a = alpha[..., None]
    return dst * (1 - a) + rgb * a


def fill(color, w=W, h=H):
    out = np.empty((h, w, 3), np.float32)
    out[:] = color
    return out


def blit(frame, spr, cx, cy, alpha=1.0, scale=1.0, anchor="c"):
    """spr: uint8 RGBA (h,w,4) sprite. Places it with anchor at (cx,cy)."""
    if alpha <= 0.004:
        return frame
    if scale != 1.0:
        sw = max(1, int(round(spr.shape[1] * scale)))
        sh = max(1, int(round(spr.shape[0] * scale)))
        spr = cv2.resize(spr, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    sh, sw = spr.shape[:2]
    if anchor == "c":
        x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    elif anchor == "l":
        x0, y0 = int(round(cx)), int(round(cy - sh / 2))
    elif anchor == "r":
        x0, y0 = int(round(cx - sw)), int(round(cy - sh / 2))
    elif anchor == "tl":
        x0, y0 = int(round(cx)), int(round(cy))
    else:
        raise ValueError(anchor)
    fh, fw = frame.shape[:2]
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(fw, x0 + sw), min(fh, y0 + sh)
    if ax1 <= ax0 or ay1 <= ay0:
        return frame
    s = spr[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0].astype(np.float32) / 255
    a = s[..., 3:4] * alpha
    region = frame[ay0:ay1, ax0:ax1]
    frame[ay0:ay1, ax0:ax1] = region * (1 - a) + s[..., :3] * a
    return frame


# ---------------------------------------------------------------- text sprites
def _c255(c, a=1.0):
    return tuple(int(round(v * 255)) for v in c) + (int(round(a * 255)),)


@functools.lru_cache(maxsize=4096)
def text_sprite(text, fname, size, color=WHITE, tracking=0, shadow=0.55, shadow_r=10, stroke=0,
                stroke_color=INK, pad=None):
    f = font(fname, size)
    # measure with tracking
    chars = list(text)
    widths = [f.getlength(ch) for ch in chars]
    tw = int(sum(widths) + tracking * max(0, len(chars) - 1))
    asc, desc = f.getmetrics()
    th = asc + desc
    pad = pad if pad is not None else int(shadow_r * 2.2 + stroke + 6)
    w, h = tw + 2 * pad, th + 2 * pad
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = pad
    for ch, cw in zip(chars, widths):
        d.text((x, pad), ch, font=f, fill=_c255(color), stroke_width=stroke, stroke_fill=_c255(stroke_color))
        x += cw + tracking
    if shadow > 0:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sh.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(shadow_r)))
        sh_arr = np.array(sh)
        sh_arr[..., 3] = (sh_arr[..., 3].astype(np.float32) * shadow).astype(np.uint8)
        sh = Image.fromarray(sh_arr)
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        base.alpha_composite(sh, (0, int(shadow_r * 0.35)))
        base.alpha_composite(layer)
        layer = base
    return np.array(layer)


def text_block(lines, fname, size, color=WHITE, line_gap=1.0, align="c", tracking=0, shadow=0.55, shadow_r=10,
               colors=None):
    """Multi-line sprite. lines: list of str. colors: optional per-line color."""
    sprs = [text_sprite(l, fname, size, (colors[i] if colors else color), tracking, shadow, shadow_r)
            for i, l in enumerate(lines)]
    lh = int(size * line_gap)
    pad = sprs[0].shape[0] - font(fname, size).getmetrics()[0] - font(fname, size).getmetrics()[1]
    w = max(s.shape[1] for s in sprs)
    h = lh * (len(sprs) - 1) + sprs[-1].shape[0]
    out = np.zeros((h, w, 4), np.uint8)
    for i, s in enumerate(sprs):
        x = (w - s.shape[1]) // 2 if align == "c" else 0 if align == "l" else w - s.shape[1]
        y = i * lh
        region = out[y:y + s.shape[0], x:x + s.shape[1]]
        a = s[..., 3:4].astype(np.float32) / 255
        region[..., :3] = (region[..., :3] * (1 - a) + s[..., :3] * a).astype(np.uint8)
        region[..., 3] = np.maximum(region[..., 3], s[..., 3])
    return out


@functools.lru_cache(maxsize=512)
def pill_sprite(text, fname, size, fg, bg, bg_alpha=0.9, padx=26, pady=12, radius=None, border=None, dot=None):
    """Rounded label. Drawn at 3x and downsampled for clean anti-aliasing."""
    S = 3
    f = font(fname, size * S)
    tw = f.getlength(text)
    asc, desc = f.getmetrics()
    th = asc + desc
    dot_w = (size * 0.55 + size * 0.45) * S if dot else 0
    w = int(tw + 2 * padx * S + dot_w)
    h = int(th * 0.82 + 2 * pady * S)
    r = radius * S if radius is not None else h // 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], r, fill=_c255(bg, bg_alpha),
                        outline=_c255(border) if border else None, width=int(2 * S) if border else 0)
    x = padx * S
    if dot:
        dr = size * 0.28 * S
        cy = h / 2
        d.ellipse([x, cy - dr, x + 2 * dr, cy + dr], fill=_c255(dot))
        x += dot_w
    d.text((x, (h - th * 0.82) / 2 - asc * 0.1), text, font=f, fill=_c255(fg))
    im = im.resize((w // S, h // S), Image.LANCZOS)
    return np.array(im)


@functools.lru_cache(maxsize=256)
def rrect_sprite(w, h, r, color, alpha=1.0, border=None, border_w=0, shadow=0):
    S = 3
    pad = int(shadow * 2.5)
    im = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    box = [pad * S, pad * S, (pad + w) * S - 1, (pad + h) * S - 1]
    d.rounded_rectangle(box, r * S, fill=_c255(color, alpha),
                        outline=_c255(border) if border else None, width=border_w * S)
    im = im.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)
    if shadow:
        a = im.getchannel("A").filter(ImageFilter.GaussianBlur(shadow))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(a.point(lambda v: int(v * 0.6)))
        base = Image.new("RGBA", im.size, (0, 0, 0, 0))
        base.alpha_composite(sh, (0, shadow // 2))
        base.alpha_composite(im)
        im = base
    return np.array(im)


def paste_rgb(frame, img, x0, y0, alpha=1.0, mask=None):
    """Paste RGB float image at (x0,y0) with optional mask (h,w)."""
    h, w = img.shape[:2]
    fh, fw = frame.shape[:2]
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(fw, x0 + w), min(fh, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0:
        return frame
    src = img[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    a = alpha if mask is None else mask[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0][..., None] * alpha
    frame[ay0:ay1, ax0:ax1] = frame[ay0:ay1, ax0:ax1] * (1 - a) + src * a
    return frame


@functools.lru_cache(maxsize=64)
def rrect_mask(w, h, r):
    m = np.zeros((h * 2, w * 2), np.uint8)
    cv2.rectangle(m, (r * 2, 0), (w * 2 - r * 2, h * 2), 255, -1)
    cv2.rectangle(m, (0, r * 2), (w * 2, h * 2 - r * 2), 255, -1)
    for cx, cy in [(r * 2, r * 2), (w * 2 - r * 2, r * 2), (r * 2, h * 2 - r * 2), (w * 2 - r * 2, h * 2 - r * 2)]:
        cv2.circle(m, (cx, cy), r * 2, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


# ---------------------------------------------------------------- global finishing
_rng = np.random.default_rng(7)
_GRAIN = [(_rng.standard_normal((H // 2, W // 2)).astype(np.float32)) for _ in range(6)]


def vignette_mask(strength=0.55, power=2.2):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (x - W / 2) / (W / 2), (y - H / 2) / (H / 2)
    d = np.sqrt(nx ** 2 * 1.0 + ny ** 2 * 0.75)
    return (1 - strength * np.clip(d, 0, 1.4) ** power / 1.4 ** power * 1.6).clip(0.25, 1).astype(np.float32)


VIG = vignette_mask()


def finish(frame, fi, grain=0.035, vig=1.0):
    if vig > 0:
        frame = frame * (1 - vig + vig * VIG[..., None])
    if grain > 0:
        g = _GRAIN[fi % len(_GRAIN)]
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        lum = frame.mean(-1, keepdims=True)
        frame = frame + g[..., None] * grain * (0.35 + 0.65 * (1 - np.abs(2 * lum - 1)))
    return np.clip(frame, 0, 1)


def radial_glow(color, cx, cy, r, strength):
    y, x = np.ogrid[0:H, 0:W]
    d = ((x - cx) ** 2 + (y - cy) ** 2) / (r * r)
    m = np.exp(-d).astype(np.float32) * strength
    return m[..., None] * np.array(color, np.float32)


def draw_arrow_down(frame, cx, y0, y1, color, alpha=1.0, th=4):
    ov = frame.copy()
    c = tuple(float(v) for v in color)
    cv2.line(ov, (int(cx), int(y0)), (int(cx), int(y1)), c, th, cv2.LINE_AA)
    cv2.line(ov, (int(cx - 14), int(y1 - 14)), (int(cx), int(y1)), c, th, cv2.LINE_AA)
    cv2.line(ov, (int(cx + 14), int(y1 - 14)), (int(cx), int(y1)), c, th, cv2.LINE_AA)
    return over(frame, ov, alpha)


def hstack(sprites, gap=0, valign="c"):
    h = max(s.shape[0] for s in sprites)
    w = sum(s.shape[1] for s in sprites) + gap * (len(sprites) - 1)
    out = np.zeros((h, w, 4), np.uint8)
    x = 0
    for s in sprites:
        y = (h - s.shape[0]) // 2
        region = out[y:y + s.shape[0], x:x + s.shape[1]]
        a = s[..., 3:4].astype(np.float32) / 255
        region[..., :3] = (region[..., :3] * (1 - a) + s[..., :3] * a).astype(np.uint8)
        region[..., 3] = np.maximum(region[..., 3], s[..., 3])
        x += s.shape[1] + gap
    return out


@functools.lru_cache(maxsize=128)
def flow_sprite(parts, fname, size, colors, gap=14, shadow=0.5):
    """parts: tuple of words joined by arrows. colors: tuple per part."""
    sprs = []
    for i, p in enumerate(parts):
        if i:
            sprs.append(text_sprite("\u2192", "dejavu", int(size * 0.9), DIM, shadow=shadow, shadow_r=6))
        sprs.append(text_sprite(p, fname, size, colors[i], shadow=shadow, shadow_r=6))
    return hstack(sprs, gap - 24)
