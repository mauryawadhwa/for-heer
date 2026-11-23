import sys
import time
from pathlib import Path

import numpy as np
import pygame

TAU = np.pi * 2
GOLDEN = np.pi * (3 - np.sqrt(5))
LOOP = 20.0
PERIOD = 2.0
BEAT0 = 0.2

T_BODY = (0.0, 2.0)
T_TAIL = (2.0, 4.4)
T_SAILS = (2.6, 4.9)
T_RIBBONS = (3.2, 5.4)
T_OARS = (3.6, 4.6)
T_RISE = (4.2, 6.6)
T_TURN = (6.6, 8.8)
T_DIVE = (8.8, 11.2)
T_BACK = (11.2, 13.4)
T_HOME = (13.4, 15.6)
T_SCALES = (5.0, 6.8)
T_FINALE = (16.2, 18.2)
FLASH = 0.6
BURST_SPARKS, BURST_SPEED = 76, 3.0

BODY, TAIL, BACK, BELLY, RIB0, RIB1, OAR0, OAR1, MARK = range(9)
HARD = np.isin(np.arange(9), (BODY, TAIL, BACK, BELLY, RIB0, RIB1, OAR0, OAR1))
POOLS = ["1il!;:", "rtfjcvxz=+", "aeoun0{}[]<>", "08B$#&%Q"]
C = lambda ch: ord(ch) - 32
FILL = np.array([[C(p[k % len(p)]) for k in range(12)] for p in POOLS])
SLOPE = np.array([C(c) for c in "-/|\\-"])

WHITE = np.array([1.00, 0.98, 0.94])
GOLD = np.array([1.00, 0.82, 0.38])
PALETTES = np.array([
    [[0.10, 0.36, 1.00], [0.30, 0.86, 1.00], [0.40, 0.30, 1.00], [0.90, 0.26, 0.92], [1.00, 0.30, 0.55]],
    [[0.95, 0.10, 0.16], [1.00, 0.55, 0.35], [0.90, 0.08, 0.14], [1.00, 0.30, 0.08], [1.00, 0.58, 0.10]],
    [[0.04, 0.66, 0.74], [0.60, 1.00, 0.90], [0.08, 0.72, 0.70], [0.98, 0.84, 0.14], [1.00, 0.96, 0.58]],
    [[0.14, 0.84, 0.36], [0.76, 1.00, 0.62], [1.00, 0.86, 0.92], [1.00, 0.44, 0.66], [0.96, 0.14, 0.42]],
    [[0.20, 0.85, 1.00], [0.90, 0.98, 1.00], [0.30, 0.74, 1.00], [0.65, 0.90, 1.00], [0.95, 0.98, 1.00]],
    [[1.00, 0.50, 0.10], [1.00, 0.90, 0.70], [0.12, 0.30, 1.00], [0.20, 0.66, 1.00], [0.60, 0.94, 1.00]],
    [[0.30, 0.24, 1.00], [0.66, 0.60, 1.00], [0.18, 1.00, 0.52], [0.18, 0.86, 1.00], [0.72, 0.48, 1.00]],
    [[0.94, 0.96, 1.00], GOLD, [1.00, 0.86, 0.50], [1.00, 0.70, 0.40], [1.00, 0.95, 0.80]],
])
WAVE_SPEED = 1.25
PALE = lambda rgb, k: rgb + (WHITE - rgb) * k
BUBBLE = np.array([0.50, 0.74, 1.00])
SPARK_INK = np.array([[0.16, 0.86, 1.00], [1.00, 0.30, 0.50], GOLD, [0.90, 0.26, 0.92], WHITE])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    y = 1 - 2 * i / n
    r = np.sqrt(1 - y * y)
    return np.stack([r * np.cos(GOLDEN * i), y, r * np.sin(GOLDEN * i)], 1)


def slope_char(dx, dy):
    ang = np.degrees(np.arctan2(dy, dx)) % 180
    return SLOPE[np.digitize(ang, [25, 72, 108, 155])]


def outline_letter(gx, gy):
    ax, ay = np.abs(gx), np.abs(gy)
    c = np.where(ax > 2 * ay, C("|"), np.where(ay > 2 * ax, np.where(gy > 0, C("_"), C("-")),
                 np.where(gx * gy > 0, C("/"), C("\\"))))
    return np.where(ax + ay == 0, C("+"), c)


def beat(t):
    ph = ((np.asarray(t, float) - BEAT0) / PERIOD) % 1.0
    return np.where(ph < 0.22, smooth(ph / 0.22), 1 - smooth((ph - 0.22) / 0.78))


def flare(t):
    t = np.asarray(t, float)
    return beat(t) * (0.35 + 0.65 * smooth((t - T_RIBBONS[1] + 0.3) / 0.5))


def stroke(t):
    return 0.70 + 0.50 * flare(t)


NOSE, LENGTH = 0.42, 0.80
TAIL_X = NOSE - 0.97 * LENGTH
PIVOT = np.array([-0.26, -0.06])
HOME_Y = 0.02
TIP, STEEP = 0.22, 1.7
CLIMB = 0.24
FACING, WEAVE = 0.60, 0.22
BODY_LAG, FIN_LAG = 0.32, 0.85

SWIM = 0.050
TAIL_BEAT = TAU / PERIOD
ALONG_BODY, ALONG_FIN = 5.0, 2.6


def span(t, between):
    return smooth((np.asarray(t, float) - between[0]) / (between[1] - between[0]))


def heading(t):
    t = np.asarray(t, float)
    return FACING + (np.pi - 2 * FACING) * (span(t, T_TURN) - span(t, T_BACK)) + WEAVE * np.sin(TAU * t / 9.0 + 0.4) \
        + 0.06 * np.sin(1.1 * t)


def height(t):
    return CLIMB * (span(t, T_RISE) - 2 * span(t, T_DIVE) + span(t, T_HOME)) + 0.05 * np.sin(0.9 * np.asarray(t, float))


def carry(p, t, lag=0.0):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    behind = np.clip((NOSE - x) / LENGTH, 0.0, 1.0)
    td = t - BODY_LAG * behind - lag
    z = z + SWIM * stroke(t) * np.clip(behind - 0.25, 0, None) ** 1.5 / 0.65 * np.sin(ALONG_BODY * behind - TAIL_BEAT * t)
    x, y = x - PIVOT[0], y - PIVOT[1]
    tip = TIP + 0.05 * np.sin(0.7 * td + 1.0) + STEEP * (height(td + 0.05) - height(td - 0.05)) / 0.1
    x, y = x * np.cos(tip) - y * np.sin(tip), x * np.sin(tip) + y * np.cos(tip)
    yaw = heading(td)
    return np.stack([x * np.cos(yaw) - z * np.sin(yaw) + 0.05 * np.sin(0.45 * td + 0.4),
                     y + HOME_Y + height(td),
                     x * np.sin(yaw) + z * np.cos(yaw)], -1)


_U = np.linspace(0, 1, 400)
_H = np.interp(_U, [0, 0.03, 0.09, 0.18, 0.32, 0.48, 0.66, 0.84, 1.0],
               [0.004, 0.026, 0.058, 0.094, 0.128, 0.128, 0.102, 0.068, 0.047])
_k = np.hanning(21)
_H = np.convolve(np.pad(_H, 10, mode="edge"), _k / _k.sum(), mode="valid")


def tall(u):
    return np.interp(u, _U, _H)


def arch(u):
    return 0.012 * np.sin(np.pi * np.clip(u, 0, 1)) - 0.010 * np.clip(1 - u / 0.2, 0, 1) ** 2


def back_at(x):
    u = (NOSE - x) / LENGTH
    return arch(u) + tall(u)


def belly_at(x):
    u = (NOSE - x) / LENGTH
    return arch(u) - tall(u)


def body(u, th):
    h = tall(u)
    return np.stack([NOSE - LENGTH * u, arch(u) + h * np.sin(th), h * (0.56 - 0.22 * u) * np.cos(th)], -1)


def body_points(n):
    grid = np.linspace(0, 1, 2000)
    area = np.cumsum(tall(grid) + 0.01)
    return np.interp((np.arange(n) + 0.5) / n, area / area[-1], grid), (np.arange(n) * GOLDEN) % TAU


NUDGE = 2e-3
BODY_U, BODY_TH = body_points(8000)
BODY_AT = [body(BODY_U, BODY_TH), body(BODY_U + NUDGE, BODY_TH), body(BODY_U, BODY_TH + NUDGE)]
SCALES = 0.5 + 0.5 * np.cos(BODY_U * 170 + BODY_TH * 8) * np.cos(BODY_U * 170 - BODY_TH * 8)
_pale = np.clip(np.maximum(smooth((0.2 - BODY_U) / 0.12), smooth((-np.sin(BODY_TH) - 0.2) / 0.6)), 0, 1)[:, None]
BODY_INK = np.array([k[0] + (k[1] - k[0]) * (0.75 * _pale + 0.40 * (SCALES[:, None] > 0.78) * (BODY_U[:, None] > 0.2))
                     for k in PALETTES])
GILL = 0.22
BODY_WAY = np.abs(BODY_U - GILL) * LENGTH

EYES = body(np.array([0.085, 0.085]), np.array([0.42, np.pi - 0.42])) + [[0, 0, 0.012], [0, 0, -0.012]]
_g = np.linspace(-0.95, 0.95, 16)
GILLS = np.concatenate([body(0.225 + 0.035 * np.cos(_g * 1.4), _g) + [0, 0, 0.006],
                        body(0.225 + 0.035 * np.cos(_g * 1.4), np.pi - _g) - [0, 0, 0.006]])
N_SCALES = 12
SCALE_AT = body(0.30 + 0.52 * ((np.arange(N_SCALES) * 5) % N_SCALES) / (N_SCALES - 1),
                np.where(np.arange(N_SCALES) % 2 == 0, 1, -1) * 0.18 * np.cos(np.arange(N_SCALES) * 2.1)
                + np.pi * (np.arange(N_SCALES) % 2))
SCALE_AT = SCALE_AT + np.where(np.arange(N_SCALES) % 2 == 0, 0.01, -0.01)[:, None] * [0, 0, 1]
SCALE_T0 = T_SCALES[0] + (T_SCALES[1] - T_SCALES[0]) * np.arange(N_SCALES) / (N_SCALES - 1)
MOUTH = np.array([NOSE + 0.01, arch(0.0) + 0.012, 0.0])


def grown(t):
    return float(np.clip((t - T_BODY[0]) / (T_BODY[1] - T_BODY[0]), 0, 1) ** 1.2)


def opened(t, span):
    return float(smooth((t - span[0]) / (span[1] - span[0])))


PLEATS = 4.5


def wave(a, r, t, root, big, k):
    swing = big * stroke(t) * r ** 1.3 * np.sin(root + ALONG_FIN * r - TAIL_BEAT * t + 0.9 * (a - 0.5))
    pleat = 0.022 * r ** 1.2 * np.sin(PLEATS * TAU * a + 1.3 * np.sin(0.6 * t + k) - 1.8 * r + k)
    return swing + pleat + 0.014 * r ** 4 * np.sin(19 * a - 3.4 * t + 2 * k)


def veil(root, ex, ey, long, out, g):
    step = long * g ** 0.8 / (out.shape[1] - 1)
    rise = np.diff(out, axis=1, prepend=out[:, :1])
    flat = np.sqrt(np.maximum(step ** 2 - rise ** 2, (0.60 * step) ** 2))
    flat[:, 0] = 0
    return np.stack([root[0] + np.cumsum(flat * ex, 1), root[1] + np.cumsum(flat * ey, 1), root[2] + out], -1), \
        FIN_LAG * long * np.linspace(0, 1, out.shape[1])


def scallop(a, rays):
    return 0.94 + 0.06 * np.abs(np.cos(rays * np.pi * a)) ** 0.5


TAIL_LEN, TAIL_SPREAD, TAIL_RAYS = 0.70, 2.50, 15


def tail(a, r, t, g):
    spread = TAIL_SPREAD * (0.16 + 0.84 * g) * (1 + 0.08 * flare(t - 0.25 * r))
    ang = (a - 0.5) * spread * (1 - 0.22 * r * r) + 0.07 * r * np.sin(3.2 * a - 1.3 * t) * g
    ang = ang - 0.16 * r * r * np.cos(ang)
    long = TAIL_LEN * (0.90 + 0.10 * np.sin(np.pi * a)) * scallop(a, TAIL_RAYS)
    out = wave(a, r, t, ALONG_BODY, 0.17, 0.0) * g
    return veil((TAIL_X, 0.040 * (2 * a - 1), 0.0), -np.cos(ang), np.sin(ang), long, out, g)


BACK_RAYS, BELLY_RAYS = 9, 11
SWEEP = 0.60


def sail(a, r, t, g, up):
    if up > 0:
        x = 0.00 - 0.24 * a
        lean, long, rays = 0.10 + 0.30 * a, (0.32 + 0.28 * a) * np.sin(np.pi * np.clip(0.14 + 0.72 * a, 0, 1)) ** 0.5, BACK_RAYS
        y0 = back_at(x) - 0.012
    else:
        x = 0.04 - 0.30 * a
        lean, long, rays = 0.08 + 0.26 * a, (0.26 + 0.42 * a) * np.sin(np.pi * np.clip(0.12 + 0.74 * a, 0, 1)) ** 0.5, BELLY_RAYS
        y0 = belly_at(x) + 0.012
    lean = 1.50 + (lean + SWEEP * r - 0.10 * flare(t - 0.25 * r) - 1.50) * g + 0.07 * r * np.sin(3.0 * a - 1.3 * t + up) * g
    out = wave(a, r, t, ALONG_BODY * (NOSE - x) / LENGTH, 0.13 + 0.03 * (up < 0), 1.7 + up) * g
    return veil((x, y0, 0.012 + 0.010 * r), -np.sin(lean), up * np.cos(lean), long * scallop(a, rays), out, g)


RIBBON_LEN = 0.60


def ribbon(k, s, w, t, g):
    side = 2 * k - 1
    reach = RIBBON_LEN * g * s
    wide = 0.036 * np.sin(np.pi * np.clip(s, 0, 1) ** 0.55) ** 0.8 * (0.4 + 0.6 * g)
    x = 0.13 - RIBBON_LEN * g * (0.22 * s + 0.30 * s * s) + w * wide
    y = belly_at(0.13) + 0.012 - RIBBON_LEN * g * (0.96 * s - 0.20 * s * s) + 0.3 * w * wide
    z = side * (0.030 + 0.07 * s) + 0.11 * s ** 1.3 * np.sin(1.4 + 3.4 * s - TAIL_BEAT * t + 2.4 * k) + 0.012 * w * np.sin(9 * s - 3 * t)
    return np.stack([x, y, z], -1), FIN_LAG * reach


OAR_LEN, OAR_RAYS = 0.15, 5


def oar(k, a, r, t, g):
    side = 2 * k - 1
    swing = 0.80 + 0.55 * np.sin(7.5 * t + np.pi * k - 1.2 * r)
    ang = (a - 0.5) * 1.9 * (0.3 + 0.7 * g)
    reach = OAR_LEN * g * r
    x = 0.205 - np.cos(ang) * np.cos(swing) * reach
    y = -0.025 + np.sin(ang) * reach
    z = side * (tall(0.27) * 0.5 + 0.004 + np.cos(ang) * np.sin(swing) * reach)
    return np.stack([x, y, z], -1), 0.0 * reach


def sheet(na, nr):
    return np.meshgrid(np.linspace(0, 1, na), np.linspace(0, 1, nr), indexing="ij")


def veil_ink(r):
    out = []
    for skin, face, v0, v1, v2 in PALETTES:
        ink = np.where((r < 0.5)[..., None], v0 + (v1 - v0) * smooth(r / 0.5)[..., None],
                       v1 + (v2 - v1) * smooth((r - 0.5) / 0.45)[..., None])
        out.append((ink + (PALE(v2, 0.6) - ink) * (r > 0.95)[..., None]).reshape(-1, 3))
    return np.array(out)


def glitter(grid, seed):
    rng = np.random.default_rng(seed)
    r = grid[-1].ravel()
    at = np.flatnonzero((rng.random(len(r)) < 0.012) & (r > 0.4))
    return at, 2 + 3 * rng.random(len(at)), rng.random(len(at)) * TAU


TAIL_A, TAIL_R = sheet(172, 66)
BACK_A, BACK_R = sheet(92, 48)
BELLY_A, BELLY_R = sheet(106, 48)
RIB_K, RIB_S, RIB_W = np.meshgrid(np.arange(2), np.linspace(0, 1, 150), np.linspace(-1, 1, 13), indexing="ij")
OAR_K, OAR_A, OAR_R = np.meshgrid(np.arange(2), np.linspace(0, 1, 40), np.linspace(0, 1, 26), indexing="ij")
TAIL_INK, BACK_INK, BELLY_INK, RIB_INK = veil_ink(TAIL_R), veil_ink(BACK_R), veil_ink(BELLY_R), veil_ink(RIB_S)
OAR_INK = np.array([np.broadcast_to(PALE(k[3], 0.20), OAR_R.shape + (3,)).reshape(-1, 3) for k in PALETTES])
WAY_TAIL, WAY_BACK, WAY_BELLY, WAY_RIB, WAY_OAR = (0.60, TAIL_LEN), (0.36, 0.50), (0.34, 0.58), (0.12, RIBBON_LEN), (0.05, OAR_LEN)
TAIL_GLITTER, BACK_GLITTER, BELLY_GLITTER = glitter((TAIL_A, TAIL_R), 1), glitter((BACK_A, BACK_R), 2), \
    glitter((BELLY_A, BELLY_R), 3)

EYE_POS, TARGET = np.array([0.0, -0.25, 6.0]), np.array([0.0, 0.0, 0.0])
FWD = unit(TARGET - EYE_POS)
RIGHT = unit(np.cross(FWD, [0, 1.0, 0]))
UP = np.cross(RIGHT, FWD)
BASIS = np.stack([RIGHT, UP, FWD])
EYE_CAM = BASIS @ EYE_POS
ZOOM = 7.4
VIEW = unit(EYE_POS - TARGET)
KEY = unit([-0.50, 0.62, 0.60])
HALF = unit(KEY + VIEW)


def to_cam(p):
    return p @ BASIS.T - EYE_CAM


def project_cam(cam, cols, rows, aspect):
    z = cam[..., 2]
    zz = np.maximum(z, 0.2)
    return (0.5 + cam[..., 0] / zz * ZOOM * 0.5) * cols, (0.5 - cam[..., 1] / zz * ZOOM * 0.5 * aspect) * rows, z


def scene(t, cols, rows, aspect):
    SX, SY, Z, L, RGBs, PRT, G, AREA = [], [], [], [], [], [], [], []
    flares = []

    old, new, wave_at = COLOURS.now(t)

    def lit(way, inks, lum):
        on = smooth((wave_at - way) / 0.2)[:, None]
        band = 0.85 * np.exp(-((wave_at - way) / 0.06) ** 2)
        rgb = inks[old] + (inks[new] - inks[old]) * on
        return rgb + (WHITE - rgb) * np.clip(band, 0, 1)[:, None] * 0.65, lum + 0.45 * band

    def add(p, lum, rgb, part, glyph, area=None):
        sx, sy, z = project_cam(to_cam(p), cols, rows, aspect)
        SX.append(sx), SY.append(sy), Z.append(z), L.append(lum), RGBs.append(rgb), G.append(glyph)
        PRT.append(part if isinstance(part, np.ndarray) else np.full(len(lum), part))
        AREA.append(np.ones(len(lum)) if area is None else area)

    edge = grown(t)
    m = slice(0, int(np.searchsorted(BODY_U, edge)))
    u = BODY_U[m]
    p, pu, pt = (carry(at[m], t) for at in BODY_AT)
    n = unit(np.cross(pu - p, pt - p))
    n = n * np.sign(n @ VIEW)[:, None]
    lum = 0.60 + 0.36 * np.clip(n @ KEY, 0, 1) + 0.55 * np.clip(n @ HALF, 0, 1) ** 20 + 0.26 * (1 - np.abs(n @ VIEW)) ** 2 \
        + 0.24 * (SCALES[m] - 0.5) * (u > 0.2)
    rgb, lum = lit(BODY_WAY[m], BODY_INK[:, m], lum)
    rgb = rgb + (WHITE - rgb) * np.clip(lum - 1.0, 0, 0.5)[:, None]
    glyph = np.full(len(u), -1)
    if t < T_BODY[1]:
        fresh = (u > edge - 0.035) & (np.arange(len(u)) % 2 == 0)
        glyph[fresh], lum[fresh], rgb[fresh] = C("*"), 1.3, WHITE
    whole = float(np.clip(1 - (t - T_BODY[1]) / FLASH, 0, 1) ** 2) if t >= T_BODY[1] else 0.0
    rgb, lum = rgb + (WHITE - rgb) * 0.7 * whole, lum + 0.4 * whole
    add(p, lum, rgb, BODY, glyph)
    if whole > 0.05:
        for at in carry(body(np.array([0.10, 0.52, 0.97]), np.array([1.1, 1.5, 1.2])), t):
            flares.append((at, int(2 + 5 * whole), whole, WHITE, True))

    def fin(surface, grid, g, part, inks, way, done, rays=None, sparkle=None, flare_at=()):
        local, lag = surface
        p = carry(local, t, lag)
        sx, sy, z = project_cam(to_cam(p), cols, rows, aspect)
        n = unit(np.cross(np.gradient(p, axis=-3), np.gradient(p, axis=-2)))
        n = (n * np.sign(n @ VIEW)[..., None]).reshape(-1, 3)
        a, r = grid[-2].ravel(), grid[-1].ravel()
        lum = 0.30 + 0.78 * np.clip(n @ KEY, 0, 1) ** 0.8 + 0.40 * np.clip(n @ HALF, 0, 1) ** 16 + 0.22 * smooth((r - 0.86) / 0.12)
        rgb, lum = lit(way[0] + way[1] * g * r, inks, lum)
        rgb = rgb + (WHITE - rgb) * np.clip((lum - 0.95) * 0.9, 0, 0.55)[:, None]
        glyph = np.full(len(r), -1)
        if rays is not None:
            ray = (np.cos(rays * TAU * a) > 0.88) & (r > 0.14)
            glyph[ray] = slope_char(np.gradient(sx, axis=1), -np.gradient(sy, axis=1) / aspect * cols / rows).ravel()[ray]
            lum[ray] = 0.62 + 0.55 * lum[ray]
            rgb[ray] = PALE(rgb[ray], 0.40)
        if sparkle is not None and g >= 1:
            at, speed, phase = sparkle
            on = at[np.sin(t * speed + phase) > 0.94]
            glyph[on], lum[on], rgb[on] = C("*"), 1.35, WHITE
        if g < 1:
            tip = (r > 0.98) & (np.arange(len(r)) // grid[-1].shape[-1] % 3 == 0)
            glyph[tip], lum[tip] = C("*"), 1.3
            rgb = np.where(tip[:, None], WHITE, rgb)
        whole = float(np.clip(1 - (t - done) / FLASH, 0, 1) ** 2) if t >= done else 0.0
        if whole > 0.05:
            rgb, lum = rgb + (WHITE - rgb) * 0.6 * whole, lum + 0.35 * whole
            for at in flare_at:
                flares.append((p[at], int(2 + 5 * whole), whole, WHITE, True))
        area = np.abs(np.gradient(sx, axis=-2) * np.gradient(sy, axis=-1) - np.gradient(sx, axis=-1) * np.gradient(sy, axis=-2))
        SX.append(sx.ravel()), SY.append(sy.ravel()), Z.append(z.ravel()), L.append(lum), RGBs.append(rgb), G.append(glyph)
        PRT.append(part if isinstance(part, np.ndarray) else np.full(len(r), part))
        AREA.append(np.where(glyph >= 0, 1.0, 1.6 * area.ravel() + 0.02))

    fins = 0
    g = opened(t, T_TAIL)
    if g > 0.01:
        fin(tail(TAIL_A, TAIL_R, t, g), (TAIL_A, TAIL_R), g, TAIL, TAIL_INK, WAY_TAIL, T_TAIL[1], TAIL_RAYS, TAIL_GLITTER,
            [(20, -1), (86, -1), (150, -1)])
    fins += g > 0.99
    g = opened(t, T_SAILS)
    if g > 0.01:
        fin(sail(BACK_A, BACK_R, t, g, 1), (BACK_A, BACK_R), g, BACK, BACK_INK, WAY_BACK, T_SAILS[1], BACK_RAYS, BACK_GLITTER,
            [(30, -1), (70, -1)])
        fin(sail(BELLY_A, BELLY_R, t, g, -1), (BELLY_A, BELLY_R), g, BELLY, BELLY_INK, WAY_BELLY, T_SAILS[1], BELLY_RAYS,
            BELLY_GLITTER, [(36, -1), (82, -1)])
    fins += 2 * (g > 0.99)
    g = opened(t, T_RIBBONS)
    if g > 0.01:
        fin(ribbon(RIB_K, RIB_S, RIB_W, t, g), (RIB_S, RIB_S), g, RIB0 + RIB_K.ravel(), RIB_INK, WAY_RIB, T_RIBBONS[1],
            flare_at=[(0, -1, 6), (1, -1, 6)])
    fins += 2 * (g > 0.99)
    g = opened(t, T_OARS)
    if g > 0.01:
        fin(oar(OAR_K, OAR_A, OAR_R, t, g), (OAR_A, OAR_R), g, OAR0 + OAR_K.ravel(), OAR_INK, WAY_OAR, T_OARS[1], OAR_RAYS)
    fins += 2 * (g > 0.99)

    if edge > 0.12:
        add(carry(EYES, t), np.full(2, 1.3), np.tile(WHITE, (2, 1)), MARK, np.full(2, C("@")))
    if edge > 0.3:
        p = carry(GILLS, t).reshape(2, -1, 3)
        cam = to_cam(p)
        stroke_ = slope_char(np.gradient(cam[..., 0] / cam[..., 2], axis=1), np.gradient(cam[..., 1] / cam[..., 2], axis=1))
        add(p.reshape(-1, 3), np.full(stroke_.size, 1.0), np.tile(PALE(PALETTES[new, 1], 0.5), (stroke_.size, 1)), MARK, stroke_.ravel())

    on = np.flatnonzero(t >= SCALE_T0)
    if len(on):
        p = carry(SCALE_AT[on], t)
        age = t - SCALE_T0[on]
        hot = np.clip(1 - age / FLASH, 0, 1) ** 2
        twinkle = 0.5 + 0.5 * np.sin(t * 5 + on * 2.4)
        add(p, 1.05 + 0.3 * twinkle + 0.3 * hot, GOLD + (WHITE - GOLD) * hot[:, None], MARK,
            np.where(twinkle > 0.5, C("*"), C("o")))
        for i in np.flatnonzero((age < 0.45) | (twinkle > 0.975)):
            k = max(float(hot[i]), 0.45)
            flares.append((p[i], int(2 + 3 * k), k, WHITE if age[i] < 0.45 else GOLD, True))

    seed = float(np.clip(1 - t / 1.0, 0, 1))
    if seed > 0.05:
        flares.append((carry(MOUTH, t), int(2 + 6 * seed), seed, WHITE, False))

    info = {"beats": max(0, int((t - BEAT0) / PERIOD) + 1), "fins": int(fins), "colours": int(new)}
    return [np.concatenate(x) for x in (SX, SY, Z, L, RGBs, PRT, G, AREA)], flares, info


_HASH = {}


def HASH(rows, cols):
    if (rows, cols) not in _HASH:
        r, c = np.mgrid[0:rows, 0:cols]
        _HASH[rows, cols] = (r * 7919 + c * 104729) % 9973
    return _HASH[rows, cols]


def bodies(sx, sy, z, lum, rgb, part, glyph, area, cols, rows, t):
    if len(sx) == 0:
        return np.zeros((rows, cols), int), np.zeros((rows, cols, 3)), np.full((rows, cols), np.inf)
    ok = np.flatnonzero((sx >= 0) & (sx < cols) & (sy >= 0) & (sy < rows) & (z > 0.3))
    cell = sy[ok].astype(int) * cols + sx[ok].astype(int)
    cover = np.clip(np.bincount(cell, weights=area[ok], minlength=rows * cols), 0, 1).reshape(rows, cols)
    order = np.argsort(-z[ok])
    gi = np.full(rows * cols, -1)
    gi[cell[order]] = ok[order]
    gi = gi.reshape(rows, cols)
    body = gi >= 0
    g = np.maximum(gi, 0)
    prt = np.where(body, part[g], -1)
    dep = np.where(body, z[g], np.inf)
    own = np.where(body, glyph[g], -1)
    solid = body & (own < 0)
    pp, pd = np.pad(prt, 1, constant_values=-1), np.pad(dep, 1, constant_values=np.inf)
    edge = np.zeros_like(body)
    gx, gy = np.zeros((rows, cols)), np.zeros((rows, cols))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb, nz = pp[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx], pd[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            other = (nb >= 0) & (nb != prt)
            seam = HARD[np.maximum(prt, 0)] & other & HARD[np.maximum(nb, 0)] & (prt < nb)
            out = solid & ((nb < 0) | (other & (nz > dep + 0.035)) | seam)
            gx += dx * out
            gy += dy * out
            if dx == 0 or dy == 0:
                edge |= out
    h = HASH(rows, cols)
    L = np.where(body, lum[g], 0.0)
    level = np.clip(((L - 0.42) * 4.6).astype(int), 0, 3)
    inside = FILL[level, (h + (t * 4 + h % 11).astype(int)) % 12]
    char = np.where(body, np.where(own >= 0, own, np.where(edge, outline_letter(gx, gy), inside)), 0)
    bright = np.where(own >= 0, L, np.where(edge, np.maximum(0.85, L), np.clip(0.45 + 0.55 * L / 0.9, 0.45, 1.0)))
    bright = bright * np.where(own >= 0, 1.0, 0.30 + 0.70 * cover ** 0.7)
    col = np.where(body[..., None], rgb[g] * np.minimum(bright, 1.4)[..., None], 0.0)
    return char, col, dep


def put(char, col, y, x, ch, rgb, k):
    if 0 <= y < char.shape[0] and 0 <= x < char.shape[1]:
        char[y, x], col[y, x] = C(ch), np.asarray(rgb) * k


def rays(char, col, cy, cx, reach, rgb, strength=1.0):
    rows, cols = char.shape
    for dist in range(1, reach + 1):
        fade = (1 - dist / (reach + 1)) * strength
        arms = [(0, 1, "-"), (0, -1, "-"), (1, 0, "|"), (-1, 0, "|")]
        if dist <= reach // 2:
            arms += [(1, 1, "\\"), (-1, -1, "\\"), (1, -1, "/"), (-1, 1, "/")]
        for dy, dx, ch in arms:
            yy, xx = cy + dy * dist, cx + dx * dist * (2 if dy == 0 else 1)
            k = fade * (0.7 if dx and dy else 1.0)
            if 0 <= yy < rows and 0 <= xx < cols and col[yy, xx].max() < k:
                char[yy, xx], col[yy, xx] = C(ch), rgb * k
    put(char, col, cy, cx, "*", rgb, 1.3 * strength)


def dots(char, col, dep, p, glyph, rgb, k, cols, rows, aspect):
    sx, sy, z = project_cam(to_cam(p), cols, rows, aspect)
    ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ok = np.flatnonzero((ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows) & (k > 0.05))
    ok = ok[z[ok] < dep[iy[ok], ix[ok]]]
    ok = ok[np.argsort(k[ok])]
    char[iy[ok], ix[ok]] = glyph[ok]
    col[iy[ok], ix[ok]] = rgb[ok] * k[ok, None]


N_MOTES, MOTE_FLY = 150, 1.8
MOTE = np.random.default_rng(8).random((N_MOTES, 5))
MOTE_LAND = 0.15 + MOTE[:, 0] * (T_BODY[1] - 0.15)
MOTE_GLYPH = np.array([C("."), C("+"), C("*")])
MOTE_INK = np.array([0.72, 0.97, 1.00])


def motes(t):
    q = (t - (MOTE_LAND - MOTE_FLY)) / MOTE_FLY
    live = np.flatnonzero((q > 0) & (q < 1))
    if len(live) == 0:
        return []
    q, r = q[live], MOTE[live]
    far = (1 - q) ** 1.4
    u = np.full(len(live), grown(t))
    th = np.floor(r[:, 1] * 3) * TAU / 3 + 0.5 * r[:, 4] + 3.6 * (1 - q) ** 1.5
    home = body(u, th)
    out = unit(np.stack([-0.55 + 0 * th, np.sin(th), np.cos(th)], 1))
    p = home + out * ((0.45 + 0.60 * r[:, 2]) * far)[:, None]
    p[:, 0] += (r[:, 3] - 0.5) * 0.35 * far
    return [(carry(p, t), MOTE_GLYPH[np.clip((q * 3).astype(int), 0, 2)],
             np.tile(MOTE_INK, (len(live), 1)) + (WHITE - MOTE_INK) * q[:, None], (0.35 + 0.95 * q) * smooth(q / 0.15))]


HEM_R = np.linspace(0, 1, 24)


def hem_of(sheet_fn):
    def at(a, t, g):
        A, R = np.meshgrid(a, HEM_R, indexing="ij")
        local, lag = sheet_fn(A, R, t, g)
        p = carry(local, t, lag)
        return p[:, -1], p[:, 0]
    return at


FEEDS = [(span, hem_of(fn), np.random.default_rng(seed).random((n, 4))) for span, fn, n, seed in (
    (T_TAIL, tail, 90, 41),
    (T_SAILS, lambda a, r, t, g: sail(a, r, t, g, 1), 50, 42),
    (T_SAILS, lambda a, r, t, g: sail(a, r, t, g, -1), 56, 43),
    (T_RIBBONS, lambda a, r, t, g: ribbon((a > 0.5).astype(int), r, 0 * r, t, g), 26, 44))]
FEED_FLY = 1.6


def fin_motes(t):
    out = []
    for span, hem, rnd in FEEDS:
        land = span[0] + 0.25 + rnd[:, 0] * (span[1] - span[0] - 0.25)
        q = (t - (land - FEED_FLY)) / FEED_FLY
        live = np.flatnonzero((q > 0) & (q < 1))
        if len(live) == 0:
            continue
        q, r = q[live], rnd[live]
        home = hem(r[:, 1], t, max(opened(t, span), 0.02))[0]
        rim, root = hem(r[:, 1], t, 1.0)
        way = unit(rim - root)
        side = unit(np.cross(way, VIEW))
        far = (1 - q) ** 1.4
        p = home + way * ((0.30 + 0.45 * r[:, 2]) * far)[:, None] \
            + side * (0.45 * far * np.sin(3.6 * (1 - q) ** 1.5 + TAU * r[:, 3]))[:, None]
        out.append((p, MOTE_GLYPH[np.clip((q * 3).astype(int), 0, 2)],
                    np.tile(MOTE_INK, (len(live), 1)) + (WHITE - MOTE_INK) * q[:, None], (0.35 + 0.95 * q) * smooth(q / 0.15)))
    return out


TO_HEM = sum(WAY_TAIL) / WAVE_SPEED
WAVE_RUN = TO_HEM + 0.25 / WAVE_SPEED


class Colours:
    def __init__(self):
        self.reset()

    def reset(self):
        self.starts = []
        self.waiting = 0
        self.bursts = []

    def ask(self, t):
        self.waiting += 1
        self.now(t)

    def now(self, t):
        if self.waiting and (not self.starts or t >= self.starts[-1] + WAVE_RUN):
            self.starts.append(t)
            self.waiting -= 1
            self.bursts = self.bursts[-2:] + [wave_burst(t, len(self.starts), len(self.starts) % len(PALETTES))]
        n = len(self.starts)
        if n == 0:
            return 0, 0, -9.0
        return (n - 1) % len(PALETTES), n % len(PALETTES), (t - self.starts[-1]) * WAVE_SPEED


COLOURS = Colours()


GLINT_LIFE, N_GLINTS = 1.8, 20
GLINT = np.random.default_rng(14).random((len(PALETTES), N_GLINTS, 2))
GLINT_A, GLINT_R = np.meshgrid((np.arange(N_GLINTS) + 0.5) / N_GLINTS, np.linspace(0, 1, 40), indexing="ij")


def glints(t):
    out = []
    for n, start in enumerate(COLOURS.starts):
        t0 = start + TO_HEM
        age = t - t0
        g = opened(t0, T_TAIL)
        if not 0 <= age < GLINT_LIFE or g < 0.3:
            continue
        n %= len(GLINT)
        local, lag = tail(GLINT_A, GLINT_R, t0, g)
        p = carry(local, t0, lag)
        rim, inner = p[:, -1], p[:, -8]
        p = rim + unit(rim - inner) * (0.40 * (1 - np.exp(-1.6 * age)))[..., None]
        p[:, 1] -= 0.04 * age * age
        twinkle = np.clip(np.sin(t * (5 + 6 * GLINT[n, :, 1]) + GLINT[n, :, 0] * 50), 0, 1)
        fade = (1 - age / GLINT_LIFE) ** 0.6
        out.append((p, np.where(twinkle > 0.6, C("*"), C("+")), np.tile(WHITE, (N_GLINTS, 1)), (0.45 + 0.85 * twinkle) * fade))
    return out


BREATH_T0 = np.arange(BEAT0 + 2 * PERIOD, LOOP, PERIOD)
BREATH = np.random.default_rng(31).random((len(BREATH_T0), 4, 3))
BREATH_GLYPH = np.array([C("."), C("o"), C("O")])


def breath(t):
    out = []
    for n, tb in enumerate(BREATH_T0):
        t0 = tb + 0.10 + 0.16 * np.arange(4)
        age = t - t0
        live = (age > 0) & (age < 2.6)
        if not live.any():
            continue
        p = np.stack([carry(MOUTH, x) for x in t0])
        rise = 0.22 * age + 0.10 * age * age
        p = p + np.stack([0.05 * np.sin(4 * age + 9 * BREATH[n, :, 0]) * np.minimum(age, 1), rise, 0 * age], 1)
        size = np.clip(0.4 + 0.5 * age + BREATH[n, :, 1], 0, 2.99).astype(int)
        fade = smooth(age / 0.15) * (1 - smooth((age - 2.0) / 0.6)) * live
        out.append((p, BREATH_GLYPH[size], np.tile(PALE(BUBBLE, 0.4), (4, 1)), 0.95 * fade))
    return out


DRAG, TAILS, TAIL_GAP = 1.7, 5, 0.035


def burst(t0, x, n, speed, life, inks, size, rng):
    centre = carry(np.array([x, 0.0, 0.0]), t0)
    way = fib_sphere(n)[rng.permutation(n)] * [1, 1, 0.55]
    way[:, 2] = -np.abs(way[:, 2])
    way = unit(way)
    vel = way * (speed * (0.45 + 0.55 * rng.random(n)))[:, None]
    ink = inks[rng.integers(0, len(inks), n)]
    return (t0, centre + 0.3 * size * way, vel, ink, life(rng), rng.random(n) * TAU, centre, size)


def make_bursts():
    rng = np.random.default_rng(12)
    return [burst(tb + 0.25, -0.10, BURST_SPARKS, BURST_SPEED, lambda g: 1.4 + 0.8 * g.random(BURST_SPARKS), SPARK_INK, 1.0, rng)
            for tb in T_FINALE]


BURSTS = make_bursts()

WAVE_SPARKS, WAVE_SPARK_SPEED, WAVE_BURST_SIZE = 30, 1.4, 0.55


def wave_burst(t0, k, palette):
    rng = np.random.default_rng(100 + k)
    return burst(t0, NOSE - LENGTH * GILL, WAVE_SPARKS, WAVE_SPARK_SPEED, lambda g: WAVE_RUN * (0.8 + 0.2 * g.random(WAVE_SPARKS)),
                 PALETTES[palette], WAVE_BURST_SIZE, rng)


def sparks(t):
    out = []
    for t0, start, vel, ink, life, seed, centre, size in BURSTS + COLOURS.bursts:
        if not 0 <= t - t0 < life.max():
            continue

        def at(age):
            p = start + vel * (1 - np.exp(-DRAG * max(age, 0.0))) / DRAG
            p[:, 1] -= 0.07 * age * age
            return p

        age = t - t0
        head, back = to_cam(at(age)), to_cam(at(age - TAIL_GAP))
        stroke_ = slope_char(head[:, 0] / head[:, 2] - back[:, 0] / back[:, 2],
                             head[:, 1] / head[:, 2] - back[:, 1] / back[:, 2])
        dying = age > life - np.minimum(0.6, 0.45 * life)
        flick = np.where(dying, np.sin(age * 38 + seed) > -0.2, True)
        fade = np.clip((life - age) / 0.5, 0, 1) * flick
        rgb = ink + (WHITE - ink) * max(0.0, 1 - age / 0.3)
        for j in range(TAILS, 0, -1):
            if age - TAIL_GAP * j > 0:
                out.append((at(age - TAIL_GAP * j), np.where(dying, C("."), stroke_), rgb, fade * (0.85 - 0.14 * j)))
        out.append((at(age), np.where(dying, C("+"), C("*")), rgb, fade * 1.3))
    return out


N_BUBBLES = 70
BUBBLE_AT = np.random.default_rng(3).random((N_BUBBLES, 5))
BUBBLE_GLYPH = np.array([C("."), C("."), C("."), C("o"), C("O")])


def frame(t, cols, rows, aspect):
    pts, flares, info = scene(t, cols, rows, aspect)
    char, col, dep = bodies(*pts, cols, rows, t)
    free = char == 0

    size = BUBBLE_AT[:, 4]
    ix = ((BUBBLE_AT[:, 0] + 0.012 * np.sin(t * (1 + 2 * BUBBLE_AT[:, 2]) + BUBBLE_AT[:, 3] * 40)) % 1.0 * cols).astype(int)
    iy = (((BUBBLE_AT[:, 1] - t * (0.02 + 0.06 * size)) % 1.0) * rows).astype(int)
    ok = free[iy, ix]
    tw = 0.6 + 0.4 * np.sin(t * (0.8 + 2.4 * BUBBLE_AT[:, 2]) + BUBBLE_AT[:, 3] * 40)
    char[iy[ok], ix[ok]] = BUBBLE_GLYPH[np.clip((size * 5).astype(int), 0, 4)][ok]
    col[iy[ok], ix[ok]] = BUBBLE * ((0.14 + 0.36 * size) * tw)[ok][:, None]

    lights = 0
    for p, glyph, rgb, k in motes(t) + fin_motes(t) + breath(t) + glints(t) + sparks(t):
        dots(char, col, dep, p, glyph, rgb, k, cols, rows, aspect)
        lights += int(((glyph == C("*")) & (k > 0.05)).sum())
    for t0, start, vel, ink, life, seed, centre, size in BURSTS + COLOURS.bursts:
        pop = float(np.clip(1 - (t - t0) / 0.5, 0, 1)) if t >= t0 else 0.0
        if pop > 0.05:
            flares.append((centre + [0, 0.1, 0.3], int((3 + 7 * pop) * size), pop * (0.5 + 0.5 * size), WHITE, True))

    for p, reach, strength, rgb, always in flares:
        sx, sy, z = project_cam(to_cam(np.asarray(p, float)[None]), cols, rows, aspect)
        y, x = int(sy[0]), int(sx[0])
        if 0 <= y < rows and 0 <= x < cols and (always or z[0] <= dep[y, x] + 0.08):
            rays(char, col, y, x, reach, rgb, strength)

    info.update(sparks=lights, letters=int((char > 0).sum()))
    return char, col, info


TITLE = "ASCII Betta Fish"
CW, CH, FONT_PX = 7, 10, 11
COLS, ROWS = 113, 103
W, H = COLS * CW, ROWS * CH
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
HINT = "Space pause · R restart · S screenshot · Esc quit"
DIM, ACCENT, SIGN = (100, 112, 136), (90, 220, 255), (170, 184, 214)
TRAIL = 0.50


class Renderer:
    def __init__(self, cols=COLS, rows=ROWS, cw=CW, ch=CH, font_px=FONT_PX, hud_px=15, hint=True):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.w, self.h = cols * cw, rows * ch
        font = pygame.font.SysFont(MONO, font_px)
        atlas = []
        for code in range(32, 127):
            s = pygame.Surface((cw, ch))
            g = font.render(chr(code), True, (255, 255, 255))
            s.blit(g, ((cw - g.get_width()) // 2, (ch - g.get_height()) // 2))
            atlas.append(pygame.surfarray.array3d(s).transpose(1, 0, 2))
        self.atlas = np.stack(atlas)
        self.art = pygame.Surface((self.w, self.h))
        self.trail = pygame.Surface((self.w, self.h))
        self.out = pygame.Surface((self.w, self.h))
        self.small = pygame.font.SysFont(MONO, hud_px)
        self.hud_gap = int(hud_px * 1.45)
        self.hint = hint
        self.last_t = None

    @staticmethod
    def blur(src, size, radius):
        small = pygame.transform.smoothscale(src, size)
        if hasattr(pygame.transform, "gaussian_blur"):
            small = pygame.transform.gaussian_blur(small, max(1, radius))
        return small

    def glow(self, src):
        w, h = self.w, self.h
        tight = self.blur(src, (w // 4, h // 4), 2)
        wide = self.blur(tight, (w // 8, h // 8), 3)
        for layer, amount in ((tight, 0.75), (wide, 0.45)):
            layer = pygame.transform.smoothscale(layer, (w, h))
            layer.fill((int(255 * amount),) * 3, special_flags=pygame.BLEND_MULT)
            self.out.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)

    def draw(self, t):
        w, h = self.w, self.h
        char, col, info = frame(t, self.cols, self.rows, w / h)
        pygame.surfarray.blit_array(self.art, self.atlas[char].transpose(1, 3, 0, 2, 4).reshape(w, h, 3))
        tint = pygame.surfarray.make_surface((np.clip(col, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2))
        self.art.blit(pygame.transform.scale(tint, (w, h)), (0, 0), special_flags=pygame.BLEND_MULT)
        if self.last_t is None or t < self.last_t:
            self.trail.fill((0, 0, 0))
        self.last_t = t
        self.trail.fill((int(TRAIL * 255),) * 3, special_flags=pygame.BLEND_MULT)
        self.trail.blit(self.art, (0, 0), special_flags=pygame.BLEND_MAX)
        self.out.fill((0, 0, 0))
        self.out.blit(self.trail, (0, 0))
        self.glow(self.out)
        return self.out


def fit_to(sw, sh):
    k = min(sw / W, sh / H)
    cw, ch = max(3, int(CW * k)), max(4, int(CH * k))
    font_px = max(6, round(FONT_PX * ch / CH))
    w, h = COLS * cw, ROWS * ch
    return (cw, ch, font_px), ((sw - w) // 2, (sh - h) // 2)


SONG = Path(__file__).with_name("amanai.mp3")


def start_song():
    try:
        pygame.mixer.music.load(str(SONG))
        pygame.mixer.music.play(loops=-1)
        return True
    except pygame.error as err:
        print(f"no music: {err}", file=sys.stderr)
        return False


def main():
    pygame.init()
    music = start_song()
    window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption(TITLE)
    pygame.mouse.set_visible(False)
    (cw, ch, font_px), at = fit_to(*window.get_size())
    renderer, clock = Renderer(cw=cw, ch=ch, font_px=font_px), pygame.time.Clock()
    t, paused = 0.0, False
    screen = renderer.draw(t)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return 0
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
                if music:
                    pygame.mixer.music.pause() if paused else pygame.mixer.music.unpause()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t = 0.0
                COLOURS.reset()
                if music:
                    pygame.mixer.music.play(loops=-1)
                    if paused:
                        pygame.mixer.music.pause()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                COLOURS.ask(t)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                pygame.image.save(screen, str(Path(__file__).with_name(f"betta_fish_{int(time.time())}.png")))
        dt = clock.tick(30) / 1000
        if not paused:
            t += min(dt, 1 / 15)
            if t > LOOP:
                t = 0.0
                COLOURS.reset()
            screen = renderer.draw(t)
        window.fill((0, 0, 0))
        window.blit(screen, at)
        pygame.display.flip()


if __name__ == "__main__":
    sys.exit(main())