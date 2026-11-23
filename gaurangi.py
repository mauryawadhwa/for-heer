import sys
import time
from pathlib import Path

import numpy as np
import pygame

MESSAGE = "F O R   Y O U   <3"

TAU = np.pi * 2
LOOP = 20.0

T_GROW = (0.5, 3.0)
T_BUDS = (2.5, 3.6)
T_PAPER = (2.9, 4.1)
T_RIBBON = 4.25
T_BLOSSOM = (4.4, 5.1)
GROWN = T_BLOSSOM[1]
T_OPEN = 5.0
STAGGER = 0.38
OPEN = 2.6
T_GLEAM = (14.6, 15.8)
T_FALL = 15.2
T_MSG = 13.0
FLASH = 0.6

POOLS = ["1il!;:", "rtfjcvxz=+", "aeoun0{}[]<>", "08B$#&%Q"]
C = lambda ch: ord(ch) - 32
FILL = np.array([[C(p[k % len(p)]) for k in range(12)] for p in POOLS])

(STEM, LEAF, SEPAL, PETAL, EDGE, GLINT, PAPER, PAPER_IN, RIBBON, DUST, GOLD, FOLIAGE, BLOSSOM,
 STAR_PINK, STAR_BLUE) = range(15)
COLORS = np.array([[70, 200, 96], [52, 196, 84], [80, 206, 100], [255, 34, 64], [255, 120, 146], [255, 255, 255],
                   [212, 220, 236], [128, 136, 156], [255, 190, 80], [220, 228, 245], [255, 214, 150],
                   [34, 130, 62], [255, 244, 248], [255, 150, 190], [170, 205, 255]], np.float32) / 255
WHITE = np.array([1.0, 1.0, 1.0], np.float32)
SHADES = np.array([
    [[255, 22, 52], [255, 90, 112]],
    [[255, 70, 140], [255, 150, 190]],
    [[255, 34, 64], [255, 110, 130]],
    [[255, 120, 176], [255, 186, 214]],
    [[255, 50, 76], [255, 126, 144]],
], np.float32) / 255
ROSE_COLOURS = SHADES[[0] + [1, 2] * 3 + [3, 4, 0] * 4]
N_ROSES = len(ROSE_COLOURS)
PAPER_LIGHT, PAPER_IN_LIGHT = 0.66, 0.40


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def normals_of(p, axis_u, axis_v, outward):
    n = np.cross(np.gradient(p, axis=axis_u), np.gradient(p, axis=axis_v))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n * np.sign((n * outward).sum(-1, keepdims=True) + 1e-9)


def turn_to(d):
    y = np.array([0.0, 1.0, 0.0])
    d = d / np.linalg.norm(d)
    v, c = np.cross(y, d), y @ d
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def spin_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


SPIRAL, ACROSS = 300, 10
ANGLE, ALONG = np.meshgrid(np.linspace(2.2 * np.pi, 20 * np.pi, SPIRAL), np.linspace(0, 1, ACROSS))
INWARD = (ANGLE - ANGLE.min()) / (ANGLE.max() - ANGLE.min())
PETAL_ID = (np.floor(3.6 * ANGLE / TAU).astype(int) - int(np.floor(3.6 * ANGLE.min() / TAU))).ravel()
OUTER = 1 - INWARD.ravel()
STEPS = 50
_PETALS = {}


def petals(open_):
    key = round(float(open_) * STEPS) / STEPS
    if key not in _PETALS:
        spread = 1.4 + 4.2 * key
        lean = (np.pi / 2) * np.exp(-ANGLE / (spread * np.pi))
        width = 1 - 0.5 * (1.25 * (1 - np.mod(3.6 * ANGLE, TAU) / np.pi) ** 2 - 0.25) ** 2
        cup = 1.95653 * ALONG ** 2 * (1.27689 * ALONG - 1) ** 2 * np.sin(lean)
        r = width * (ALONG * np.sin(lean) + cup * np.cos(lean))
        y = width * (ALONG * np.cos(lean) - cup * np.sin(lean))
        p = np.stack([r * np.sin(ANGLE), y, r * np.cos(ANGLE)], -1) * 0.8
        wrap = smooth(key / 0.7)
        p[..., [0, 2]] *= 0.72 + 0.28 * wrap
        p[..., 1] *= 0.66 + 0.34 * wrap
        out = np.stack([np.sin(ANGLE), np.full_like(ANGLE, 0.3), np.cos(ANGLE)], -1)
        n = normals_of(p, 1, 0, out)
        _PETALS[key] = (p.reshape(-1, 3).astype(np.float32), n.reshape(-1, 3).astype(np.float32), (ALONG > 0.92).ravel())
    return _PETALS[key]


N_SEPAL, SEP_LEN, SEP_ACROSS = 5, 10, 3
_sl, _sw = np.meshgrid(np.linspace(0, 1, SEP_LEN), np.linspace(-1, 1, SEP_ACROSS))


_SEPALS = {}


def sepals(open_):
    open_ = round(float(open_) * 10) / 10
    if open_ in _SEPALS:
        return _SEPALS[open_]
    P, N = [], []
    reach, droop = 0.24 + 0.06 * open_, 0.10 + 0.14 * open_
    for i in range(N_SEPAL):
        a = TAU * i / N_SEPAL + 0.6
        out = np.array([np.sin(a), 0.0, np.cos(a)])
        side = np.array([np.cos(a), 0.0, -np.sin(a)])
        b0, b1, b2 = np.array([0, 0.03, 0]) + 0.03 * out, 0.16 * out + [0, 0.10 - 0.10 * open_, 0], reach * out - [0, droop, 0]
        u = _sl[..., None]
        mid = (1 - u) ** 2 * b0 + 2 * (1 - u) * u * b1 + u ** 2 * b2
        half = 0.05 * np.sin(np.pi * np.minimum(_sl * 1.5, 1) / 2) * (1 - _sl) ** 0.7
        pts = mid + side * (half * _sw)[..., None]
        P.append(pts.reshape(-1, 3))
        N.append(normals_of(pts, 1, 0, np.array([0.0, 1.0, 0.0])).reshape(-1, 3))
    _SEPALS[open_] = (np.concatenate(P).astype(np.float32), np.concatenate(N).astype(np.float32))
    return _SEPALS[open_]


ROSE_SIZE = 0.30
WAIST = np.array([0.0, -0.62, 0.0])
BOTTOM_Y = -1.10
DOME, DOME_R = np.array([0.0, -0.38, 0.0]), 0.64
RING_TILT = 0.47


def on_dome(tilt, around):
    return np.array([np.sin(tilt) * np.sin(around), np.cos(tilt), np.sin(tilt) * np.cos(around)])


def layout():
    lean = [on_dome(0.0, 0.0)]
    lean += [on_dome(RING_TILT, TAU * k / 6 + TAU / 12) for k in range(6)]
    lean += [on_dome(2 * RING_TILT, TAU * k / 12) for k in range(12)]
    pos = [DOME + DOME_R * d for d in lean]
    rng = np.random.default_rng(5)
    turns = [turn_to(d) @ spin_y(rng.uniform(0, TAU)) for d in lean]
    order = [0] + [1, 4, 2, 5, 3, 6] + [7 + (5 * j) % 12 for j in range(12)]
    starts = np.empty(N_ROSES)
    starts[order] = T_OPEN + STAGGER * np.arange(N_ROSES)
    return pos, lean, turns, starts


ROSE_POS, ROSE_DIR, ROSE_TURN, ROSE_START = layout()
ROSE_POS = [p.astype(np.float32) for p in ROSE_POS]
ROSE_TURN = [R.astype(np.float32) for R in ROSE_TURN]


STEM_TOP = max(float(q[1]) for q in ROSE_POS)


def grow_time(y):
    return T_GROW[0] + (T_GROW[1] - T_GROW[0]) * np.clip((y - BOTTOM_Y) / (STEM_TOP - BOTTOM_Y), 0, 1)


_bud = np.random.default_rng(23).random(N_ROSES)
BUD_POP = np.array([T_BUDS[0] + (T_BUDS[1] - T_BUDS[0] - 0.35) * (float(q[1]) - (-0.01)) / (STEM_TOP + 0.01)
                    for q in ROSE_POS]) + 0.35 * _bud


def opening(t):
    e = smooth((t - ROSE_START) / OPEN)
    return e * (1.35 - 0.35 * e)


def tube(curve, radius, around=10):
    d = np.gradient(curve, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    side = np.cross(d, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
    up = np.cross(side, d)
    a = np.linspace(0, TAU, around, endpoint=False)
    n = np.cos(a)[None, :, None] * side[:, None] + np.sin(a)[None, :, None] * up[:, None]
    return (curve[:, None] + radius * n).reshape(-1, 3), n.reshape(-1, 3)


def stems():
    P, N = [], []
    for k in range(9):
        spot = np.array([0.026 * np.sin(1.7 * k), 0.0, 0.026 * np.cos(1.7 * k)]) * (k > 0)
        top = WAIST + spot + [0.0, 0.10, 0.0]
        end = WAIST + spot * 2.4 + [0.0, BOTTOM_Y - WAIST[1], 0.0]
        p_, n_ = tube(top + (end - top) * np.linspace(0, 1, 40)[:, None], 0.013, 6)
        P.append(p_)
        N.append(n_)
    for k in range(N_ROSES):
        base = ROSE_POS[k] - ROSE_DIR[k] * 0.02
        start = WAIST + [0.0, 0.10, 0.0] + np.array([0.026 * np.sin(1.7 * k), 0.0, 0.026 * np.cos(1.7 * k)])
        bend = np.array([0.35 * base[0], 0.5 * (start[1] + base[1]), 0.35 * base[2]])
        u = np.linspace(0, 1, 30)[:, None]
        p_, n_ = tube((1 - u) ** 2 * start + 2 * (1 - u) * u * bend + u ** 2 * base, 0.011, 5)
        P.append(p_)
        N.append(n_)
    return np.concatenate(P), np.concatenate(N)


def foliage():
    tilt, a = np.meshgrid(np.linspace(0, 1.25, 20), np.linspace(0, TAU, 90, endpoint=False))
    r = 0.60 + 0.012 * np.sin(7 * a + 14 * tilt)
    p = DOME + r[..., None] * np.stack([np.sin(tilt) * np.sin(a), np.cos(tilt), np.sin(tilt) * np.cos(a)], -1)
    start = grow_time(0.0)
    appear = start + (T_GROW[1] - start) * tilt / 1.25
    return p.reshape(-1, 3), normals_of(p, 1, 0, np.array([0.0, 1.0, 0.0])).reshape(-1, 3), appear.ravel()


N_BLOSSOM = 110
_bl = np.random.default_rng(41).random((N_BLOSSOM, 3))


def blossoms():
    tilt, a = 2 * RING_TILT + 0.12 + 0.30 * _bl[:, 0], _bl[:, 1] * TAU
    p = DOME + (DOME_R + 0.10) * np.stack([np.sin(tilt) * np.sin(a), np.cos(tilt), np.sin(tilt) * np.cos(a)], 1)
    return p.astype(np.float32), _bl[:, 2]


BLOSSOM_P, BLOSSOM_PHASE = blossoms()


def leaf(base, d, length):
    s, w = np.meshgrid(np.linspace(0, 1, 34), np.linspace(-1, 1, 11))
    d = d / np.linalg.norm(d)
    wide = np.cross(d, [0.0, 1.0, 0.0])
    wide /= np.linalg.norm(wide)
    up = np.cross(wide, d)
    half = 0.30 * length * np.sqrt(s) * (1 - s) ** 0.8 * 1.9 * (1 + 0.06 * np.sin(s * 40))
    crease = 0.10 * length * w ** 2
    curl = -0.28 * length * s ** 1.8
    p = base + d * (s * length)[..., None] + wide * (w * half)[..., None] + up * (crease + curl)[..., None]
    return p.reshape(-1, 3), normals_of(p, 1, 0, up).reshape(-1, 3), s.ravel()


def leaves():
    P, N, part, appear = [], [], [], []
    for j in range(1, 7):
        a = TAU * j / 8 + TAU / 16
        base = np.array([0.60 * np.sin(a), -0.17, 0.60 * np.cos(a)])
        d = np.array([np.sin(a), 0.05, np.cos(a)])
        p, n, along = leaf(base, d, 0.30 + 0.03 * np.sin(5 * a))
        P.append(p)
        N.append(n)
        part.append(np.full(len(p), 20 + j))
        appear.append(grow_time(base[1]) + 0.12 * j + 0.5 * along)
    return np.concatenate(P), np.concatenate(N), np.concatenate(part), np.concatenate(appear)


def paper():
    a, v = np.meshgrid(np.linspace(0, TAU, 150, endpoint=False), np.linspace(0, 1, 28))
    top = -0.15 + 0.60 * (1 - np.cos(a)) / 2 + 0.03 * np.cos(9 * a)
    r = (0.075 + (0.645 + 0.20 * (1 - np.cos(a)) / 2) * v ** 0.75) * (1 + 0.07 * v * np.sin(9 * a))
    y = WAIST[1] + v * (top - WAIST[1])
    p = np.stack([r * np.sin(a), y, r * np.cos(a)], -1)
    n = normals_of(p, 1, 0, np.stack([np.sin(a), np.zeros_like(a), np.cos(a)], -1))
    rim = v > 0.965
    wrap = (1 + np.cos(a)) / 2
    appear = T_PAPER[0] + (T_PAPER[1] - T_PAPER[0]) * (0.6 * wrap + 0.4 * v)
    return p.reshape(-1, 3), n.reshape(-1, 3), rim.ravel(), appear.ravel()


def ribbon():
    P, N, part = [], [], []

    def add(p, n, k):
        P.append(p.reshape(-1, 3))
        N.append(n.reshape(-1, 3))
        part.append(np.full(len(P[-1]), k))

    a, w = np.meshgrid(np.linspace(0, TAU, 120, endpoint=False), np.linspace(-1, 1, 6))
    band = np.stack([0.09 * np.sin(a), WAIST[1] + 0.022 * w, 0.09 * np.cos(a)], -1)
    add(band, np.stack([np.sin(a), 0 * a, np.cos(a)], -1), 70)
    knot = WAIST + [0.0, 0.0, 0.10]
    for side, k in ((-1, 71), (1, 72)):
        u, q = np.meshgrid(np.linspace(0, TAU, 80), np.linspace(0, 1, 9))
        rx, ry = 0.03 + 0.055 * q, 0.010 + 0.036 * q
        x, y = side * (0.02 + rx + rx * np.cos(u + np.pi)), ry * np.sin(u)
        tip = side * 0.30
        x, y = x * np.cos(tip) - y * np.sin(tip), x * np.sin(tip) + y * np.cos(tip)
        z = 0.03 * np.sin(u / 2) ** 2 + 0.015 * q
        p = knot + np.stack([x, y, z], -1)
        add(p, normals_of(p, 1, 0, np.array([0.0, 0.3, 1.0])), k)
    for side, k in ((-1, 73), (1, 74)):
        s, w = np.meshgrid(np.linspace(0, 1, 40), np.linspace(-1, 1, 7))
        keep = s < 1 - 0.10 * (1 - np.abs(w))
        centre = np.stack([side * (0.03 + 0.08 * s), -0.02 - 0.22 * s, 0.03 + 0.03 * np.sin(3 * s)], -1)
        across = np.stack([0.030 * np.cos(0.4 * side) + 0 * s, 0.030 * np.sin(0.4 * side) + 0 * s, 0 * s], -1)
        p = knot + centre + across * w[..., None]
        n = normals_of(p, 1, 0, np.array([0.0, 0.0, 1.0]))
        add(p[keep], n[keep], k)
    u, q = np.meshgrid(np.linspace(0, TAU, 40, endpoint=False), np.linspace(0, 1, 6))
    p = knot + np.stack([0.032 * q * np.cos(u), 0.028 * q * np.sin(u), 0.035 - 0.02 * q ** 2], -1)
    add(p, normals_of(p, 1, 0, np.array([0.0, 0.0, 1.0])), 75)
    return np.concatenate(P), np.concatenate(N), np.concatenate(part)


class Bouquet:
    def __init__(self):
        sp, sn = stems()
        fp, fn, f_appear = foliage()
        lp, ln, lpart, l_appear = leaves()
        pp, pn, prim, p_appear = paper()
        rp, rn, rpart = ribbon()
        self.p = np.concatenate([sp, fp, lp, pp, rp]).astype(np.float32)
        self.n = np.concatenate([sn, fn, ln, pn, rn]).astype(np.float32)
        self.tag = np.concatenate([np.full(len(sp), STEM), np.full(len(fp), FOLIAGE), np.full(len(lp), LEAF),
                                   np.full(len(pp), PAPER), np.full(len(rp), RIBBON)])
        self.part = np.concatenate([np.full(len(sp), 1), np.full(len(fp), 50), lpart, np.full(len(pp), 60), rpart])
        self.rim = np.concatenate([np.zeros(len(sp) + len(fp) + len(lp), bool), prim, np.zeros(len(rp), bool)])
        self.ribbon = self.tag == RIBBON
        self.appear = np.concatenate([grow_time(sp[:, 1]), f_appear, l_appear, p_appear,
                                      np.full(len(rp), T_RIBBON)]).astype(np.float32)
        self.pop = np.concatenate([np.zeros(len(sp) + len(fp) + len(lp) + len(pp), bool), np.ones(len(rp), bool)])
        knot = WAIST + [0.0, 0.0, 0.10]
        self.centre = self.p.copy()
        self.centre[self.pop] = np.where((rpart == 70)[:, None], WAIST, knot)


BOUQUET = Bouquet()
ROSE_APPEAR = None
BLOSSOM_APPEAR = T_BLOSSOM[0] + (T_BLOSSOM[1] - T_BLOSSOM[0]) * np.random.default_rng(43).random(N_BLOSSOM)


N_FALL = 18
FALL = np.random.default_rng(17).random((N_FALL, 8))
FALL_PERIOD = 5.0
FALL_POSES = [("(", ")"), ("/", ""), ("-", ""), ("\\", ""), ("{", "}"), ("<", ">")]
FALL_TURN = [0, 1, 2, 3, 4, 3, 2, 1]
FALL_FROM = [k for k in range(N_ROSES) if ROSE_DIR[k][2] > -0.3]


def falling(t):
    out = []
    for k, (a, b, c, d, e, f, g, h) in enumerate(FALL):
        since = t - T_FALL - k * FALL_PERIOD / N_FALL
        if since < 0:
            continue
        age = since % FALL_PERIOD
        rose = FALL_FROM[int(a * len(FALL_FROM))]
        start = ROSE_POS[rose] + ROSE_TURN[rose] @ (np.array([np.cos(b * TAU), 0.10, np.sin(b * TAU)]) * ROSE_SIZE * 0.35)
        drift = np.array([0.20 * (c - 0.5) * age + 0.06 * np.sin(age * (2 + 2 * d) + e * TAU),
                          -0.30 * age, 0.25 * age * f + 0.05 * np.cos(age * (2 + g) + h * TAU)])
        pose = FALL_TURN[int(age * (3 + 3 * d) + e * 8) % len(FALL_TURN)]
        fade = smooth(age / 0.3) * smooth((FALL_PERIOD - age) / 0.6)
        out.append((start + drift, rose, pose, fade))
    return out


POP = np.random.default_rng(13).random((N_ROSES, 14, 4))
N_STARS = 460
STARS = np.random.default_rng(21).random((N_STARS, 6))
SIDE_STARS = np.random.default_rng(22).random((3 * N_STARS, 6))
STAR_FLARES = 6


def pops(t, op):
    P, S, T = [], [], []
    for k in range(N_ROSES):
        age = t - (ROSE_START[k] + 0.85 * OPEN)
        if not 0 < age < 1.4:
            continue
        g = POP[k]
        a, up = g[:, 0] * TAU, 0.4 + 0.8 * g[:, 1]
        speed = 0.45 + 0.5 * g[:, 2]
        v = np.stack([np.cos(a) * speed, up, np.sin(a) * speed], 1)
        head = ROSE_POS[k] + ROSE_DIR[k] * 0.12
        P.append(head + v * age - np.array([0, 0.9, 0]) * age ** 2)
        S.append((1 - age / 1.4) ** 1.2 * (0.8 + 0.5 * g[:, 3]))
        T.append(np.where(g[:, 3] > 0.5, GLINT, GOLD))
    if not P:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(S), np.concatenate(T)


def stars(t, spread=1.0):
    g = np.concatenate([STARS, SIDE_STARS[:int(N_STARS * (spread - 1))]])
    x = 2.5 * g[:, 0] - 1.25
    side = np.arange(len(g)) >= N_STARS
    u = g[side, 0]
    x[side] = np.where(u < 0.5, -1, 1) * 1.25 * (1 + (spread - 1) * (2 * u % 1.0))
    x = x + 0.04 * np.sin(t * 0.5 + g[:, 5] * TAU)
    y = -1.75 + 3.1 * ((g[:, 1] + t * 0.006 * (0.5 + g[:, 3])) % 1.0)
    z = -1.3 + 2.2 * g[:, 2]
    beat = 2.5 + 4.5 * g[:, 3]
    peak = np.clip(1 - np.abs((t / beat + g[:, 4]) % 1.0 - 0.5) * beat / 0.24, 0, 1)
    glow = 0.30 + 0.35 * g[:, 5] + peak ** 1.5
    kind = np.select([g[:, 2] % 0.25 < 0.045, g[:, 2] % 0.25 < 0.08, g[:, 2] % 0.25 < 0.105], [STAR_PINK, STAR_BLUE, GOLD], GLINT)
    return np.stack([x, y, z], 1), glow, kind


def lights(t):
    key = np.array([0.45, 0.80, 0.55])
    sheen = np.array([0.9 * np.sin(t * 0.5), 0.35, 0.6])
    return key / np.linalg.norm(key), sheen / np.linalg.norm(sheen)


def sway(t):
    return 0.38 * np.sin(TAU * t / LOOP) * smooth(t / 1.5)


EYE, TARGET = np.array([0.0, 1.75, 4.0]), np.array([0.0, -0.28, 0.0])
_FWD = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
_RIGHT = np.cross(_FWD, [0, 1.0, 0])
_RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_RIGHT, _FWD)
VIEW = (-_FWD).astype(np.float32)
_RIGHT, _UP, _FWD = (v.astype(np.float32) for v in (_RIGHT, _UP, _FWD))
EYE32 = EYE.astype(np.float32)


def look(P):
    rel = P - EYE32
    return rel @ _RIGHT, rel @ _UP, rel @ _FWD


_ROSES, _ROSE_LOOKS = {}, {}


def rose(k, open_, flash):
    key = (open_, flash)
    if k not in _ROSES or _ROSES[k][0] != key:
        p, n, rim = petals(open_)
        sp, sn = sepals(open_)
        if k not in _ROSE_LOOKS:
            _ROSE_LOOKS[k] = (
                np.concatenate([np.where(rim, EDGE, PETAL), np.full(len(sp), SEPAL)]),
                np.concatenate([np.where(rim[:, None], ROSE_COLOURS[k, 1], ROSE_COLOURS[k, 0]),
                                np.broadcast_to(COLORS[SEPAL], (len(sp), 3))]),
                np.concatenate([1000 + 100 * k + PETAL_ID, np.full(len(sp), 40 + k)]))
        size = np.float32(ROSE_SIZE * (1 + 0.12 * flash))
        R = ROSE_TURN[k]
        _ROSES[k] = (key, (ROSE_POS[k] + (np.concatenate([p, sp]) * size) @ R.T, np.concatenate([n, sn]) @ R.T)
                     + _ROSE_LOOKS[k])
    return _ROSES[k][1]


_BUILT = {}


def assemble(states):
    if _BUILT.get("key") != states:
        b = BOUQUET
        P, N, T, K, tone = [b.p], [b.n], [b.tag], [b.part], [COLORS[b.tag]]
        W, gain, appear = [np.zeros(len(b.p), np.float32)], [np.where(b.tag == FOLIAGE, 0.55, 1.0)], [b.appear]
        pop, centre = [b.pop], [b.centre]
        at, n = {}, len(b.p)
        for k, open_, flash in states:
            rp, rn, rt, rtone, rk = rose(k, open_, flash)
            at[k], n = n, n + len(rp)
            P.append(rp)
            N.append(rn)
            T.append(rt)
            K.append(rk)
            tone.append(rtone)
            W.append(np.where(rt == SEPAL, 0.0, 0.75 * flash))
            gain.append(np.concatenate([HEART, np.ones(len(rp) - len(HEART))]))
            appear.append(ROSE_APPEAR[k])
            pop.append(np.ones(len(rp), bool))
            centre.append(np.broadcast_to(ROSE_POS[k], rp.shape))
        v = {"P": np.concatenate(P), "N": np.concatenate(N), "T": np.concatenate(T), "K": np.concatenate(K),
             "tone": np.concatenate(tone).astype(np.float32), "W": np.concatenate(W).astype(np.float32),
             "gain": np.concatenate(gain).astype(np.float32), "appear": np.concatenate(appear), "at": at,
             "pop": np.concatenate(pop), "centre": np.concatenate(centre).astype(np.float32)}
        v["lit"] = np.flatnonzero(v["W"] > 0)
        v["gloss"] = np.where(v["T"] == RIBBON, 0.9, 0.3).astype(np.float32)
        v["paper"] = np.flatnonzero(v["T"] == PAPER)
        v["paper_rim"] = b.rim[v["paper"]]
        v["ribbon"] = np.flatnonzero(v["T"] == RIBBON)
        _BUILT.update(key=states, value=v)
    return _BUILT["value"]


def scene(t, spread=1.0):
    global ROSE_APPEAR, BLOSSOM_APPEAR
    if ROSE_APPEAR is None:
        ROSE_APPEAR = [np.full(len(rose(k, 0.0, 0.0)[0]), BUD_POP[k], np.float32) for k in range(N_ROSES)]
    op = opening(t)
    yaw = sway(t)
    Ry = spin_y(yaw)

    states, heads = [], []
    for k in range(N_ROSES):
        if (Ry @ ROSE_DIR[k]) @ VIEW < -0.3:
            continue
        flash = np.clip(1 - (t - ROSE_START[k]) / FLASH, 0, 1) ** 2 if t >= ROSE_START[k] else 0.0
        states.append((k, round(float(op[k]) * STEPS) / STEPS, round(float(flash), 1)))
        heads.append(((ROSE_POS[k] + ROSE_DIR[k] * 0.10 * (0.5 + op[k])) @ Ry.T, flash))
    v = assemble(tuple(states))
    P = v["P"]
    growing = t < GROWN + FLASH
    if growing:
        age = t - v["appear"]
        pf = np.flatnonzero(v["pop"] & (age >= 0) & (age < FLASH))
        if len(pf):
            g = smooth(age[pf] / 0.2) * (1 + 0.35 * np.clip(1 - age[pf] / FLASH, 0, 1) ** 2)
            P = P.copy()
            P[pf] = v["centre"][pf] + (P[pf] - v["centre"][pf]) * g[:, None].astype(np.float32)
    P, N, K = P @ Ry.T, v["N"] @ Ry.T, v["K"]
    T, rgb = v["T"].copy(), v["tone"].copy()

    key, sheen = (x.astype(np.float32) for x in lights(t))
    half_k = ((key + VIEW) / np.linalg.norm(key + VIEW)).astype(np.float32)
    half_s = ((sheen + VIEW) / np.linalg.norm(sheen + VIEW)).astype(np.float32)
    back = N @ VIEW < 0
    side = np.where(back, np.float32(-1), np.float32(1))
    diff = np.clip(side * (N @ key), 0, 1)
    a2 = np.clip(side * (N @ half_k), 0, 1) ** 2
    b2 = np.clip(side * (N @ half_s), 0, 1) ** 2
    a8, b8 = (a2 * a2) ** 2, (b2 * b2) ** 2
    shade = (0.62 + 0.60 * diff + v["gloss"] * (a8 * a8 * a8 + 0.7 * b8 * b8 * b2)) * v["gain"]
    pp = v["paper"]
    shade[pp] = 0.40 + 0.45 * diff[pp] + 1.1 * (a2[pp] ** 3 + b2[pp] ** 3)
    inside = pp[back[pp] & ~v["paper_rim"]]
    T[inside], rgb[inside] = PAPER_IN, COLORS[PAPER_IN]

    if T_GLEAM[0] < t < T_GLEAM[1] + 0.3:
        rb = v["ribbon"]
        at = -0.35 + 0.70 * (t - T_GLEAM[0]) / (T_GLEAM[1] - T_GLEAM[0])
        g = np.exp(-((look(P[rb])[0] - at) / 0.05) ** 2)
        shade[rb] += 1.45 * g
        rgb[rb] += (WHITE - rgb[rb]) * (0.7 * g)[:, None]

    for k, start in v["at"].items():
        if op[k] < 0.85:
            continue
        idx = start + DEW_AT
        beat = np.clip(1 - np.abs((t + DEW_OFF + 0.7 * k) % DEW_EVERY - 0.25) / 0.25, 0, 1)
        on = idx[beat > 0.3]
        T[on], rgb[on] = GLINT, WHITE
        shade[idx] = np.maximum(shade[idx], 0.6 + 0.7 * beat)

    lit = v["lit"]
    if len(lit):
        w = v["W"][lit]
        rgb[lit] += (WHITE - rgb[lit]) * w[:, None]
        shade[lit] += 0.8 * w

    if growing:
        fresh = ~v["pop"] & (age >= 0) & (age < 0.14)
        T[fresh], shade[fresh], rgb[fresh] = GLINT, 1.2, WHITE
        hot = v["pop"] & (age >= 0) & (age < FLASH)
        f = (np.clip(1 - age[hot] / FLASH, 0, 1) ** 2).astype(np.float32)
        rgb[hot] += (WHITE - rgb[hot]) * f[:, None]
        shade[hot] += 1.3 * f
        seen = age >= 0
        P, shade, rgb, T, K = P[seen], shade[seen], rgb[seen], T[seen], K[seen]

    fall = falling(t)
    gp, gs, gt = stars(t, spread)
    sp, ss, st = pops(t, op)
    lit_up = np.clip(1 - (t - BLOSSOM_APPEAR) / FLASH, 0, 1) ** 2 * (t >= BLOSSOM_APPEAR)
    bs = (0.55 + 0.35 * np.clip(np.sin(t * 1.3 + BLOSSOM_PHASE * 40), 0, 1) ** 4 + 0.9 * lit_up) * (t >= BLOSSOM_APPEAR)
    extra = np.concatenate([np.concatenate([BLOSSOM_P, sp]) @ Ry.T, gp])
    info = {"open": int((op > 0.98).sum()), "falling": len(fall), "deg": float(np.degrees(yaw)),
            "grown": float(np.clip((t - T_GROW[0]) / (GROWN - T_GROW[0]), 0, 1))}
    fall = [(pos @ Ry.T, rose_k, pose, fade) for pos, rose_k, pose, fade in fall]
    fx_shade = np.concatenate([bs, ss, gs])
    fx_tag = np.concatenate([np.full(N_BLOSSOM, BLOSSOM), st, gt])
    return (P, shade, rgb, T, K), (extra, fx_shade, fx_tag), heads, fall, info


N_DEW = 14
_dew = np.random.default_rng(31)
DEW_AT = np.flatnonzero((OUTER > 0.6) & (ALONG.ravel() > 0.6))[_dew.integers(0, 300, N_DEW)]
DEW_OFF, DEW_EVERY = _dew.uniform(0, 10, N_DEW), _dew.uniform(2.0, 4.0, N_DEW)
HEART = (0.66 + 0.34 * OUTER ** 0.8).astype(np.float32)


def outline_letter(gx, gy):
    ax, ay = np.abs(gx), np.abs(gy)
    c = np.where(ax > 2 * ay, C("|"), np.where(ay > 2 * ax, np.where(gy > 0, C("_"), C("-")),
                 np.where(gx * gy > 0, C("/"), C("\\"))))
    return np.where((ax + ay) == 0, C("+"), c)


def rays(char, col, cy, cx, reach, rgb, strength=1.0, over=1.0):
    rows, cols = char.shape
    for dist in range(1, reach + 1):
        fade = (1 - dist / (reach + 1)) * strength
        arms = [(0, 1, "-"), (0, -1, "-"), (1, 0, "|"), (-1, 0, "|")]
        if dist <= reach // 2:
            arms += [(1, 1, "\\"), (-1, -1, "\\"), (1, -1, "/"), (-1, 1, "/")]
        for dy, dx, stroke in arms:
            yy, xx = cy + dy * dist, cx + dx * dist * (2 if dy == 0 else 1)
            k = fade * (0.7 if dx and dy else 1.0)
            if 0 <= yy < rows and 0 <= xx < cols and col[yy, xx].max() < k * over:
                char[yy, xx], col[yy, xx] = C(stroke), rgb * k
    if 0 <= cy < rows and 0 <= cx < cols:
        char[cy, cx], col[cy, cx] = C("*"), rgb * 1.3 * strength


FOCAL = 3.3
MIDDLE = 0.50
FOLD = 0.12


def frame(t, cols, rows, cw, ch):
    aspect = (cols * cw) / (rows * ch)
    (P, shade, rgb, T, K), (fx_p, fx_s, fx_t), heads, fall, info = scene(t, max(1.0, aspect / (W / H)))
    x, y, z = look(P)
    zz = np.maximum(z, 0.2)
    sx = (0.5 + x / zz * FOCAL * 0.5 / aspect) * cols
    sy = (MIDDLE - y / zz * FOCAL * 0.5) * rows
    order = np.argsort(-z)
    ix, iy = sx[order].astype(int), sy[order].astype(int)
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows) & (z[order] > 0.2)
    o = order[ok]
    ix, iy = ix[ok], iy[ok]

    part = np.full((rows, cols), -1)
    depth = np.full((rows, cols), 99.0)
    lum = np.zeros((rows, cols))
    tag = np.zeros((rows, cols), int)
    tint = np.zeros((rows, cols, 3), np.float32)
    part[iy, ix], depth[iy, ix], lum[iy, ix], tag[iy, ix], tint[iy, ix] = K[o], z[o], shade[o], T[o], rgb[o]
    body = part >= 0

    pp = np.pad(part, 1, constant_values=-1)
    pd = np.pad(depth, 1, constant_values=99)
    edge = np.zeros_like(body)
    gx, gy = np.zeros((rows, cols)), np.zeros((rows, cols))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = pp[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            nz = pd[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            other_rose = (nb // 100 != part // 100) & (nb >= 1000) & (part >= 1000) & (nz >= depth)
            out = body & ((nb < 0) | ((nb != part) & (nz > depth + FOLD)) | other_rose)
            gx += dx * out
            gy += dy * out
            if dx == 0 or dy == 0:
                edge |= out

    r, c = np.mgrid[0:rows, 0:cols]
    h = (r * 7919 + c * 104729) % 9973
    level = np.clip((lum * 3.2).astype(int), 0, 3)
    inside = FILL[level, (h + (t * 4 + h % 11).astype(int)) % 12]
    sparkle = body & (tag == GLINT)
    char = np.where(body, np.where(edge, outline_letter(gx, gy), inside), 0)
    char = np.where(sparkle, C("*"), char)
    light = np.where(body, np.where(edge, np.maximum(0.85, lum), np.clip(0.45 + 0.55 * lum / 0.9, 0.45, 1.25)), 0)
    light = np.where(tag == PAPER, light * PAPER_LIGHT, np.where(tag == PAPER_IN, light * PAPER_IN_LIGHT, light))
    col = tint * light[..., None]
    steady = body.copy()

    if len(fx_p):
        px, py, pz = look(fx_p)
        qx = ((0.5 + px / np.maximum(pz, 0.2) * FOCAL * 0.5 / aspect) * cols).astype(int)
        qy = ((MIDDLE - py / np.maximum(pz, 0.2) * FOCAL * 0.5) * rows).astype(int)
        m = (qx >= 0) & (qx < cols) & (qy >= 0) & (qy < rows) & (fx_s > 0.04) & (pz > 0.2)
        far = (fx_t == BLOSSOM) & (body[np.clip(qy, 0, rows - 1), np.clip(qx, 0, cols - 1)]
                                   & (pz > depth[np.clip(qy, 0, rows - 1), np.clip(qx, 0, cols - 1)] + 0.01))
        m &= ~far
        qx, qy, pz, s, k = qx[m], qy[m], pz[m], fx_s[m], fx_t[m]
        front = ~body[qy, qx] | ((pz < depth[qy, qx] - 0.05) & ((k == BLOSSOM) | (s > 0.6)))
        qx, qy, s, k = qx[front], qy[front], s[front], k[front]
        bloss = k == BLOSSOM
        char[qy, qx] = np.where(bloss, np.where(s > 0.8, C("*"), C("o")),
                                np.where(s > 0.7, C("*"), np.where(s > 0.35, C("+"), C("."))))
        col[qy, qx] = COLORS[k] * np.clip(np.where(bloss, s, 0.25 + s), 0, 1.3)[:, None]
        steady[qy, qx] = bloss
        hot = np.flatnonzero(~bloss & (s > 0.8))
        for j in hot[np.argsort(-s[hot])[:STAR_FLARES]]:
            rays(char, col, int(qy[j]), int(qx[j]), int(1 + 4 * min(s[j], 1.3) ** 2 / 1.3), COLORS[k[j]], 0.75 * min(s[j], 1.2))

    for pos, rose_k, pose, fade in fall:
        fx_, fy_, fz_ = look(pos[None])
        cx = int((0.5 + fx_[0] / fz_[0] * FOCAL * 0.5 / aspect) * cols)
        cy = int((MIDDLE - fy_[0] / fz_[0] * FOCAL * 0.5) * rows)
        for j, letter in enumerate(FALL_POSES[pose]):
            x_ = cx + j
            if letter and 0 <= cy < rows and 0 <= x_ < cols and (not body[cy, x_] or fz_[0] < depth[cy, x_]):
                char[cy, x_] = C(letter)
                col[cy, x_] = ROSE_COLOURS[rose_k, 0] * (1.15 * fade)
                steady[cy, x_] = False

    for (hp, f) in heads:
        if f > 0.15:
            hx, hy, hz = look(hp[None])
            cx = int((0.5 + hx[0] / hz[0] * FOCAL * 0.5 / aspect) * cols)
            cy = int((MIDDLE - hy[0] / hz[0] * FOCAL * 0.5) * rows)
            rays(char, col, cy, cx, int(3 + 5 * f), WHITE, 0.6 + 0.6 * f)
    score = np.where(sparkle, lum + (h % 7) * 0.01, 0)
    for n, idx in enumerate(np.argsort(score.ravel())[::-1][:4]):
        if score.ravel()[idx] <= 0:
            break
        cy, cx = divmod(int(idx), cols)
        rays(char, col, cy, cx, int(3 + 5 * (0.5 + 0.5 * np.sin(t * 6 + n * 1.9))), WHITE, 1.0, over=2.2)
    if T_RIBBON <= t < T_RIBBON + FLASH:
        kx, ky, kz = look((WAIST + [0.0, 0.0, 0.13])[None] @ spin_y(sway(t)).T)
        g = (1 - (t - T_RIBBON) / FLASH) ** 2
        rays(char, col, int((MIDDLE - ky[0] / kz[0] * FOCAL * 0.5) * rows),
             int((0.5 + kx[0] / kz[0] * FOCAL * 0.5 / aspect) * cols), int(2 + 6 * g), WHITE, 0.6 + 0.6 * g)
    if T_GLEAM[1] - 0.2 < t < T_GLEAM[1] + 0.6:
        kx, ky, kz = look((WAIST + [0.0, 0.0, 0.13])[None] @ spin_y(sway(t)).T)
        g = np.exp(-((t - T_GLEAM[1] - 0.15) / 0.2) ** 2)
        rays(char, col, int((MIDDLE - ky[0] / kz[0] * FOCAL * 0.5) * rows),
             int((0.5 + kx[0] / kz[0] * FOCAL * 0.5 / aspect) * cols), int(2 + 5 * g), COLORS[GOLD], 0.5 + 0.7 * g)

    info["message"] = message(char, col, t, cw, ch) if t > T_MSG else None
    info["letters"] = int((char > 0).sum())
    return char, col, steady, info


PINK = np.array([1.0, 0.74, 0.86], np.float32)
LOVE_RED = np.array([1.0, 0.16, 0.30], np.float32)


MESSAGE_PX = 15


def message_px(width):
    return max(8, round(MESSAGE_PX * width / 792))


def message(char, col, t, cw, ch):
    rows, cols = char.shape
    _, y_, z_ = look(np.array([[0.0, BOTTOM_Y - 0.28, 0.0]], np.float32))
    row = int((MIDDLE - y_[0] / z_[0] * FOCAL * 0.5) * rows)
    size = message_px(rows * ch * W / H)
    advance = 0.6 * size
    x0 = (cols * cw - advance * len(MESSAGE)) / 2
    c0, c1 = int(x0 // cw) - 2, int((x0 + advance * len(MESSAGE)) // cw) + 3
    reach = int(np.ceil(0.75 * size / ch))
    char[max(row - reach, 0):row + reach + 1, max(c0, 0):c1] = 0
    col[max(row - reach, 0):row + reach + 1, max(c0, 0):c1] = 0
    shown = min(len(MESSAGE), int((t - T_MSG) * 10))
    love = MESSAGE.index("<3")
    ph = (t - T_MSG - love / 10 + 0.6) % 1.1
    beat = np.exp(-(ph / 0.06) ** 2) + 0.55 * np.exp(-((ph - 0.24) / 0.06) ** 2)
    letters = []
    for i, letter in enumerate(MESSAGE[:shown]):
        if letter == " ":
            continue
        new = np.clip(1 - ((t - T_MSG) * 10 - i) / 3, 0, 1)
        if i >= love:
            tint = (LOVE_RED + (WHITE - LOVE_RED) * new) * (1.0 + 0.5 * beat)
        else:
            shimmer = np.clip(np.sin(t * 2.6 - i * 0.55), 0, 1) ** 6
            tint = PINK + (WHITE - PINK) * max(new, 0.8 * shimmer)
        letters.append((letter, x0 + advance * (i + 0.5), np.clip(tint, 0, 1)))
    return {"size": size, "y": (row + 0.5) * ch, "letters": letters}


TITLE = "Rose bouquet in ASCII"
CW, CH, FONT_PX = 7, 10, 11
COLS, ROWS = 113, 105
W, H = COLS * CW, ROWS * CH
HINT = "Space pause · R restart · S screenshot · Esc quit"
DIM, ACCENT, SIGN = (110, 104, 120), (255, 122, 182), (176, 168, 196)
TRAIL = 0.45
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
GREYS = [(i, i, i) for i in range(256)]
_ATLAS = {}


def glyph_atlas(cw, ch, font_px):
    if (cw, ch, font_px) not in _ATLAS:
        pygame.font.init()
        font = pygame.font.SysFont(MONO, font_px)
        cells = []
        for code in range(32, 127):
            s = pygame.Surface((cw, ch))
            g = font.render(chr(code), True, (255, 255, 255))
            s.blit(g, ((cw - g.get_width()) // 2, (ch - g.get_height()) // 2))
            cells.append(pygame.surfarray.array3d(s).transpose(1, 0, 2)[..., 0])
        _ATLAS[cw, ch, font_px] = np.stack(cells)
    return _ATLAS[cw, ch, font_px]


class Renderer:
    def __init__(self, cols=COLS, rows=ROWS, cw=CW, ch=CH, font_px=FONT_PX, hud_px=15, hint=True):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.w, self.h = cols * cw, rows * ch
        self.atlas = glyph_atlas(cw, ch, font_px)
        self.art = pygame.Surface((self.w, self.h))
        self.trail = pygame.Surface((self.w, self.h))
        self.out = pygame.Surface((self.w, self.h))
        self.tint = pygame.Surface((cols, rows))
        self.moving, self.still = pygame.Surface((self.w, self.h)), pygame.Surface((self.w, self.h))
        self.small = pygame.font.SysFont(MONO, hud_px)
        self.big = pygame.font.SysFont(MONO, message_px(self.h * W / H))
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

    def tinted(self, col, into):
        pygame.surfarray.blit_array(self.tint, (np.clip(col, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2))
        pygame.transform.scale(self.tint, (self.w, self.h), into)
        into.blit(self.art, (0, 0), special_flags=pygame.BLEND_MULT)
        return into

    def draw(self, t):
        w, h = self.w, self.h
        char, col, steady, info = frame(t, self.cols, self.rows, self.cw, self.ch)
        grey = pygame.image.frombuffer(self.atlas[char].transpose(0, 2, 1, 3).tobytes(), (w, h), "P")
        grey.set_palette(GREYS)
        self.art.blit(grey, (0, 0))
        if self.last_t is None or t < self.last_t:
            self.trail.fill((0, 0, 0))
        self.last_t = t
        self.trail.fill((int(TRAIL * 255),) * 3, special_flags=pygame.BLEND_MULT)
        self.trail.blit(self.tinted(col * ~steady[..., None], self.moving), (0, 0), special_flags=pygame.BLEND_MAX)
        self.out.blit(self.trail, (0, 0))
        self.out.blit(self.tinted(col * steady[..., None], self.still), (0, 0), special_flags=pygame.BLEND_MAX)
        if info["message"]:
            m = info["message"]
            for letter, x, rgb in m["letters"]:
                g = self.big.render(letter, True, tuple(int(255 * v) for v in rgb))
                self.out.blit(g, (int(x - g.get_width() / 2), int(m["y"] - g.get_height() / 2)))
        self.glow(self.out)
        self.t, self.info = t, info
        return self.out

    def hud(self, surface, margin=12):
        sw, sh = surface.get_size()
        info, t = self.info, self.t
        done = min(t / LOOP, 1.0) * 100
        roses = f"      roses open {info['open']:>2}/{N_ROSES}" if info["grown"] >= 1 else ""
        text = f"growing {done:5.1f}%{roses}      letters {info['letters']:>5,}"
        line = self.small.render(text, True, DIM)
        surface.blit(line, ((sw - line.get_width()) // 2, sh - margin - self.hud_gap))


def fit_to(sw, sh):
    k = min(sw / W, sh / H)
    cw, ch = max(3, int(CW * k)), max(4, int(CH * k))
    font_px = max(6, round(FONT_PX * ch / CH))
    cols, rows = sw // cw, sh // ch
    return (cols, rows, cw, ch, font_px), ((sw - cols * cw) // 2, (sh - rows * ch) // 2)


SONG = Path(__file__).with_name("her.mp3")


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
    (cols, rows, cw, ch, font_px), at = fit_to(*window.get_size())
    renderer, clock = Renderer(cols=cols, rows=rows, cw=cw, ch=ch, font_px=font_px), pygame.time.Clock()
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
                if music:
                    pygame.mixer.music.play(loops=-1)
                    if paused:
                        pygame.mixer.music.pause()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                pygame.image.save(screen, str(Path(__file__).with_name(f"bouquet_{int(time.time())}.png")))
        dt = clock.tick(30) / 1000
        if not paused:
            t += min(dt, 1 / 15)
            if t > LOOP:
                t = 0.0
            screen = renderer.draw(t)
        window.fill((0, 0, 0))
        window.blit(screen, at)
        renderer.hud(window)
        pygame.display.flip()


if __name__ == "__main__":
    sys.exit(main())