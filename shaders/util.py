"""Funciones auxiliares para los fragment shaders (todo vectorizado: N fragmentos a la vez)."""
import numpy as np


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(c0, c1, t):
    t = np.asarray(t, np.float32)[:, None]
    return np.asarray(c0, np.float32) * (1 - t) + np.asarray(c1, np.float32) * t


def ramp(x, stops):
    """Rampa de color: stops = [(posicion, (r,g,b)), ...]."""
    pos = [s[0] for s in stops]
    cols = np.array([s[1] for s in stops], np.float32)
    x = np.clip(x, pos[0], pos[-1])
    return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], 1).astype(np.float32)


def dot(a, b):
    return (a * b).sum(1)


def nrm(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def rotY(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[:, 0] + s * v[:, 2], v[:, 1], -s * v[:, 0] + c * v[:, 2]], 1)


def rot_axis(v, c, th):
    """Rodrigues: rota v alrededor del eje unitario c un angulo th (por fragmento)."""
    ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
    c = np.asarray(c, np.float32)
    return v * ct + np.cross(c[None, :], v) * st + c[None, :] * (v @ c)[:, None] * (1 - ct)


def tangents(n):
    ref = np.where((np.abs(n[:, 1]) > 0.95)[:, None], np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32))
    t1 = nrm(np.cross(n, ref))
    t2 = np.cross(n, t1)
    return t1, t2
