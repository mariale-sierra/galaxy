"""Ruido procedural vectorizado con NumPy (gradient noise 3D, fBm y Worley)."""
import numpy as np

_rng = np.random.default_rng(7)
_perm = np.concatenate([_rng.permutation(256)] * 2).astype(np.int32)
_G = _rng.normal(size=(256, 3))
_G = (_G / np.linalg.norm(_G, axis=1, keepdims=True)).astype(np.float32)
_R = _rng.random(256).astype(np.float32)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def noise3(p):
    """Gradient noise 3D. p: (N,3) -> (N,) aprox. en [-1,1]."""
    p = np.asarray(p, np.float32)
    f = np.floor(p)
    i = f.astype(np.int32)
    d = p - f
    xi, yi, zi = i[:, 0], i[:, 1], i[:, 2]
    u, v, w = _fade(d[:, 0]), _fade(d[:, 1]), _fade(d[:, 2])

    def corner(dx, dy, dz):
        h = _perm[_perm[_perm[(xi + dx) & 255] + ((yi + dy) & 255)] + ((zi + dz) & 255)]
        g = _G[h]
        return g[:, 0] * (d[:, 0] - dx) + g[:, 1] * (d[:, 1] - dy) + g[:, 2] * (d[:, 2] - dz)

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
    x00 = c000 + (c100 - c000) * u
    x10 = c010 + (c110 - c010) * u
    x01 = c001 + (c101 - c001) * u
    x11 = c011 + (c111 - c011) * u
    y0 = x00 + (x10 - x00) * v
    y1 = x01 + (x11 - x01) * v
    return (y0 + (y1 - y0) * w) * 1.15


def fbm(p, octaves=4, lac=2.0, gain=0.5):
    """Suma de octavas de ruido. Devuelve (N,) aprox. en [-1,1]."""
    p = np.asarray(p, np.float32)
    tot, amp, norm = 0.0, 1.0, 0.0
    for _ in range(octaves):
        tot = tot + amp * noise3(p)
        norm += amp
        p = p * lac + 17.3
        amp *= gain
    return tot / norm


def worley(p):
    """Ruido celular. Devuelve (distancia al punto mas cercano, id aleatorio [0,1))."""
    p = np.asarray(p, np.float32)
    f = np.floor(p)
    i = f.astype(np.int32)
    d = p - f
    xi, yi, zi = i[:, 0], i[:, 1], i[:, 2]
    best = np.full(len(p), 1e9, np.float32)
    bid = np.zeros(len(p), np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                h = _perm[_perm[_perm[(xi + dx) & 255] + ((yi + dy) & 255)] + ((zi + dz) & 255)]
                g = _G[h]
                fx = dx + 0.5 + 0.38 * g[:, 0] - d[:, 0]
                fy = dy + 0.5 + 0.38 * g[:, 1] - d[:, 1]
                fz = dz + 0.5 + 0.38 * g[:, 2] - d[:, 2]
                dist = fx * fx + fy * fy + fz * fz
                m = dist < best
                best = np.where(m, dist, best)
                bid = np.where(m, h, bid)
    return np.sqrt(best), _R[bid]
