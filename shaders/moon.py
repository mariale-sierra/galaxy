"""Luna gris con crateres (Worley) y relieve por bump mapping."""
import numpy as np
from noise import fbm, worley
from .util import sstep, mix, dot, nrm, tangents

P = dict(crater_scale=3.4, bump=0.07)


def _height(p):
    d1, i1 = worley(p * P["crater_scale"])
    d2, i2 = worley(p * P["crater_scale"] * 2.3 + 5.0)
    r1, r2 = d1 / 0.38, d2 / 0.36
    h = np.where(i1 > 0.35, -(1 - r1 ** 2).clip(0) * 0.9 + 0.35 * np.exp(-((r1 - 1.05) / 0.18) ** 2), 0)
    h += np.where(i2 > 0.55, -(1 - r2 ** 2).clip(0) * 0.5 + 0.2 * np.exp(-((r2 - 1.05) / 0.2) ** 2), 0)
    return h + 0.05 * fbm(p * 8, 3), r1, i1


def moon_shader(F):
    n = F.n
    h, r1, i1 = _height(n)
    e = 0.012
    t1, t2 = tangents(n)
    h1, _, _ = _height(nrm(n + e * t1))
    h2, _, _ = _height(nrm(n + e * t2))
    nb = nrm(n - P["bump"] / e * ((h1 - h)[:, None] * t1 + (h2 - h)[:, None] * t2))
    diff = np.clip(dot(nb, F.L), 0, 1)
    base = 0.55 + 0.25 * fbm(n * 5, 4)
    alb = np.stack([base, base * 0.98, base * 0.95], 1)
    alb = alb * (1 - 0.45 * ((i1 > 0.35) & (r1 < 0.6)))[:, None]
    return alb * (0.04 + 1.0 * diff)[:, None] * np.array([1.0, 0.97, 0.9], np.float32)
