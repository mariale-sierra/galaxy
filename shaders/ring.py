"""Anillos del gigante gaseoso: bandas por radio + sombra analitica del planeta."""
import numpy as np
from noise import noise3
from .util import ramp, dot, sstep

P = dict(stripe_freq=95.0, shadow_strength=0.85)
RING_COL = [(0.0, (0.55, 0.62, 0.80)), (0.4, (0.85, 0.90, 1.0)), (0.7, (0.95, 0.80, 0.95)), (1.0, (0.70, 0.78, 0.95))]


def ring_shader(F):
    p = F.lp
    r = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
    z = np.zeros_like(r)
    s = 0.5 + 0.5 * np.sin(r * P["stripe_freq"] + 2.5 * noise3(np.stack([r * 6, z, z + 1.0], 1)))
    g = 0.5 + 0.5 * noise3(np.stack([r * 30, z + 2.0, z], 1))
    col = ramp(0.55 * s + 0.45 * g, RING_COL) * (0.75 + 0.25 * s)[:, None]
    ldiff = np.abs(F.L[:, 1])
    # sombra del planeta (esfera unitaria en el origen local) sobre el anillo
    b = dot(p, F.L)
    c = dot(p, p) - 1.0
    disc = b * b - c
    hit = (disc > 0) & ((-b - np.sqrt(np.maximum(disc, 0))) > 0)
    col = col * (0.35 + 0.65 * ldiff)[:, None]
    col = col * (1 - P["shadow_strength"] * hit)[:, None]
    return col
