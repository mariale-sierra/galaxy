"""Estrella "Luma Solar": sol naranja-dorado con granulacion convectiva, manchas y filamentos.
No depende de la luz externa (es emisiva)."""
import numpy as np
from noise import fbm, worley
from .util import sstep, mix, ramp, dot

P = dict(
    speed=0.35,            # velocidad de la animacion (ebullicion de la superficie)
    granule_scale=2.6,     # tamano de las celdas de convección (mayor = celdas mas pequenas)
    warp_amount=0.55,      # cuanto se distorsionan las celdas con ruido
    spot_scale=1.25,       # tamano de las manchas solares
    spot_threshold=0.20,   # umbral: mas bajo = mas manchas
    flare_strength=0.8,    # brillo de los filamentos calientes
    limb_base=0.55,        # oscurecimiento hacia el borde (limb darkening)
    corona_strength=0.75,  # brillo del borde
)

HEAT = [(0.0, (0.45, 0.05, 0.01)), (0.30, (0.85, 0.22, 0.02)), (0.55, (1.0, 0.50, 0.05)),
        (0.78, (1.0, 0.80, 0.25)), (1.0, (1.0, 0.98, 0.75))]


def star_shader(F):
    n, t = F.n, F.t * P["speed"]
    q = n * P["granule_scale"]
    o1 = np.array([t, 0.6 * t, -0.4 * t], np.float32)
    o2 = np.array([-0.5 * t, 0.8 * t, 0.3 * t], np.float32)
    # 1) campo de distorsion animado (3 fBm) -> las celdas "hierven"
    warp = np.stack([fbm(q + o1 + 5.0, 3), fbm(q + o2 - 3.0, 3), fbm(q + o1[::-1] + 11.0, 3)], 1) * P["warp_amount"]
    # 2) celdas Worley: centro caliente, bordes (carriles) frios
    d, cid = worley(q * 2.2 + warp * 2.0 + np.array([0, t * 1.5, 0], np.float32))
    cell = 1 - sstep(0.10, 0.80, d)
    # 3) temperatura = celdas + turbulencia grande + variacion por celda + pulso
    turb = 0.5 + 0.5 * fbm(q * 1.5 + warp + o1, 4)
    pulse = 1 + 0.08 * np.sin(t * 6 + cid * 6.28)
    heat = np.clip((0.50 * cell + 0.32 * turb + 0.22 * cid) * pulse, 0, 1)
    col = ramp(heat, HEAT)
    # 4) manchas solares (umbra oscura + penumbra)
    s = fbm(n * P["spot_scale"] + np.array([0, 0, t * 0.05 + 7], np.float32), 3)
    th = P["spot_threshold"]
    pen = sstep(th, th + 0.12, s)
    umb = sstep(th + 0.08, th + 0.22, s)
    col = col * (1 - 0.45 * pen)[:, None]
    col = mix(col, (0.10, 0.015, 0.0), 0.88 * umb)
    # 5) filamentos calientes (ruido "ridged")
    ridge = 1 - np.abs(fbm(q * 2.0 + warp * 1.5 + o2, 4))
    veins = sstep(0.88, 0.985, ridge) * (1 - pen)
    col = col + veins[:, None] * np.array([1.0, 0.85, 0.40], np.float32) * P["flare_strength"]
    # 6) limb darkening + brillo de borde (corona)
    mu = np.clip(dot(n, F.V), 0, 1)
    col = col * (P["limb_base"] + (1 - P["limb_base"]) * mu ** 0.5)[:, None] * 1.15
    rim = (1 - mu) ** 3
    col = col + rim[:, None] * np.array([1.0, 0.55, 0.10], np.float32) * P["corona_strength"]
    return col
