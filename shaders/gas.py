"""Gigante gaseoso "Aqualia": bandas azul-hielo liquidas + gran vortice localizado."""
import numpy as np
from noise import fbm
from .util import sstep, mix, ramp, dot, nrm, rotY, rot_axis

P = dict(
    band_freq=7.5,        # numero de bandas
    warp=0.16,            # cuanto "se licuan" los bordes de las bandas
    flow_speed=0.6,       # velocidad del fluir
    shear=6.0,            # cuanto cambia la velocidad con la latitud (cizalla)
    storm_lat=-0.30,      # latitud (y) del vortice
    storm_lon=0.35,       # longitud del vortice (rad)
    storm_radius=0.30,    # tamano del vortice
    storm_twist=5.5,      # cuanto se curvan las bandas alrededor (rad en el centro)
    rim_strength=0.8,
)
BANDS = [(0.00, (0.05, 0.08, 0.38)), (0.25, (0.15, 0.35, 0.85)), (0.50, (0.35, 0.78, 1.00)),
         (0.75, (0.93, 0.97, 1.00)), (1.00, (0.68, 0.58, 0.93))]


def gas_shader(F):
    n, t = F.n, F.t
    # storm center
    ys = P["storm_lat"]
    xz = np.sqrt(1 - ys * ys)
    c = np.array([xz * np.sin(P["storm_lon"]), ys, xz * np.cos(P["storm_lon"])], np.float32)
    d = np.linalg.norm(n - c, axis=1)
    # 1) flujo diferencial: cada latitud se desplaza distinto -> cizalla liquida
    ang = t * 0.04 + 0.30 * np.sin(t * P["flow_speed"] + n[:, 1] * P["shear"])
    n1 = rotY(n, ang)
    # 2) vortice: rota el punto de muestreo alrededor del centro (mas en el centro)
    twist = P["storm_twist"] * np.exp(-(d / P["storm_radius"]) ** 2)
    n2 = rot_axis(n1, c, twist)
    # 3) bandas = latitud deformada por ruido (dos escalas) -> bordes ondulantes
    off = np.array([0, t * 0.12, 0], np.float32)
    w1 = fbm(n2 * np.array([2.0, 4.0, 2.0], np.float32) + off, 4)
    w2 = fbm(n2 * np.array([6.0, 14.0, 6.0], np.float32) - off, 3)
    lat = n2[:, 1] + P["warp"] * w1 + 0.035 * w2
    b1 = 0.5 + 0.5 * np.sin(lat * P["band_freq"] * np.pi)
    b2 = 0.5 + 0.5 * np.sin(lat * P["band_freq"] * 0.41 * np.pi + 1.3)
    streak = 0.12 * fbm(n2 * np.array([3.0, 24.0, 3.0], np.float32), 3)
    v = np.clip(0.62 * b1 + 0.38 * b2 + streak, 0, 1)
    col = ramp(v, BANDS)
    # 4) ojo del vortice: nucleo claro + anillo oscuro + brazos en espiral
    eye = 1 - sstep(0.0, P["storm_radius"] * 0.55, d)
    ring = np.exp(-((d - P["storm_radius"] * 0.75) / (P["storm_radius"] * 0.22)) ** 2)
    col = mix(col, (0.12, 0.12, 0.55), 0.55 * ring)
    col = mix(col, (1.0, 0.82, 0.92), 0.85 * eye)
    # 5) iluminacion suave + atmosfera
    dif = np.clip((dot(n, F.L) + 0.15) / 1.15, 0, 1)
    col = col * (np.array([0.06, 0.08, 0.20], np.float32) + dif[:, None] * np.array([1.0, 0.98, 0.95], np.float32) * 1.05)
    mu = np.clip(dot(n, F.V), 0, 1)
    lit = 0.25 + 0.75 * sstep(-0.3, 0.6, dot(n, F.L))
    col = col + ((1 - mu) ** 3 * lit * P["rim_strength"])[:, None] * np.array([0.45, 0.75, 1.0], np.float32)
    return col
