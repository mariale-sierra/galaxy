"""Planeta rocoso "Verdeluz" (inspirado en los mundos de Rosalina / Super Mario Galaxy):
pasto, roca marron con crateres-cueva, piedras, parches de flores, estanques turquesa,
nubes con sombras, flores bioluminiscentes de noche y halo celeste."""
import numpy as np
from noise import fbm, noise3, worley
from .util import sstep, mix, dot, nrm, rotY, tangents

P = dict(
    terrain_speed=0.05,   # rotacion del terreno (rad/s)
    cloud_speed=0.30,     # rotacion de las nubes (rad/s): distinta a la del terreno
    cloud_cover=0.42,     # 0 = cielo despejado, 1 = cubierto
    cloud_height=0.10,    # altura de las nubes: separa la sombra de la nube
    cloud_shadow=0.55,    # oscuridad de las sombras
    bump=0.060,           # fuerza del relieve
    land_scale=1.45,      # tamano de los "continentes" de roca/pasto
    crater_scale=4.6,     # cantidad de crateres
    flower_scale=3.2,     # cantidad de parches de flores
    pond_level=-0.33,     # nivel del agua (mas alto = mas estanques)
    night_glow=1.0,       # brillo nocturno de las flores
    atmosphere=0.9,
)

GRASS_A, GRASS_B = np.array([0.10, 0.42, 0.09], np.float32), np.array([0.36, 0.74, 0.16], np.float32)
ROCK_A, ROCK_B = np.array([0.42, 0.31, 0.25], np.float32), np.array([0.64, 0.53, 0.44], np.float32)
DIRT = np.array([0.26, 0.15, 0.07], np.float32)
STONE = np.array([0.55, 0.58, 0.60], np.float32)
PETALS = np.array([[0.62, 0.30, 0.90], [1.00, 0.88, 0.22], [0.96, 0.96, 1.00]], np.float32)


def terrain(p):
    """Campo de altura + mascaras. p = direcciones unitarias en el marco del terreno."""
    land = fbm(p * P["land_scale"] + 3.1, 4) + 0.12 * noise3(p * 6.0)
    rock = sstep(0.04, 0.085, land)
    dc, idc = worley(p * P["crater_scale"] + 9.0)
    rr = dc / 0.36
    cr = idc > 0.38
    bowl = np.where(cr, -(1 - rr ** 2).clip(0) * 0.9 + 0.35 * np.exp(-((rr - 1.05) / 0.18) ** 2), 0.0)
    ds, ids = worley(p * 7.5 + 2.0)
    rs = ds / 0.30
    st = ids > 0.72
    dome = np.where(st, np.sqrt((1 - rs ** 2).clip(0)) * 0.5, 0.0) * (1 - rock)
    h = 0.06 * fbm(p * 10, 3) + 0.20 * rock + bowl * rock + dome
    return h, dict(land=land, rock=rock, rr=rr, cr=cr, rs=rs, st=st, idc=idc)


def cloud_density(n, t):
    cp = rotY(n, t * P["cloud_speed"] + 2.0)
    w = fbm(cp * 2.0 + 5.0, 3)
    c = fbm(cp * 2.4 + w[:, None] * 0.6, 5)
    thr = 0.34 - 0.62 * P["cloud_cover"]
    return sstep(thr, thr + 0.22, c)


def rocky_shader(F):
    t, n = F.t, F.n
    ang = t * P["terrain_speed"]
    p = rotY(n, ang)                  # marco del terreno
    Lt, Vt = rotY(F.L, ang), rotY(F.V, ang)

    # ---------- relieve (bump mapping: perturba la normal con el gradiente de la altura) ----------
    h, m = terrain(p)
    e = 0.012
    t1, t2 = tangents(p)
    h1, _ = terrain(nrm(p + e * t1))
    h2, _ = terrain(nrm(p + e * t2))
    nb = nrm(p - P["bump"] / e * ((h1 - h)[:, None] * t1 + (h2 - h)[:, None] * t2))
    diff = np.clip(dot(nb, Lt), 0, 1)

    # ---------- albedo por capas ----------
    land, rock, rr, cr, rs, st = m["land"], m["rock"], m["rr"], m["cr"], m["rs"], m["st"]
    grass = mix(GRASS_A, GRASS_B, 0.5 + 0.5 * fbm(p * 9, 3))
    rockc = mix(ROCK_A, ROCK_B, 0.5 + 0.5 * fbm(p * 6 + 4, 3))
    alb = mix(grass, rockc, rock)
    edge = 1 - sstep(0.0, 0.05, np.abs(land - 0.055))               # franja de tierra entre pasto y roca
    alb = mix(alb, DIRT, 0.9 * edge)
    hole = cr * rock * (1 - sstep(0.40, 0.85, rr))                  # cuevas oscuras en la roca
    alb = mix(alb, (0.06, 0.045, 0.04), hole)
    stone = st * (1 - sstep(0.80, 1.0, rs)) * (1 - rock)            # piedras redondas sobre el pasto
    alb = mix(alb, STONE, stone)
    # flores
    df, idf = worley(p * P["flower_scale"] + 17.0)
    rf = df / 0.30
    isflower = (idf > 0.60)
    fmask = isflower * (1 - sstep(0.80, 1.0, rf)) * (1 - rock) * (1 - edge) * (1 - stone)
    kind = np.floor(((idf * 7.31) % 1.0) * 3).astype(int).clip(0, 2)
    petal = PETALS[kind]
    dots = sstep(0.05, 0.35, np.sin(p[:, 0] * 48) * np.sin(p[:, 1] * 48) * np.sin(p[:, 2] * 48))
    fcol = mix(GRASS_A * 1.1, petal, 0.25 + 0.75 * dots)
    alb = mix(alb, fcol, fmask)
    # estanques en las zonas bajas (reflejan la luz)
    pond = (1 - sstep(P["pond_level"] - 0.05, P["pond_level"] + 0.02, land)) * (1 - rock)
    water = mix((0.02, 0.22, 0.45), (0.18, 0.80, 0.90), sstep(-0.55, P["pond_level"], land))
    alb = mix(alb, water, pond)

    # ---------- iluminacion ----------
    amb = np.array([0.10, 0.14, 0.30], np.float32) * 0.9
    col = alb * (amb + diff[:, None] * np.array([1.0, 0.96, 0.88], np.float32))
    Hh = nrm(Lt + Vt)
    ndh = np.clip(dot(nb, Hh), 0, 1)
    spec = (ndh ** 110) * pond * 2.2 + (ndh ** 28) * stone * 0.25 + (ndh ** 40) * hole * 0.0
    col = col + (spec * (diff > 0))[:, None] * np.array([1.0, 1.0, 0.95], np.float32)
    mu = np.clip(dot(n, F.V), 0, 1)
    col = col + (pond * (1 - mu) ** 3 * 0.55)[:, None] * np.array([0.4, 0.7, 1.0], np.float32)

    # ---------- noche: flores y estanques bioluminiscentes ----------
    night = 1 - sstep(-0.08, 0.22, dot(n, F.L))
    pulse = 0.85 + 0.15 * np.sin(t * 2.0 + idf * 6.0)
    col = col + (fmask * night * P["night_glow"] * pulse)[:, None] * petal * 1.1
    col = col + (pond * night * 0.35)[:, None] * np.array([0.1, 0.8, 0.9], np.float32)

    # ---------- nubes y sombras de nubes ----------
    dens = cloud_density(n, t)
    Lt_tan = F.L - n * dot(n, F.L)[:, None]                          # componente tangente de la luz
    ps = nrm(n - Lt_tan * P["cloud_height"] / 0.25)                  # donde estaria la nube que tapa este punto
    shadow = cloud_density(ps, t)
    col = col * (1 - P["cloud_shadow"] * shadow * (1 - dens))[:, None]
    wrap = np.clip((dot(n, F.L) + 0.15) / 1.15, 0, 1)
    cloud_col = np.array([0.97, 0.98, 1.0], np.float32) * (0.10 + 0.95 * wrap)[:, None] + np.array([0.02, 0.03, 0.07], np.float32)
    col = mix(col, cloud_col, dens * 0.92)

    # ---------- atmosfera celeste ----------
    lit = 0.30 + 0.70 * sstep(-0.3, 0.6, dot(n, F.L))
    col = col + ((1 - mu) ** 3 * lit * P["atmosphere"])[:, None] * np.array([0.35, 0.90, 1.0], np.float32)
    return col
