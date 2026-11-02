"""Software renderer en CPU: rasterizador con z-buffer + fragment shaders diferidos.

Pipeline:
 1. Vertex stage: model -> mundo -> camara -> pantalla (NumPy).
 2. Raster: por triangulo, coordenadas baricentricas con correccion de perspectiva.
    Solo guarda (objeto, triangulo, baricentricas) del fragmento mas cercano.
 3. Fragment stage (diferido): el shader corre UNA vez por pixel visible.
"""
import numpy as np
from noise import fbm


class Mesh:
    def __init__(self, V, F):
        self.V, self.F = V.astype(np.float32), F.astype(np.int32)


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    V = np.array(V, np.float32)
    V -= V.mean(0)
    V /= np.linalg.norm(V, axis=1).mean()  # radio ~1
    return Mesh(V, np.array(F, np.int32))


def make_rings(bands, seg=96):
    """Anillos planos (gaps reales entre bandas). Solo para el extra de anillos."""
    V, F = [], []
    for r0, r1 in bands:
        base = len(V)
        for k in range(seg):
            a = 2 * np.pi * k / seg
            V.append([r0 * np.cos(a), 0, r0 * np.sin(a)])
            V.append([r1 * np.cos(a), 0, r1 * np.sin(a)])
        for k in range(seg):
            i0, o0 = base + 2 * k, base + 2 * k + 1
            i1, o1 = base + 2 * ((k + 1) % seg), base + 2 * ((k + 1) % seg) + 1
            F += [[i0, o0, i1], [o0, o1, i1]]
    return Mesh(np.array(V, np.float32), np.array(F, np.int32))


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


class Obj:
    def __init__(self, mesh, shader, scale=1.0, pos=(0, 0, 0), rot=None, glow=None, name=""):
        self.mesh, self.shader, self.scale = mesh, shader, scale
        self.pos = np.array(pos, np.float32)
        self.R = np.eye(3, dtype=np.float32) if rot is None else rot.astype(np.float32)
        self.glow = glow  # (color, radio_en_fraccion_de_alto, intensidad)
        self.name = name


class Frag:
    """Datos que recibe cada fragment shader (todo en el espacio LOCAL del objeto)."""
    pass


class Scene:
    def __init__(self, objs, light_pos, yaw=0.0, pitch=0.15, dist=5.0):
        self.objs, self.light_pos = objs, np.array(light_pos, np.float32)
        self.yaw, self.pitch, self.dist = yaw, pitch, dist


_bg_cache = {}


def background(W, H):
    key = (W, H)
    if key in _bg_cache:
        return _bg_cache[key]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = np.array([0.015, 0.02, 0.10], np.float32) * (1 - y) + np.array([0.03, 0.10, 0.32], np.float32) * y
    img = np.broadcast_to(img, (H, W, 3)).copy()
    # nebulosa suave
    xs, ys = np.meshgrid(np.linspace(0, 3 * W / H, W), np.linspace(0, 3, H))
    p = np.stack([xs.ravel(), ys.ravel(), np.full(W * H, 1.7)], 1)
    n = np.clip(fbm(p, 4).reshape(H, W) + 0.1, 0, 1)
    img += n[..., None] * np.array([0.30, 0.10, 0.45], np.float32) * 0.45
    rng = np.random.default_rng(3)
    for _ in range(W * H // 900):
        x, yy = rng.integers(0, W), rng.integers(0, H)
        b = rng.random() ** 2
        c = np.array([0.8, 0.9, 1.0]) if rng.random() < 0.7 else np.array([1.0, 0.8, 1.0])
        img[yy, x] = np.clip(img[yy, x] + c * (0.35 + 0.65 * b), 0, 1)
        if b > 0.6 and 1 <= x < W - 1 and 1 <= yy < H - 1:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                img[yy + dy, x + dx] = np.clip(img[yy + dy, x + dx] + c * 0.35, 0, 1)
    _bg_cache[key] = img
    return img


def _box0(a, r):
    h = a.shape[0]
    pad = np.pad(a, ((r + 1, r), (0, 0)))
    cs = np.cumsum(pad, axis=0)
    return (cs[2 * r + 1:] - cs[:h]) / (2 * r + 1)


def blur(a, r):
    r = max(int(r), 1)
    for _ in range(3):
        a = _box0(a, r)
        a = _box0(a.T, r).T
    return a


def render(scene, W, H, t, ssaa=1):
    w, h = W * ssaa, H * ssaa
    zbuf = np.full((h, w), np.inf, np.float32)
    oid = np.full((h, w), -1, np.int16)
    tid = np.zeros((h, w), np.int32)
    bary = np.zeros((h, w, 3), np.float32)

    cp, sp, cy, sy = np.cos(scene.pitch), np.sin(scene.pitch), np.cos(scene.yaw), np.sin(scene.yaw)
    eye = np.array([scene.dist * cp * sy, scene.dist * sp, scene.dist * cp * cy], np.float32)
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 1, 0]).astype(np.float32)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd).astype(np.float32)
    tanf = np.tan(np.radians(45) / 2)
    aspect = w / h
    near = 0.1

    # ---- vertex + raster ----
    for k, o in enumerate(scene.objs):
        Wv = (o.mesh.V * o.scale) @ o.R.T + o.pos
        rel = Wv - eye
        cx, cyv, cz = rel @ right, rel @ up, rel @ fwd
        sx = (cx / (cz * tanf * aspect) * 0.5 + 0.5) * w
        sy_ = (0.5 - cyv / (cz * tanf) * 0.5) * h
        for ti, (a, b, c) in enumerate(o.mesh.F):
            if cz[a] <= near or cz[b] <= near or cz[c] <= near:
                continue
            x0, x1, x2 = sx[a], sx[b], sx[c]
            y0, y1, y2 = sy_[a], sy_[b], sy_[c]
            minx, maxx = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))), w - 1)
            miny, maxy = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))), h - 1)
            if minx > maxx or miny > maxy:
                continue
            det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(det) < 1e-9:
                continue
            px = np.arange(minx, maxx + 1, dtype=np.float32)[None, :] + 0.5
            py = np.arange(miny, maxy + 1, dtype=np.float32)[:, None] + 0.5
            l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
            l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
            l2 = 1 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            q0, q1, q2 = l0 / cz[a], l1 / cz[b], l2 / cz[c]  # perspectiva correcta
            zz = 1.0 / (q0 + q1 + q2)
            zs = zbuf[miny:maxy + 1, minx:maxx + 1]
            m = inside & (zz < zs)
            if not m.any():
                continue
            zs[m] = zz[m]
            oid[miny:maxy + 1, minx:maxx + 1][m] = k
            tid[miny:maxy + 1, minx:maxx + 1][m] = ti
            bary[miny:maxy + 1, minx:maxx + 1][m] = np.stack([q0 * zz, q1 * zz, q2 * zz], -1)[m]

    # ---- fragment stage ----
    img = np.repeat(np.repeat(background(W, H), ssaa, 0), ssaa, 1).copy()
    for k, o in enumerate(scene.objs):
        ys, xs = np.nonzero(oid == k)
        if len(ys) == 0:
            continue
        tri = o.mesh.F[tid[ys, xs]]
        b = bary[ys, xs]
        V = o.mesh.V
        lp = V[tri[:, 0]] * b[:, 0:1] + V[tri[:, 1]] * b[:, 1:2] + V[tri[:, 2]] * b[:, 2:3]
        Pw = (lp * o.scale) @ o.R.T + o.pos
        Lw = scene.light_pos - Pw
        Lw /= np.linalg.norm(Lw, axis=1, keepdims=True)
        Vw = eye - Pw
        Vw /= np.linalg.norm(Vw, axis=1, keepdims=True)
        F = Frag()
        F.t = t
        F.lp = lp
        F.L = Lw @ o.R          # luz en espacio local
        F.V = Vw @ o.R          # vista en espacio local
        F.n = lp / np.maximum(np.linalg.norm(lp, axis=1, keepdims=True), 1e-8)
        F.diff = np.clip((F.n * F.L).sum(1), 0, 1)
        F.name = o.name
        col = o.shader(F)
        img[ys, xs] = np.clip(col, 0, 1)

    # ---- halos / bloom (post-proceso 2D sobre el fondo) ----
    bgmask = (oid == -1)
    for k, o in enumerate(scene.objs):
        if o.glow is None or o.name == "ring":
            continue
        color, rad, strength = o.glow
        m = (oid == k).astype(np.float32)
        if m.sum() == 0:
            continue
        g = 0.6 * blur(m, rad * h * 0.35) + 0.4 * blur(m, rad * h)
        add = g[..., None] * np.array(color, np.float32) * strength
        img += add * bgmask[..., None]

    img = np.clip(img, 0, 1)
    if ssaa > 1:
        img = img.reshape(H, ssaa, W, ssaa, 3).mean((1, 3))
    return (img * 255).astype(np.uint8)
