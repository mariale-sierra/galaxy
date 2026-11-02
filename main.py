#!/usr/bin/env python3
"""Laboratorio: planetas y estrellas con shaders (estilo Super Mario Galaxy).

Interactivo:   python main.py            (teclas 1-4, flechas, +/-, espacio, S)
Sin ventana:   python main.py --shader rocky --time 3 --out shots/rocky.png
GIF:           python main.py --shader gas --gif shots/gas.gif --frames 36
"""
import argparse, os, time
import numpy as np
from renderer import load_obj, make_rings, rot_x, rot_z, Obj, Scene, render
from shaders import star_shader, rocky_shader, gas_shader, moon_shader, ring_shader

HERE = os.path.dirname(os.path.abspath(__file__))
SPHERE = load_obj(os.path.join(HERE, "sphere.obj"))   # MISMA esfera para todos los cuerpos
RINGS = make_rings([(1.30, 1.45), (1.50, 1.82), (1.88, 2.05), (2.10, 2.25)])  # extra: anillos

NAMES = {"1": "star", "2": "rocky", "3": "gas", "4": "system"}


def build(name, t):
    """Devuelve la escena (objetos + luz + camara) para el modo pedido."""
    if name == "star":
        sun = Obj(SPHERE, star_shader, 1.1, glow=((1.0, 0.55, 0.12), 0.10, 1.3), name="sun")
        return Scene([sun], (8, 4, 6), yaw=0.0, pitch=0.1, dist=4.2)
    if name == "rocky":
        planet = Obj(SPHERE, rocky_shader, 0.95, rot=rot_z(0.25), glow=((0.35, 0.9, 1.0), 0.05, 0.9), name="rocky")
        a = 0.9 + t * 0.35
        moon = Obj(SPHERE, moon_shader, 0.25, pos=(1.7 * np.cos(a), 0.45 * np.sin(a), 1.7 * np.sin(a)), name="moon")
        return Scene([planet, moon], (7, 3.5, 5.5), yaw=0.0, pitch=0.12, dist=5.2)
    if name == "gas":
        R = rot_z(0.40) @ rot_x(0.30)
        planet = Obj(SPHERE, gas_shader, 0.85, rot=R, glow=((0.3, 0.55, 1.0), 0.05, 0.8), name="gas")
        ring = Obj(RINGS, ring_shader, 0.85, rot=R, name="ring")
        return Scene([planet, ring], (7, 4, 6), yaw=0.0, pitch=0.12, dist=5.6)
    if name == "system":
        sun = Obj(SPHERE, star_shader, 1.5, glow=((1.0, 0.55, 0.12), 0.07, 1.2), name="sun")
        a, b = 0.6 + t * 0.12, 3.6 + t * 0.07
        rp = np.array([4.3 * np.cos(a), 0, 4.3 * np.sin(a)])
        gp = np.array([8.2 * np.cos(b), 0, 8.2 * np.sin(b)])
        planet = Obj(SPHERE, rocky_shader, 0.55, pos=rp, rot=rot_z(0.25), glow=((0.35, 0.9, 1.0), 0.03, 0.7), name="rocky")
        m = t * 0.5
        moon = Obj(SPHERE, moon_shader, 0.15, pos=rp + 1.0 * np.array([np.cos(m), 0.2 * np.sin(m), np.sin(m)]), name="moon")
        R = rot_z(0.40) @ rot_x(0.30)
        gas = Obj(SPHERE, gas_shader, 0.95, pos=gp, rot=R, glow=((0.3, 0.55, 1.0), 0.03, 0.6), name="gas")
        ring = Obj(RINGS, ring_shader, 0.95, pos=gp, rot=R, name="ring")
        return Scene([sun, planet, moon, gas, ring], (0, 0, 0), yaw=0.0, pitch=0.55, dist=19.0)
    raise SystemExit("shader desconocido: " + name)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader", default=None, help="star | rocky | gas | system (o 1-4)")
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--size", default=None, help="WxH")
    ap.add_argument("--ssaa", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=None)
    a = ap.parse_args()
    name = NAMES.get(a.shader, a.shader)
    if a.out or a.gif:
        from PIL import Image
        W, H = map(int, (a.size or "640x480").split("x"))
        def frame(t):
            sc = build(name, t)
            if a.yaw is not None: sc.yaw = a.yaw
            if a.pitch is not None: sc.pitch = a.pitch
            return render(sc, W, H, t, a.ssaa)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            Image.fromarray(frame(a.time)).save(a.out); print("guardado", a.out)
        if a.gif:
            os.makedirs(os.path.dirname(os.path.abspath(a.gif)), exist_ok=True)
            fr = [Image.fromarray(frame(a.time + i / a.fps)) for i in range(a.frames)]
            fr[0].save(a.gif, save_all=True, append_images=fr[1:], duration=int(1000 / a.fps), loop=0)
            print("guardado", a.gif)
        return
    interactive(name or "rocky", a)


def interactive(name, a):
    import pygame
    W, H = (map(int, a.size.split("x")) if a.size else (400, 300))
    pygame.init()
    screen = pygame.display.set_mode((W * 2, H * 2))
    clock = pygame.time.Clock()
    t, paused = 0.0, False
    sc = build(name, t)
    yaw, pitch, dist = sc.yaw, sc.pitch, sc.dist
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return
            if e.type == pygame.KEYDOWN:
                if e.unicode in NAMES:
                    name = NAMES[e.unicode]; sc = build(name, t); yaw, pitch, dist = sc.yaw, sc.pitch, sc.dist
                elif e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_s:
                    from PIL import Image
                    os.makedirs("screenshots", exist_ok=True)
                    Image.fromarray(render(sc, 640, 480, t, 2)).save(f"screenshots/{name}_{int(time.time())}.png")
                elif e.key == pygame.K_ESCAPE: return
        k = pygame.key.get_pressed()
        yaw += (k[pygame.K_RIGHT] - k[pygame.K_LEFT]) * 0.05
        pitch = float(np.clip(pitch + (k[pygame.K_UP] - k[pygame.K_DOWN]) * 0.04, -1.4, 1.4))
        dist *= 1 + (k[pygame.K_MINUS] - k[pygame.K_EQUALS]) * 0.03
        sc = build(name, t); sc.yaw, sc.pitch, sc.dist = yaw, pitch, dist
        img = render(sc, W, H, t, 1)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (W * 2, H * 2)), (0, 0))
        pygame.display.set_caption(f"{name}  t={t:.1f}  [1-4 shader | flechas | +/- | espacio | S captura]")
        pygame.display.flip()
        dt = clock.tick(30) / 1000
        if not paused: t += max(dt, 0.1)


if __name__ == "__main__":
    cli()
