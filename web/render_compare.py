"""Cinematic split-screen: the Earth-planned direct route vs MARVIN's Granite-chosen detour, on the
same Purgatory dune, filmed from the same wide 3/4 camera so the avoidance is unmistakable.

Left  — the orbital plan drives straight at the outcrop, into the dune, and stalls (like Opportunity).
Right — MARVIN takes the lightsim-verified detour south, rounds the dune, and caches the sample.

Writes web/vendor/compare.mp4. Run:  .venv/bin/python -m web.render_compare
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
import tempfile

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
from PIL import Image, ImageDraw, ImageFont

from demo.purgatory import GOAL, TARGETS, dune_terrain
from planner import route
from world.sim import MarsSim

PW, PH, FPS = 620, 640, 30           # per-panel width/height
SHOOT_EVERY = 3


def _font(size: int):
    for p in ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_side(waypoints, terrain, sample_end: bool) -> tuple[list, bool]:
    sim = MarsSim(seed=42, terrain=terrain, targets=TARGETS, render_mesh=True)
    r = mujoco.Renderer(sim.model, PH, PW)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [1.15, 0.0, 0.12]      # frame the whole scene: start, dune, and outcrop
    cam.distance, cam.elevation, cam.azimuth = 5.4, -43.0, 208.0
    frames = []

    def shoot():
        cam.azimuth += 0.12                # a slow, steady drift keeps it alive without distracting
        r.update_scene(sim.data, cam)
        frames.append(r.render().copy())

    ctr = [0]
    orig = sim.step
    def step(vL, vR, n=1):
        orig(vL, vR, n)
        ctr[0] += 1
        if ctr[0] % SHOOT_EVERY == 0:
            shoot()
    sim.step = step

    for _ in range(20):
        shoot()
    for wp in waypoints:
        sim.drive_to(*wp)
    x, y, _ = sim.pose()
    reached = math.hypot(x - GOAL[0], y - GOAL[1]) < 0.6
    if sample_end and reached:
        sim.sample("outcrop")
    for _ in range(26):
        shoot()
    return frames, reached


def label(img, title, sub, sub_color):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 66], fill=(13, 13, 13, 255))
    d.text((18, 12), title, font=_font(21), fill=(244, 244, 244))
    d.text((18, 40), sub, font=_font(15), fill=sub_color)


def compose(left, right, l_reached, r_reached) -> list:
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))       # hold the last frame so both panels end together
    right += [right[-1]] * (n - len(right))
    gap, out = 4, []
    red, green, grey = (250, 77, 86), (66, 190, 101), (141, 141, 141)
    for i in range(n):
        canvas = Image.new("RGB", (PW * 2 + gap, PH), (13, 13, 13))
        canvas.paste(Image.fromarray(left[i]), (0, 0))
        canvas.paste(Image.fromarray(right[i]), (PW + gap, 0))
        # panels get their labels drawn onto copies (cheap enough at this scale)
        lp = canvas.crop((0, 0, PW, PH))
        rp = canvas.crop((PW + gap, 0, PW * 2 + gap, PH))
        near_end = i > n - 40
        label(lp, "EARTH-PLANNED", "STUCK AT DUNE" if near_end else "direct route from orbit",
              red if near_end else grey)
        label(rp, "MARVIN", "SAMPLE CACHED" if (near_end and r_reached) else "Granite's verified detour",
              green if near_end else grey)
        canvas.paste(lp, (0, 0))
        canvas.paste(rp, (PW + gap, 0))
        out.append(canvas)
    return out


def run():
    terrain = dune_terrain()
    routes = route.candidate_routes((0.0, 0.0), GOAL)   # deterministic; Granite picks 'detour south'
    left, lr = render_side(routes["direct"], terrain, sample_end=False)
    right, rr = render_side(routes["detour south"], terrain, sample_end=True)
    frames = compose(left, right, lr, rr)

    tmp = tempfile.mkdtemp()
    for i, f in enumerate(frames):
        f.save(f"{tmp}/f{i:05d}.png")
    os.makedirs("web/vendor", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "web/vendor/compare.mp4"], check=True)
    for p in glob.glob(f"{tmp}/*.png"):
        os.remove(p)
    print(f"left reached={lr}  right reached={rr}  frames={len(frames)}  "
          f"mp4 {os.path.getsize('web/vendor/compare.mp4') // 1024} KB")


if __name__ == "__main__":
    run()
