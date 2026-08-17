"""A* route search over an elevation cost map — the rover's onboard navigator.

The terrain height grid (from the real DEM) becomes a traversability cost map: crossing between two
cells costs the distance times a penalty that rises steeply with the slope between them. Steep ground
is penalised whether it goes UP (a dune) or DOWN (a drop into a hole) — both can bog or topple a
rover. A* then finds the least-cost path.

Two cost policies expose the whole point of the planner:
  - "shortest": distance only, ignores terrain — it will cut straight across a dune.
  - "safe":     distance x slope-penalty — it routes around the soft, steep ground.
The surrogate ("lightsim") then checks each path's tail risk under uncertainty, and Granite decides.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

R = 5.0                         # world spans [-R, R] (matches world/sim.py TERRAIN_RADIUS)
SLOPE_REF_DEG = 10.0            # slope where the penalty reaches ~1x the caution weight
COST_POWER = 2.5               # penalty grows faster than linearly: gentle rises stay cheap, steep
                               # dunes/holes get expensive — so a little climb can beat a long detour
INFLATE_M = 0.45               # light clearance margin (obstacle inflation)

# Caution levels A* searches — each is a slope-cost weight. 0 = shortest (ignore terrain); higher =
# more willing to detour around slope. Granite picks among the resulting routes by distance vs risk.
CAUTION = {"direct": 0.0, "balanced": 1.5, "cautious": 4.0, "safe": 9.0}
_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _hazard_field(terrain: np.ndarray, mpc: float) -> np.ndarray:
    """Per-cell slope (deg), dilated by INFLATE_M so cells NEAR steep ground also read as costly —
    this is what makes A* leave a clearance margin instead of scraping past a dune's edge.

    The height grid is smoothed first: the cost map cares about real relief (dunes, holes), not the
    centimetre surface noise that would otherwise make A* zig-zag into a jagged, undrivable path."""
    smooth = gaussian_filter(terrain, sigma=1.5)
    gy, gx = np.gradient(smooth, mpc)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    radius = max(1, round(INFLATE_M / mpc))
    return maximum_filter(slope_deg, size=2 * radius + 1)


def _cell_to_xy(i: int, j: int, n: int) -> tuple[float, float]:
    return (-R + j / (n - 1) * 2 * R, -R + i / (n - 1) * 2 * R)


def _xy_to_cell(xy, n: int) -> tuple[int, int]:
    x, y = xy
    j = int(round((x + R) / (2 * R) * (n - 1)))
    i = int(round((y + R) / (2 * R) * (n - 1)))
    return max(0, min(n - 1, i)), max(0, min(n - 1, j))


def _edge_cost(hazard, mpc, a, b, weight) -> float:
    (ai, aj), (bi, bj) = a, b
    dist = mpc * math.hypot(bi - ai, bj - aj)
    if weight <= 0.0:                           # "direct": distance only, ignore terrain
        return dist
    h = max(hazard[ai, aj], hazard[bi, bj])     # inflated slope near either cell
    return dist * (1.0 + weight * (h / SLOPE_REF_DEG) ** COST_POWER)


def _astar(n, hazard, mpc, start, goal, weight) -> list[tuple[int, int]]:
    h = lambda c: mpc * math.hypot(goal[0] - c[0], goal[1] - c[1])   # admissible (distance) heuristic
    open_heap = [(h(start), 0.0, start)]
    came, gscore = {start: None}, {start: 0.0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            break
        if g > gscore.get(cur, math.inf):
            continue
        ci, cj = cur
        for di, dj in _NB:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            ng = g + _edge_cost(hazard, mpc, cur, (ni, nj), weight)
            if ng < gscore.get((ni, nj), math.inf):
                gscore[(ni, nj)] = ng
                came[(ni, nj)] = cur
                heapq.heappush(open_heap, (ng + h((ni, nj)), ng, (ni, nj)))
    if goal not in came:
        return [start, goal]
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = came[c]
    return path[::-1]


def _simplify(cells, n, start_xy, goal_xy) -> list[tuple[float, float]]:
    """Drop near-collinear waypoints so the skid-steer drives a few clean legs, not every cell."""
    pts = [start_xy] + [_cell_to_xy(i, j, n) for i, j in cells] + [tuple(goal_xy)]
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        ax, ay = out[-1]
        bx, by = pts[k]
        cx, cy = pts[k + 1]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(cross) > 0.05:                      # keep only real corners
            out.append(pts[k])
    out.append(tuple(goal_xy))
    # de-dup consecutive near-identical points
    dedup = [out[0]]
    for p in out[1:]:
        if math.dist(p, dedup[-1]) > 0.15:
            dedup.append(p)
    return dedup


def _length(wps, start_xy) -> float:
    pos, d = start_xy, 0.0
    for wp in wps:
        d += math.dist(pos, wp)
        pos = wp
    return round(d, 2)


def plan_variants(start_xy, goal_xy, terrain: np.ndarray, meters_per_cell: float) -> list[dict]:
    """A* several candidate routes over the height grid, one per caution level (see CAUTION) — from
    the distance-only 'direct' line to the terrain-avoiding 'safe' route. Identical routes are merged
    (a small hill may not change the path; a real dune does). Returns an ordered list of
    {name, waypoints, length_m} for the lightsim to score and Granite to choose among."""
    n = terrain.shape[0]
    start_c = _xy_to_cell(start_xy, n)
    goal_c = _xy_to_cell(goal_xy, n)
    hazard = _hazard_field(terrain, meters_per_cell)
    out, seen = [], set()
    for name, weight in CAUTION.items():
        cells = _astar(n, hazard, meters_per_cell, start_c, goal_c, weight)
        wps = _simplify(cells, n, start_xy, goal_xy)
        key = tuple((round(x, 1), round(y, 1)) for x, y in wps)
        if key in seen:                          # same road as a less-cautious level — skip the dup
            continue
        seen.add(key)
        out.append({"name": name, "waypoints": wps, "length_m": _length(wps, start_xy)})
    return out
