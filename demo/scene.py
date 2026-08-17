"""The interactive demo's terrain — realistic, varied Mars ground: rolling hills and hollows, not a
wall in front of the rover. The sample sits across a stretch of steeper ground, so the planner has a
real choice: cross the steep slope (short, riskier) or arc over gentler ground (longer, safer).

The terrain is fully procedural (a seed), and the planner only ever sees it through the rover's
`sensed_dem()` — so this is a genuine navigation test, not a route hand-fed to the model.
"""
import numpy as np

from world import dem

TERRAIN_SEED = 7
RELIEF_M = 0.4                     # the gentle, drivable rolling base
GOAL = (0.0, 3.0)
TARGETS = [("outcrop", 0.0, 3.0, 0.90, "carbonate")]

# A localized steep soft-sand mound between the rover and the sample — the kind of trap that stranded
# Opportunity/Spirit. It's ONE feature set into varied terrain, not a wall on a flat plane.
_MOUND = {"x": 0.0, "y": 1.5, "amp": 0.9, "sx": 0.75, "sy": 0.55}


def terrain(n: int = 64):
    base = dem.rolling(n=n, seed=TERRAIN_SEED, relief_m=RELIEF_M)
    r = 5.0
    xs = np.linspace(-r, r, n)
    xx, yy = np.meshgrid(xs, xs)
    m = _MOUND
    mound = m["amp"] * np.exp(-(((xx - m["x"]) ** 2) / (2 * m["sx"] ** 2)
                              + ((yy - m["y"]) ** 2) / (2 * m["sy"] ** 2)))
    return (base + mound).astype(np.float64)
