"""Prepare the Perseverance visual mesh (world/assets/perseverance.obj) used for rendering.

Provenance: NASA's public-domain Perseverance model from the NASA 3D Resources collection
(github.com/nasa/NASA-3D-Resources — "Mars Perseverance Rover"). The distributed GLB is
Draco-compressed; we decompress it once with `@gltf-transform/cli` (npm) into
world/assets/perseverance_dec.glb, then this script bakes the scene-graph transforms, orients it
Z-up, sits it on z=0, and decimates it to ~30k faces so MuJoCo loads it quickly.

The decimated OBJ is committed (~1.3 MB) so the render is reproducible without the raw assets.
Re-run only if you want to regenerate it:

    gltf-transform cp world/assets/perseverance.glb world/assets/perseverance_dec.glb  # decompress
    .venv/bin/python -m scripts.prepare_rover_mesh
"""
from __future__ import annotations

import numpy as np
import trimesh

SRC = "world/assets/perseverance_dec.glb"
OUT = "world/assets/perseverance.obj"
KEEP_REDUCTION = 0.85     # remove 85% of faces → ~30k, plenty for a background render


def main() -> None:
    mesh = trimesh.load(SRC, force="mesh")                       # bake node transforms → one mesh
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # Y-up→Z-up
    mesh.apply_translation([-mesh.centroid[0], -mesh.centroid[1], -mesh.bounds[0][2]])   # center, base z=0
    mesh = mesh.simplify_quadric_decimation(KEEP_REDUCTION)
    mesh.export(OUT)
    print(f"wrote {OUT}: {len(mesh.faces)} faces, extents {np.round(mesh.extents, 2)}")


if __name__ == "__main__":
    main()
