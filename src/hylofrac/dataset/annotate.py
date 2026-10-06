"""Ground-truth annotation: adjacency graph and contact surfaces.

The fragments are exported by the simulator in local coordinates with
their center of mass at the origin; the assembly frame of transforms.json
holds every fragment in the pose it was born in (the intact body is
replaced by all fragments in a single step, before the post-fracture
dynamics separates them). In that frame adjacent fragments touch along
their fracture interfaces, so the annotation detects the interfaces
directly on the fragment meshes::

    1. Reconstruct the assembled fragment meshes (assembly-frame SE(3),
       see hylofrac.dataset.export).
    2. For every fragment pair with intersecting bounding boxes, collect
       the faces whose three vertices all lie within ``eps`` of the other
       fragment (interface faces touch after the mesh resolution).
    3. The interface area of the pair is the smaller of the two sides;
       pairs above ``min_area`` are adjacent.
    4. Write ``gt/adjacency.json`` (edges with contact area) and one OBJ
       per adjacent pair into ``contact_surfaces/`` (one side of the
       interface, in the assembly frame).

Usage::

    hylofrac-annotate --scenes <raw_scene_dirs> --out dataset/hylofrac \\
        [--eps 0.02] [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from hylofrac.dataset.export import VARIANT_RE, assembly_frame_poses

FRAGMENT_RE = re.compile(r"^specimen_C\d+\.obj$")

MIN_AREA = 1e-4


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"no usable vertices in {path}")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError(f"no faces in {path}")
    return mesh


def assembled_fragments(output_dir: str, poses: Dict[str, Dict[str, object]]
                        ) -> List[trimesh.Trimesh]:
    """Load the local fragment meshes and return them in the assembly
    pose (world coordinates of the assembly frame)."""
    frags = sorted(f for f in os.listdir(output_dir) if FRAGMENT_RE.match(f))
    meshes: List[trimesh.Trimesh] = []
    for fname in frags:
        name = fname[:-4]
        pose = poses.get(name)
        if pose is None:
            raise ValueError(f"no assembly-frame pose for {name}")
        mesh = load_mesh(os.path.join(output_dir, fname))
        rmat = np.asarray(pose["R"], dtype=np.float64)
        trans = np.asarray(pose["t"], dtype=np.float64)
        matrix = np.eye(4)
        matrix[:3, :3] = rmat
        matrix[:3, 3] = trans
        mesh.apply_transform(matrix)
        meshes.append(mesh)
    return meshes


def _face_areas(verts: np.ndarray, faces: np.ndarray,
                sel: np.ndarray) -> float:
    tri = verts[faces[sel]]
    return float(np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1).sum() / 2.0)


def touching_interfaces(meshes: List[trimesh.Trimesh], eps: float,
                        min_area: float = MIN_AREA
                        ) -> List[Dict[str, object]]:
    """Detect the touching interfaces between assembled fragments.

    A face of fragment i is an interface face when all its three vertices
    lie within ``eps`` of fragment j; the pair is adjacent when both
    sides of the interface exceed ``min_area`` (the reported area is the
    smaller side, which rejects spurious touches along the outer-surface
    boundary lines).
    """
    n = len(meshes)
    verts = [np.asarray(m.vertices, dtype=np.float64) for m in meshes]
    faces = [np.asarray(m.faces) for m in meshes]
    bounds = [np.asarray(m.bounds) for m in meshes]
    trees = [cKDTree(v) for v in verts]

    edges: List[Dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (bounds[i][0] <= bounds[j][1] + eps).all() or \
                    not (bounds[j][0] <= bounds[i][1] + eps).all():
                continue
            d, _ = trees[j].query(verts[i])
            sel_i = (d < eps)[faces[i]].all(axis=1)
            if not sel_i.any():
                continue
            d, _ = trees[i].query(verts[j])
            sel_j = (d < eps)[faces[j]].all(axis=1)
            if not sel_j.any():
                continue
            area = min(_face_areas(verts[i], faces[i], sel_i),
                       _face_areas(verts[j], faces[j], sel_j))
            if area <= min_area:
                continue
            edges.append({"i": i, "j": j, "area": area,
                          "faces_i": faces[i][sel_i]})
    return edges


def annotate_scene(scene_dir: str, dest_dir: str, eps: float,
                   min_area: float = MIN_AREA) -> Dict[str, object]:
    """Annotate one scene into an already exported scene directory.

    ``dest_dir`` is the packaged scene directory
    (<out>/<split>/<specimen>/<variant>) created by hylofrac-export; the
    adjacency graph and contact surfaces are written next to its gt files.
    """
    sid = os.path.basename(scene_dir.rstrip(os.sep))
    output_dir = os.path.join(scene_dir, "output")
    frags = sorted(f for f in os.listdir(output_dir) if FRAGMENT_RE.match(f))
    _, poses = assembly_frame_poses(
        os.path.join(output_dir, "transforms.json"), [f[:-4] for f in frags])
    meshes = assembled_fragments(output_dir, poses)

    gt_dir = os.path.join(dest_dir, "gt")
    cs_dir = os.path.join(gt_dir, "contact_surfaces")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(cs_dir, exist_ok=True)

    adjacency: Dict[str, object] = {
        "scene": sid, "eps": eps, "min_area": min_area,
        "n_fragments": len(meshes), "edges": [],
    }
    for edge in touching_interfaces(meshes, eps, min_area):
        i, j = edge["i"], edge["j"]
        contact_name = f"contact_{i:02d}_{j:02d}.obj"
        sub = trimesh.Trimesh(
            vertices=np.asarray(meshes[i].vertices, dtype=np.float64),
            faces=edge["faces_i"], process=False)
        sub.export(os.path.join(cs_dir, contact_name))
        adjacency["edges"].append({
            "frag_i": i, "frag_j": j,
            "source_i": frags[i][:-4], "source_j": frags[j][:-4],
            "area": edge["area"],
            "n_faces": len(edge["faces_i"]),
            "contact_mesh": contact_name,
        })
    with open(os.path.join(gt_dir, "adjacency.json"), "w", encoding="utf-8") as fh:
        json.dump(adjacency, fh, indent=2)
    return adjacency


def scene_dest_dir(out_root: str, split_map: Dict[str, str], scene_id: str) -> str:
    """Locate the packaged scene directory for a scene id."""
    m = VARIANT_RE.match(scene_id)
    if not m:
        raise ValueError(f"scene id does not match <specimen>_v<variant>: {scene_id}")
    split = split_map.get(m.group("specimen"), "val")
    return os.path.join(out_root, split, m.group("specimen"), f"v{m.group('variant')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate adjacency graphs and contact surfaces from "
                    "the assembled fragment interfaces.")
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="raw scene directories to annotate (batch outputs)")
    parser.add_argument("--out", required=True,
                        help="dataset root with exported scenes (hylofrac-export)")
    parser.add_argument("--split-file", default="",
                        help="JSON mapping specimen id -> train/val/test (same as export)")
    parser.add_argument("--eps", type=float, default=0.02,
                        help="vertex-to-fragment distance threshold (normalized units)")
    parser.add_argument("--resume", action="store_true",
                        help="skip scenes with an existing adjacency.json")
    args = parser.parse_args()

    split_map: Dict[str, str] = {}
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file, "r", encoding="utf-8") as fh:
            split_map = json.load(fh)

    n_ok = 0
    for scene_dir in args.scenes:
        sid = os.path.basename(scene_dir.rstrip(os.sep))
        m = VARIANT_RE.match(sid)
        if not m:
            print(f"[skip] bad scene id: {sid}", flush=True)
            continue
        dest = scene_dest_dir(args.out, split_map, sid)
        if args.resume and os.path.exists(os.path.join(dest, "gt", "adjacency.json")):
            n_ok += 1
            continue
        try:
            adjacency = annotate_scene(scene_dir, dest, args.eps)
            n_ok += 1
            print(f"[ok] {sid}: {len(adjacency['edges'])} edges "
                  f"-> {dest}", flush=True)
        except Exception as exc:
            print(f"[error] {sid}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[done] scenes={n_ok}/{len(args.scenes)}", flush=True)


if __name__ == "__main__":
    main()
