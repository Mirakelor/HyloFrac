"""Lightweight OBJ mesh IO built on NumPy.

The HyloFrac pipeline deals with meshes of up to several million vertices.
Reading and writing plain OBJ with NumPy keeps peak memory low and avoids
heavy dependencies in the dataset-construction path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def read_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read an OBJ file.

    Returns (vertices, faces) with float64 vertices [N, 3] and int64
    zero-based triangle faces [M, 3]. Polygonal faces are triangulated by
    fan. Non-triangle primitives and vertex attributes are ignored.
    """
    vs: list[list[float]] = []
    fs: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f" and len(parts) >= 4:
                idx = []
                for tok in parts[1:]:
                    vi = int(tok.split("/")[0])
                    idx.append(vi - 1 if vi > 0 else len(vs) + vi)
                for k in range(1, len(idx) - 1):
                    fs.append([idx[0], idx[k], idx[k + 1]])
    if not vs:
        raise ValueError(f"no vertices in {path}")
    verts = np.asarray(vs, dtype=np.float64)
    faces = np.asarray(fs, dtype=np.int64) if fs else np.zeros((0, 3), dtype=np.int64)
    return verts, faces


def count_vertices(path: str) -> int:
    """Count vertex records without materializing the mesh."""
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                n += 1
    return n


def write_obj(path: str, verts: np.ndarray, faces: Optional[np.ndarray] = None,
              precision: int = 9) -> None:
    """Write an OBJ file with all vertices and (optionally) faces.

    Faces use one-based indices; a face row is emitted only when faces are
    provided, so the file is always a complete mesh.
    """
    v = np.asarray(verts, dtype=np.float64)
    fmt = f"v {{:.{precision}g}} {{:.{precision}g}} {{:.{precision}g}}\n"
    with open(path, "w", encoding="utf-8") as fh:
        for row in v:
            fh.write(fmt.format(row[0], row[1], row[2]))
        if faces is not None:
            f = np.asarray(faces, dtype=np.int64)
            for row in f:
                fh.write(f"f {row[0] + 1} {row[1] + 1} {row[2] + 1}\n")


def apply_rigid(verts: np.ndarray, rmat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Map local points to world: v_world = R @ v_local + t.

    Both the matrix convention (rows are x/y/z basis vectors such that
    ``v @ R.T + t`` applies the rotation) and the result are validated.
    """
    v = np.asarray(verts, dtype=np.float64)
    r = np.asarray(rmat, dtype=np.float64)
    t = np.asarray(trans, dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation length 3")
    return v @ r.T + t
