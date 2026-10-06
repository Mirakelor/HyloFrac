"""Scene loading with the HyloFrac evaluation protocol.

A packaged scene directory (docs/benchmark.md, section 6) is loaded into a
:class:`SceneSample` that provides, per fragment:

- an observation point cloud: 1,000 uniformly sampled surface points of the
  assembled fragment, centered (centroid subtracted) and rotated by a
  uniform random SO(3) rotation;
- the ground-truth translation, defined as the sampled-point centroid;
- the ground-truth rotation, the inverse of the random observation rotation
  (applying GT pose to the observation restores the assembled pose);
- the fragment mesh volume (used as the weight in Qpos / adjacency metrics).

Fragments are stored in local coordinates; the assembled pose comes from
the per-fragment SE(3) of ``gt/transforms.json``. The point sampling and
rotations are seeded from the scene id so that every evaluation run
reproduces the same numbers.

The loader also implements the T3 robustness degradations (docs/benchmark.md,
section 1) as independent options on :class:`SceneSample`:

- ``missing_frac`` removes that fraction of the smallest fragments
  (by surface area), keeping at least two, and prunes names / poses /
  adjacency accordingly;
- ``erosion_depth`` displaces surface vertices inward along their normals
  by that depth plus uniform noise of the same magnitude, before sampling
  (GT poses and adjacency are unchanged);
- ``foreign_fragments`` injects external fragment meshes as distractors:
  they appear in :attr:`all_pcs` (the method input) but carry no GT entry.

Each fragment's sampling seed derives from the scene id and the fragment
name, so the intact observations of a scene are identical across
degradations (T3a keeps the very same per-fragment clouds as T1).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _scene_seed(scene_id: str, offset: int = 0) -> np.random.Generator:
    digest = hashlib.sha256(scene_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") ^ offset
    return np.random.default_rng(seed)


def sample_surface(mesh: trimesh.Trimesh, n_points: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample ``n_points`` surface points from a mesh."""
    points, _ = trimesh.sample.sample_surface(
        mesh, n_points, seed=int(rng.integers(0, 2**31)))
    return np.asarray(points, dtype=np.float64)


def uniform_rotation(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly random rotation from SO(3) via quaternion sampling."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
        np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
        np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
        np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def erode_mesh(mesh: trimesh.Trimesh, depth: float,
               rng: np.random.Generator) -> trimesh.Trimesh:
    """Inward displacement of the surface vertices along their normals by
    ``depth`` plus uniform noise of the same magnitude (T3b)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    noise = rng.uniform(-depth, depth, size=verts.shape)
    eroded = trimesh.Trimesh(vertices=verts + normals * (depth + noise),
                             faces=mesh.faces, process=False)
    return eroded


def anchor_index(vols=None, areas=None) -> int:
    """Index of the anchor fragment that defines the GT frame.

    ``docs/benchmark.md`` section 4: the reference geometric baseline takes
    "fragment 0 ... as the anchor and ... submitted with the identity pose",
    and the submission format is defined in that frame; the GT poses
    therefore live in fragment 0's observation frame and the anchor's own GT
    pose is the identity.  Index 0 is stable under the T3 degradations
    (fragments keep their names and the missing-fragment degradation removes
    the *smallest* fragments first).

    Training and evaluation must agree on this index: it defines the frame
    the GT poses live in.
    """
    return 0


def select_foreign_fragments(data_root: str, split: str, part: str,
                             scene_id: str, n: int,
                             rng: np.random.Generator
                             ) -> List[Tuple[str, str]]:
    """Pick ``n`` distractor fragments for T3c: random fragments from
    random scenes of the same morphology and split but of a *different
    specimen* than the query scene. Returns [(obj_path, name)]."""
    import re

    specimen = scene_id.split("_v")[0]
    pool: List[str] = []
    for sid in sorted(os.listdir(os.path.join(data_root, split))):
        if sid == specimen or not re.fullmatch(r"\d{6,9}", sid):
            continue
        meta_path = os.path.join(data_root, split, sid, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if str(meta.get("part", "")).lower() != part.lower():
            continue
        frag_dir = os.path.join(data_root, split, sid, "fragments")
        frags = sorted(f for f in os.listdir(frag_dir)
                       if f.startswith("frag_") and f.endswith(".obj")) \
            if os.path.isdir(frag_dir) else []
        for f in frags:
            pool.append((os.path.join(frag_dir, f), f"{sid}_{f[:-4]}"))
    if not pool:
        return []
    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[int(i)] for i in chosen]


def gt_poses(scene_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-fragment (R, t) from gt/transforms.json, mapping fragment-local
    to assembled world coordinates."""
    with open(os.path.join(scene_dir, "gt", "transforms.json"),
              "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for frag_id, entry in gt.get("fragments", {}).items():
        poses[frag_id] = (np.asarray(entry["R"], dtype=np.float64),
                          np.asarray(entry["t"], dtype=np.float64))
    return poses


def world_points(mesh: trimesh.Trimesh, rmat: np.ndarray, trans: np.ndarray,
                 n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform surface points of a mesh under the rigid pose (rmat, trans).

    Points are sampled in the local frame and moved by the pose, which is
    distributionally identical to sampling the posed mesh.
    """
    local = sample_surface(mesh, n_points, rng)
    return local @ rmat.T + trans


def _load_gt_adjacency(scene_dir: str, names: List[str]
                       ) -> Optional[Tuple[Dict[str, set], Dict[Tuple[str, str], float]]]:
    """Read the GT adjacency graph from gt/adjacency.json, if present."""
    adj_path = os.path.join(scene_dir, "gt", "adjacency.json")
    if not os.path.exists(adj_path):
        return None
    with open(adj_path, "r", encoding="utf-8") as fh:
        adj = json.load(fh)
    edges = adj.get("edges", [])
    if not edges:
        return None
    adjacency: Dict[str, set] = {name: set() for name in names}
    areas: Dict[Tuple[str, str], float] = {}
    for edge in edges:
        a = f"frag_{edge['frag_i']:02d}"
        b = f"frag_{edge['frag_j']:02d}"
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
            areas[(a, b)] = float(edge.get("area", 0.0))
    return adjacency, areas


def derive_adjacency(pcs: np.ndarray, rmat: np.ndarray, trans: np.ndarray,
                     names: List[str], eps: float = 0.1,
                     min_frac: float = 0.05) -> Dict[str, set]:
    """Adjacency graph derived from an assembly (posed point clouds).

    Two fragments are adjacent when more than ``min_frac`` of one fragment's
    points lie within ``eps`` of the other fragment. ``eps`` must exceed the
    sampling spacing of the point clouds (about sqrt(area / n_points)).
    Used as the fallback GT graph and to derive adjacency predictions.
    """
    n = len(pcs)
    world = [pcs[i] @ rmat[i].T + trans[i] for i in range(n)]
    adjacency: Dict[str, set] = {name: set() for name in names}
    for i in range(n):
        for j in range(i + 1, n):
            dist, _ = cKDTree(world[j]).query(world[i])
            frac = float((dist < eps).mean())
            if frac > min_frac:
                adjacency[names[i]].add(names[j])
                adjacency[names[j]].add(names[i])
    return adjacency


class SceneSample:
    """One packaged scene under the evaluation protocol.

    Optional T3 degradations (docs/benchmark.md, section 1): ``missing_frac``
    (fraction of the smallest fragments removed, at least two kept),
    ``erosion_depth`` (surface erosion before sampling; GT unchanged) and
    ``foreign_fragments`` (list of (mesh_path, name) distractors exposed
    through :attr:`all_pcs` without any GT entry). Per-fragment sampling
    seeds derive from the scene id and fragment name, so intact per-fragment
    observations are identical across degradations.
    """

    def __init__(self, scene_dir: str, scene_id: str, n_points: int = 1000,
                 seed_offset: int = 0, missing_frac: float = 0.0,
                 erosion_depth: float = 0.0,
                 foreign_fragments: Optional[List[Tuple[str, str]]] = None,
                 gt_frame: str = "anchor"
                 ) -> None:
        """``gt_frame`` selects the frame the GT poses live in.

        ``"anchor"`` (default, protocol): the GT pose of the anchor fragment
        (fragment 0) is the identity and every other fragment carries its
        pose relative to the anchor's observation frame.  This is the frame
        the reference geometric baseline submits in ("fragment 0 is the
        anchor ... submitted with the identity pose") and the frame GARF is
        given through its anchored reference fragment; with the random
        per-fragment observation rotation of this loader it is the only
        frame in which PA / RMSE(R) / RMSE(T) / Sym Chamfer are well posed
        (the canonical frame differs from it by an unobservable random
        rotation, which pins every method at chance level).  Qpos is a
        separate matter: it rigidly aligns on the largest-volume fragment
        (``metrics.qpos``), which need not be fragment 0.

        ``"absolute"`` keeps the pre-2026-09-15 behaviour: the GT pose maps
        the observation back into the canonical assembly frame (i.e. the
        inverse of the fragment's own observation rotation).  It is kept
        only to reproduce submissions produced before the protocol fix.
        """
        if not os.path.isdir(scene_dir):
            raise FileNotFoundError(f"scene directory not found: {scene_dir}")
        if gt_frame not in ("anchor", "absolute"):
            raise ValueError(f"gt_frame must be 'anchor' or 'absolute', "
                             f"got {gt_frame!r}")
        self.gt_frame = gt_frame
        frag_dir = os.path.join(scene_dir, "fragments")
        files = sorted(f for f in os.listdir(frag_dir)
                       if f.startswith("frag_") and f.endswith(".obj"))
        if not files:
            raise ValueError(f"no fragments in {scene_dir}")
        poses = gt_poses(scene_dir)

        frags = []  # (name, mesh, rmat, trans), original order
        for fname in files:
            name = fname[:-4]
            mesh = trimesh.load(os.path.join(frag_dir, fname), process=False)
            if mesh is None or len(mesh.faces) == 0:
                raise ValueError(f"bad fragment mesh {fname}")
            frags.append((name, mesh, *poses[name]))

        if missing_frac > 0.0 and len(frags) >= 3:
            n_remove = min(max(1, round(missing_frac * len(frags))),
                           len(frags) - 2)
            frags.sort(key=lambda f: f[1].area)  # smallest first
            frags = frags[n_remove:]
            frags.sort(key=lambda f: f[0])       # keep name order

        self.names: List[str] = []
        self.pcs: List[np.ndarray] = []
        self.gt_R: List[np.ndarray] = []
        self.gt_t: List[np.ndarray] = []
        self.vols: List[float] = []
        self.areas: List[float] = []
        rot_obs_list: List[np.ndarray] = []
        centroid_list: List[np.ndarray] = []
        for name, mesh, rmat, trans in frags:
            self.names.append(name)
            self.vols.append(abs(float(mesh.volume)) if mesh.is_watertight
                             else 0.0)
            self.areas.append(float(mesh.area))
            frng = _scene_seed(f"{scene_id}|{name}", seed_offset)
            if erosion_depth > 0.0:
                erng = _scene_seed(f"{scene_id}|{name}|erode{erosion_depth}",
                                   seed_offset)
                mesh = erode_mesh(mesh, erosion_depth, erng)
            world = world_points(mesh, rmat, trans, n_points, frng)
            centroid = world.mean(axis=0)
            local = world - centroid
            rot_obs = uniform_rotation(frng)
            self.pcs.append(local @ rot_obs.T)
            self.gt_R.append(rot_obs.T)
            self.gt_t.append(centroid)
            rot_obs_list.append(rot_obs)
            centroid_list.append(centroid)
        self.pcs = np.stack(self.pcs)  # [P, N, 3]
        self.gt_R = np.stack(self.gt_R)
        self.gt_t = np.stack(self.gt_t)
        self.vols = np.asarray(self.vols)
        self.areas = np.asarray(self.areas)

        # anchor = fragment 0, which defines the GT frame (see
        # anchor_index). Qpos separately aligns on the largest-volume
        # fragment; see metrics.qpos -- the two need not coincide.
        self.anchor = anchor_index(self.vols, self.areas)
        if self.gt_frame == "anchor":
            rot_obs = np.stack(rot_obs_list)          # observation operators
            centroids = np.stack(centroid_list)
            r_a = rot_obs[self.anchor]                 # anchor operator
            # GT_i = (R_a R_i^T, R_a (c_i - c_a)): applying it to fragment i's
            # centred, rotated observation places the fragment into the
            # assembly as seen in the anchor's observation frame, so the
            # anchor itself is the identity pose.
            # (R_a R_i^T) as the matrix whose transpose is the operator
            self.gt_R = np.einsum("ij,pkj->pik", r_a, rot_obs)
            self.gt_t = (centroids - centroids[self.anchor]) @ r_a.T

        # foreign distractors (T3c): sampled like intact fragments, no GT
        self.foreign_pcs = np.zeros((0, n_points, 3))
        for fpath, fname in (foreign_fragments or []):
            fmesh = trimesh.load(fpath, process=False)
            if fmesh is None or len(fmesh.faces) == 0:
                raise ValueError(f"bad foreign fragment mesh {fpath}")
            frng = _scene_seed(f"{scene_id}|{fname}", seed_offset)
            local = sample_surface(fmesh, n_points, frng)
            local = local - local.mean(axis=0)
            rot_obs = uniform_rotation(frng)
            self.foreign_pcs = np.concatenate(
                [self.foreign_pcs, (local @ rot_obs.T)[None]], axis=0)
        self.n_foreign = len(self.foreign_pcs)

        # GT adjacency from the annotation file; geometric fallback if absent.
        gt_adj = _load_gt_adjacency(scene_dir, self.names)
        if gt_adj is not None:
            self.adjacency, self.adj_areas = gt_adj
        else:
            self.adjacency = derive_adjacency(self.pcs, self.gt_R, self.gt_t,
                                              self.names)
            self.adj_areas = {}

        self.metadata: Dict[str, object] = {}
        meta_path = os.path.join(scene_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)

    @property
    def all_pcs(self) -> np.ndarray:
        """Method input: the GT fragment clouds followed by any foreign
        distractors (``np.concatenate([pcs, foreign_pcs])``)."""
        if self.n_foreign:
            return np.concatenate([self.pcs, self.foreign_pcs], axis=0)
        return self.pcs

    @property
    def n_parts(self) -> int:
        return len(self.names)

    @property
    def part(self) -> str:
        """Morphology tag from metadata (cranium | mandible | unknown)."""
        raw = str(self.metadata.get("part", ""))
        low = raw.lower()
        if "mandible" in low:
            return "mandible"
        if "cranium" in low or "skull" in low:
            return "cranium"
        return "unknown"
