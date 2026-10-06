"""Export raw simulator scenes into the packaged HyloFrac layout.

A raw scene directory (as produced by the fracture batch runner) contains::

    <scene>/input/specimen.obj        complete mesh in the canonical pose
    <scene>/output/specimen_C*.obj    fragments in LOCAL coordinates
                                      (the simulator exports each fragment
                                      mesh in its own frame, centroid at the
                                      origin; see RigidFractureLab main.cpp
                                      ``expressMeshInLocalFrame``)
    <scene>/output/transforms.json    per-frame rigid-body SE(3) records
    <scene>/result.json               batch bookkeeping

This tool packages each finished scene into the released layout
(docs/benchmark.md, section 6)::

    <out>/<split>/<specimen>/<variant>/
        intact.obj                    complete mesh (copy of input)
        fragments/frag_XX.obj         fragment meshes in local coordinates
        gt/transforms.json            per-fragment SE(3) of the assembly
                                      frame (all fragments born in place)
        metadata.json                 bookkeeping fields of the scene and
                                      the per-object provenance

Fragments are stored once, in local coordinates; the assembled pose is
v_world = R @ v_local + t with (R, t) from gt/transforms.json.

Usage::

    hylofrac-export --batchs batch0 batch1 ... --out dataset/hylofrac \
        [--split-file splits.json] [--part-file parts.json] \
        [--provenance-file provenance.json] [--resume]

``--split-file`` maps a specimen id to a split name ("train" | "val" |
"test"); scenes whose specimen is absent default to "val".  ``--part-file``
maps a specimen id to its morphology ("cranium" | "mandible") and
``--provenance-file`` maps a specimen id to the per-object provenance
recorded in ``metadata.json`` (source archive, media id, specimen title,
taxonomy, rights holder, copyright statement and reuse terms).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np

from hylofrac.dataset.objio import count_vertices

VARIANT_RE = re.compile(r"^(?P<specimen>\d{6,9})_v(?P<variant>\d+)$")
FRAGMENT_RE = re.compile(r"^specimen_C\d+\.obj$")


def quat_to_matrix(q: List[float]) -> np.ndarray:
    """Rotation matrix from a unit quaternion [x, y, z, w] (scalar-last)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def assembly_frame_poses(transforms_path: str,
                         fragment_names: List[str]
                         ) -> tuple[int, Dict[str, Dict[str, object]]]:
    """(frame step, per-object (R, t)) of the assembly frame.

    The assembly frame is the first frame whose objects cover all fragment
    names: the simulator replaces the intact body by all fragments in one
    step (they are born in place, before the post-fracture dynamics
    separates them), so this frame is the canonical assembled pose.
    """
    with open(transforms_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"no frames in {transforms_path}")
    final = set(fragment_names)
    for fr in frames:
        objects = fr.get("objects", [])
        names = {obj.get("name", "") for obj in objects}
        if not final <= names:
            continue
        poses: Dict[str, Dict[str, object]] = {}
        for obj in objects:
            name = obj.get("name", "")
            pos = obj.get("position")
            quat = obj.get("rotation_quat")
            if not name or pos is None or quat is None:
                continue
            poses[name] = {
                "R": quat_to_matrix([float(x) for x in quat]),
                "t": np.asarray([float(x) for x in pos], dtype=np.float64),
            }
        return int(fr.get("step", 0)), poses
    raise ValueError(
        f"no assembly frame covering all fragments "
        f"({len(final)} objects) in {transforms_path}")


def export_scene(scene_dir: str, out_dir: str, split: str = "val",
                 part_of: Optional[Dict[str, str]] = None,
                 provenance_of: Optional[Dict[str, Dict[str, object]]] = None
                 ) -> Dict[str, object]:
    """Package one finished scene directory into the released layout.

    Returns a summary dict with fragment counts and exported paths.
    """
    part_of = part_of or {}
    provenance_of = provenance_of or {}
    sid = os.path.basename(scene_dir.rstrip(os.sep))
    m = VARIANT_RE.match(sid)
    if not m:
        raise ValueError(f"scene id does not match <specimen>_v<variant>: {sid}")
    specimen, variant = m.group("specimen"), m.group("variant")

    input_dir = os.path.join(scene_dir, "input")
    output_dir = os.path.join(scene_dir, "output")
    src_intact = os.path.join(input_dir, "specimen.obj")
    transforms_path = os.path.join(output_dir, "transforms.json")
    if not os.path.exists(src_intact) or not os.path.exists(transforms_path):
        raise FileNotFoundError(f"incomplete scene {scene_dir}")

    fragment_files = sorted(f for f in os.listdir(output_dir) if FRAGMENT_RE.match(f))
    if not fragment_files:
        raise ValueError(f"no fragments in {scene_dir}")
    assembly_step, poses = assembly_frame_poses(
        transforms_path, [f[:-4] for f in fragment_files])

    dest = os.path.join(out_dir, split, specimen, f"v{variant}")
    for sub in ("fragments", "gt"):
        os.makedirs(os.path.join(dest, sub), exist_ok=True)
    shutil.copy(src_intact, os.path.join(dest, "intact.obj"))

    fragments: Dict[str, object] = {}
    for i, fname in enumerate(fragment_files):
        name = fname[:-4]
        pose = poses.get(name)
        if pose is None:
            raise ValueError(f"no assembly-frame pose for {name} in {scene_dir}")
        frag_id = f"frag_{i:02d}"
        src = os.path.join(output_dir, fname)
        shutil.copy(src, os.path.join(dest, "fragments", f"{frag_id}.obj"))
        fragments[frag_id] = {
            "source": name,
            "R": pose["R"].tolist(),
            "t": pose["t"].tolist(),
            "n_verts": count_vertices(src),
        }

    gt = {
        "scene": sid,
        "specimen": specimen,
        "variant": f"v{variant}",
        "n_fragments": len(fragments),
        "fragments": fragments,
    }
    with open(os.path.join(dest, "gt", "transforms.json"), "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2)

    meta: Dict[str, object] = {"scene": sid, "specimen": specimen,
                               "variant": f"v{variant}",
                               "part": part_of.get(specimen, ""),
                               "assembly_frame": assembly_step}
    provenance = provenance_of.get(specimen)
    if provenance:
        meta["provenance"] = provenance
    result_path = os.path.join(scene_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        for key in ("fragments", "n_objects", "elapsed_s", "crack_surfaces", "done", "error"):
            if key in result:
                meta[key] = result[key]
    with open(os.path.join(dest, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return {"scene": sid, "split": split, "specimen": specimen,
            "variant": f"v{variant}", "n_fragments": len(fragments),
            "dir": dest}


def collect_scenes(batch_dirs: List[str]) -> List[str]:
    """List finished scene directories under the batch roots.

    A scene counts as finished when it contains a result.json and an output
    directory with at least one fragment mesh.
    """
    scenes: List[str] = []
    for batch in batch_dirs:
        if not os.path.isdir(batch):
            print(f"[skip] batch directory not found: {batch}", flush=True)
            continue
        for entry in sorted(os.listdir(batch)):
            scene = os.path.join(batch, entry)
            if not os.path.isdir(scene):
                continue
            if os.path.exists(os.path.join(scene, "result.json")):
                scenes.append(scene)
    return scenes


def load_split_map(split_file: str) -> Dict[str, str]:
    if not os.path.exists(split_file):
        return {}
    with open(split_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package raw simulator scenes into the HyloFrac layout.")
    parser.add_argument("--batchs", nargs="+", required=True,
                        help="batch output directories containing scene folders")
    parser.add_argument("--out", required=True,
                        help="target dataset root (dataset/hylofrac)")
    parser.add_argument("--split-file", default="",
                        help="JSON mapping specimen id -> train/val/test")
    parser.add_argument("--part-file", default="",
                        help="JSON mapping specimen id -> cranium/mandible")
    parser.add_argument("--provenance-file", default="",
                        help="JSON mapping specimen id -> provenance record "
                             "written to metadata.json")
    parser.add_argument("--resume", action="store_true",
                        help="skip scenes whose gt/transforms.json already exist")
    args = parser.parse_args()

    split_map = load_split_map(args.split_file)
    part_map = load_split_map(args.part_file)
    provenance_map = load_split_map(args.provenance_file)
    scenes = collect_scenes(args.batchs)
    n_ok = 0
    no_provenance = set()
    for scene_dir in scenes:
        sid = os.path.basename(scene_dir.rstrip(os.sep))
        m = VARIANT_RE.match(sid)
        if not m:
            print(f"[skip] bad scene id: {sid}", flush=True)
            continue
        specimen = m.group("specimen")
        if args.provenance_file and specimen not in provenance_map:
            no_provenance.add(specimen)
        split = split_map.get(specimen, "val")
        out_dir = os.path.join(args.out, split, specimen, f"v{m.group('variant')}")
        if args.resume and os.path.exists(os.path.join(out_dir, "gt", "transforms.json")):
            n_ok += 1
            continue
        try:
            summary = export_scene(scene_dir, args.out, split, part_map,
                                   provenance_map)
            n_ok += 1
            print(f"[ok] {sid} -> {summary['dir']} fragments={summary['n_fragments']}",
                  flush=True)
        except Exception as exc:
            print(f"[error] {sid}: {type(exc).__name__}: {exc}", flush=True)

    print(f"[done] scenes={n_ok}/{len(scenes)} -> {args.out}", flush=True)
    if no_provenance:
        print(f"[warn] no provenance record for {len(no_provenance)} "
              f"specimen(s): {', '.join(sorted(no_provenance))}", flush=True)


if __name__ == "__main__":
    main()
