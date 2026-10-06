"""Quality checks over a packaged HyloFrac dataset tree.

Checks per scene:
    - packaged layout completeness (intact, fragments, gt files);
    - fragment meshes are watertight and their face counts are sane;
    - the fragments approximately reconstruct the intact mesh
      (volume ratio in a tolerance band);
    - GT adjacency is consistent with the fragment count (0 < edges <=
      n(n-1)/2) and, unless --no-contact-meshes is given, contact meshes
      exist for every edge.  Pass --no-contact-meshes when the archive is
      released without gt/contact_surfaces (the evaluator does not read
      them; see docs/benchmark.md, section 6).

Output: a CSV report with one row per scene.

Usage::

    hylofrac-qc --data dataset/hylofrac --out qc_report.csv [--vol-tol 0.15] \\
        [--no-contact-meshes]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import trimesh

from hylofrac.dataset.objio import count_vertices


def mesh_volume(path: str) -> Optional[float]:
    """Signed-free volume of a mesh, or None when it is not closed."""
    try:
        mesh = trimesh.load(path, process=False)
    except Exception:
        return None
    if mesh is None or len(mesh.faces) == 0:
        return None
    if not mesh.is_watertight:
        return None
    return float(mesh.volume)


def union_volume(mesh_paths: List[str]) -> Optional[float]:
    """Volume of the union of fragment meshes.

    Fragments of one scene are disjoint in the assembled pose and rigid
    poses preserve volume, so the union volume equals the sum of the
    individual watertight volumes.
    """
    total = 0.0
    for path in mesh_paths:
        vol = mesh_volume(path)
        if vol is None:
            return None
        total += vol
    return total


def check_scene(scene_dir: str, vol_tol: float,
                require_contact_meshes: bool = True) -> Dict[str, object]:
    """Run all checks on one packaged scene directory."""
    report: Dict[str, object] = {"scene": os.path.basename(scene_dir)}
    intact_path = os.path.join(scene_dir, "intact.obj")
    frag_dir = os.path.join(scene_dir, "fragments")
    gt_dir = os.path.join(scene_dir, "gt")
    cs_dir = os.path.join(gt_dir, "contact_surfaces")

    ok = True
    for required in (intact_path, frag_dir, gt_dir):
        if not os.path.exists(required):
            report[f"missing_{os.path.basename(required)}"] = True
            ok = False
            return {**report, "ok": False}
    frag_files = sorted(f for f in os.listdir(frag_dir) if f.endswith(".obj"))
    if len(frag_files) < 2:
        report["fragments_count"] = len(frag_files)
        ok = False

    report["n_fragments"] = len(frag_files)
    report["n_verts_total"] = sum(count_vertices(os.path.join(frag_dir, f))
                                  for f in frag_files)

    # watertightness of the fragments (rigid poses do not change it)
    n_open = 0
    for f in frag_files:
        mesh = trimesh.load(os.path.join(frag_dir, f), process=False)
        if mesh is None or not mesh.is_watertight:
            n_open += 1
    report["n_open_fragments"] = n_open
    if n_open > 0:
        ok = False

    # reconstruction: fragment volumes vs intact volume
    intact_vol = mesh_volume(intact_path)
    union_vol = union_volume([os.path.join(frag_dir, f) for f in frag_files])
    if intact_vol is None or union_vol is None or intact_vol <= 0:
        report["volume_ratio"] = None
        ok = False
    else:
        ratio = union_vol / intact_vol
        report["volume_ratio"] = round(ratio, 4)
        if abs(ratio - 1.0) > vol_tol:
            ok = False

    # adjacency sanity
    adj_path = os.path.join(gt_dir, "adjacency.json")
    if os.path.exists(adj_path):
        with open(adj_path, "r", encoding="utf-8") as fh:
            adj = json.load(fh)
        edges = adj.get("edges", [])
        n = len(frag_files)
        report["n_edges"] = len(edges)
        if not (0 < len(edges) <= n * (n - 1) // 2):
            ok = False
        if require_contact_meshes:
            missing_cs = 0
            for edge in edges:
                name = edge.get("contact_mesh")
                if name and not os.path.exists(os.path.join(cs_dir, name)):
                    missing_cs += 1
            report["missing_contact_meshes"] = missing_cs
            if missing_cs > 0:
                ok = False
        else:
            # The archive is released without gt/contact_surfaces; the
            # adjacency graph carries the interface areas, and the meshes
            # can be regenerated from the fragments and the assembly poses.
            report["contact_meshes"] = "not checked"
    else:
        report["adjacency"] = "missing"
        ok = False

    report["ok"] = ok
    return report


def collect_scenes(data_root: str) -> List[str]:
    scenes: List[str] = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for specimen in sorted(os.listdir(split_dir)):
            spec_dir = os.path.join(split_dir, specimen)
            if not os.path.isdir(spec_dir):
                continue
            for variant in sorted(os.listdir(spec_dir)):
                scene_dir = os.path.join(spec_dir, variant)
                if os.path.isdir(scene_dir):
                    scenes.append(scene_dir)
    return sorted(scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality checks for a packaged dataset.")
    parser.add_argument("--data", required=True, help="packaged dataset root")
    parser.add_argument("--out", default="qc_report.csv")
    parser.add_argument("--vol-tol", type=float, default=0.15,
                        help="allowed |assembled_union / intact - 1| tolerance")
    parser.add_argument("--no-contact-meshes", action="store_true",
                        help="do not require gt/contact_surfaces/*.obj; use for "
                             "an archive that is released without them")
    args = parser.parse_args()

    scenes = collect_scenes(args.data)
    reports: List[Dict[str, object]] = []
    for scene_dir in scenes:
        try:
            report = check_scene(scene_dir, args.vol_tol,
                                 require_contact_meshes=not args.no_contact_meshes)
        except Exception as exc:
            report = {"scene": os.path.basename(scene_dir), "ok": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        reports.append(report)
        flag = "OK " if report["ok"] else "BAD"
        print(f"[{flag}] {report['scene']}", flush=True)

    n_ok = sum(1 for r in reports if r.get("ok"))
    print(f"[done] scenes={len(reports)} ok={n_ok} bad={len(reports) - n_ok} -> {args.out}",
          flush=True)

    if reports:
        keys = sorted({k for r in reports for k in r})
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reports)


if __name__ == "__main__":
    main()
