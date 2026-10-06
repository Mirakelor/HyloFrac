"""HyloFrac evaluation entry point.

Evaluates baseline predictions (unified submission format) against a
packaged dataset and writes a subset report (overall + morphology x
difficulty groups).

Usage::

    hylofrac-evaluate --data dataset/hylofrac --pred <pred_root> \\
        [--splits val test] [--out report.md] [--n-points 1000] \\
        [--degrade none|missing-1..3|erosion-1..3|foreign-1..3]

Layouts:

- ``--data``: dataset root ``dataset/hylofrac/<split>/<specimen>/<variant>/``
  with ``fragments/``, ``gt/`` and ``metadata.json`` per scene;
- ``--pred``: a directory of submission files ``<scene_id>.json``.

``--degrade`` applies one of the T3 robustness degradations
(docs/benchmark.md, section 1) while evaluating; the same scene ids and
submission format are used as for T1/T2, so a T1 submission can be scored
under every degradation level without modification.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from hylofrac.eval.loader import (SceneSample, _scene_seed,
                                  select_foreign_fragments)
from hylofrac.eval.metrics import scene_metrics
from hylofrac.eval.report import aggregate, write_report
from hylofrac.eval.submit import (SubmissionError, load_submission,
                                  read_adjacency)

MISSING_FRACTIONS = {1: 0.1, 2: 0.2, 3: 0.3}
EROSION_DEPTHS = {1: 0.02, 2: 0.05, 3: 0.1}
FOREIGN_COUNTS = {1: 1, 2: 2, 3: 3}


def parse_degrade(spec: str) -> Optional[Dict[str, float]]:
    """Map ``missing-2`` / ``erosion-3`` / ``foreign-1`` to loader options."""
    if not spec or spec == "none":
        return None
    kind, _, level = spec.partition("-")
    try:
        level = int(level)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad degrade level: {spec}")
    if kind == "missing" and level in MISSING_FRACTIONS:
        return {"missing_frac": MISSING_FRACTIONS[level]}
    if kind == "erosion" and level in EROSION_DEPTHS:
        return {"erosion_depth": EROSION_DEPTHS[level]}
    if kind == "foreign" and level in FOREIGN_COUNTS:
        return {"foreign_n": float(FOREIGN_COUNTS[level])}
    raise argparse.ArgumentTypeError(
        f"bad degrade spec: {spec} (use missing-1..3, erosion-1..3, "
        f"foreign-1..3 or none)")


def find_scene(data_root: str, scene_id: str,
               splits: List[str]) -> Optional[str]:
    """Locate a packaged scene directory by scene id across the splits.

    The layout is ``<data_root>/<split>/<specimen>/v<variant>`` where
    ``scene_id = <specimen>_v<variant>``.
    """
    if "_v" not in scene_id:
        return None
    specimen, variant = scene_id.split("_v", 1)
    for split in splits:
        scene_dir = os.path.join(data_root, split, specimen, f"v{variant}")
        if os.path.isdir(scene_dir) and os.path.isdir(os.path.join(scene_dir, "fragments")):
            return scene_dir
    return None


def evaluate_submission(pred_path: str, data_root: str, splits: List[str],
                        n_points: int, seed_offset: int,
                        degrade: Optional[Dict[str, float]] = None,
                        splits_used: Optional[List[str]] = None
                        ) -> Optional[Dict[str, object]]:
    """Evaluate one submission file; returns the per-scene metric dict or
    None when the scene cannot be evaluated.

    ``degrade`` carries loader options (``missing_frac`` / ``erosion_depth``
    / ``foreign_n``) for the T3 robustness tasks.
    """
    scene_id = os.path.basename(pred_path)[:-5]
    try:
        _, fragments = load_submission(pred_path)
    except SubmissionError as exc:
        print(f"[skip] {scene_id}: {exc}", flush=True)
        return None
    scene_dir = find_scene(data_root, scene_id, splits)
    if scene_dir is None:
        print(f"[skip] {scene_id}: packaged scene not found", flush=True)
        return None
    split = os.path.relpath(scene_dir, data_root).split(os.sep)[0]
    if splits_used is not None and split not in splits_used:
        splits_used.append(split)

    foreign_fragments = None
    degrade = degrade or {}
    if degrade.get("foreign_n"):
        meta_path = os.path.join(scene_dir, "metadata.json")
        part = ""
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                part = str(json.load(fh).get("part", ""))
        rng = _scene_seed(f"{scene_id}|foreign", seed_offset)
        foreign_fragments = select_foreign_fragments(
            data_root, split, part, scene_id,
            int(degrade["foreign_n"]), rng)

    sample = SceneSample(scene_dir, scene_id, n_points=n_points,
                         seed_offset=seed_offset,
                         missing_frac=float(degrade.get("missing_frac", 0.0)),
                         erosion_depth=float(degrade.get("erosion_depth", 0.0)),
                         foreign_fragments=foreign_fragments)
    missing = [name for name in sample.names if name not in fragments]
    if missing:
        print(f"[skip] {scene_id}: missing predictions for {missing[:3]}",
              flush=True)
        return None

    pred_r = np.stack([fragments[name]["R"] for name in sample.names])
    pred_t = np.stack([fragments[name]["t"] for name in sample.names])
    pred_adj = None
    pairs = read_adjacency(pred_path)
    if pairs:
        pred_adj = {name: set() for name in sample.names}
        for a, b in pairs:
            if a in pred_adj and b in pred_adj:
                pred_adj[a].add(b)
                pred_adj[b].add(a)
    metrics = scene_metrics(sample, pred_r, pred_t, pred_adj=pred_adj)
    metrics["scene"] = scene_id
    metrics["part"] = sample.part
    metrics["degrade"] = "none" if not degrade else ",".join(
        f"{k}={v}" for k, v in sorted(degrade.items()))
    print(f"[ok] {scene_id}: n={metrics['n_parts']} "
          f"pa={metrics['pa']:.3f} qpos={metrics['qpos']:.3f}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HyloFrac submissions.")
    parser.add_argument("--data", required=True, help="packaged dataset root")
    parser.add_argument("--pred", required=True,
                        help="directory of submission JSON files")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", default="eval_report.md",
                        help="report path (.md or .csv)")
    parser.add_argument("--n-points", type=int, default=1000,
                        help="points sampled per fragment")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="offset for reproducible sampling/rotation seeds")
    parser.add_argument("--degrade", type=parse_degrade, default=None,
                        help="T3 degradation: none | missing-1..3 | "
                             "erosion-1..3 | foreign-1..3")
    args = parser.parse_args()

    results: List[Dict[str, object]] = []
    skipped = 0
    for fname in sorted(os.listdir(args.pred)):
        if not fname.endswith(".json"):
            continue
        result = evaluate_submission(os.path.join(args.pred, fname),
                                     args.data, args.splits,
                                     args.n_points, args.seed_offset,
                                     degrade=args.degrade)
        if result is not None:
            results.append(result)
        else:
            skipped += 1

    tag = "none" if args.degrade is None else "-".join(
        f"{k.split('_')[0]}{int(v)}" for k, v in sorted(args.degrade.items()))
    print(f"[done] degrade={tag} evaluated={len(results)} skipped={skipped}")
    if results:
        rows = aggregate(results)
        write_report(rows, args.out)


if __name__ == "__main__":
    main()
