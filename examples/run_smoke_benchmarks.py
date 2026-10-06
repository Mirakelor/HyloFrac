"""One-shot smoke and audit runner for the GPU training host.

Against a packaged HyloFrac dataset this runs:

1. the full test suite (pytest), including the reference-equivalence
   checks;
2. one training epoch per learned baseline on the train split, followed
   by a full-metric validation run on the val split;
3. the geometric baseline over a val subset through the unified evaluator
   metrics.

Usage::

    python examples/run_smoke_benchmarks.py --data dataset/hylofrac \
        [--runs-root runs] \
        [--methods global lstm dgl phformer diffassemble garf] \
        [--epochs 1] [--n-points 1000] [--val-scenes N] [--no-pytest]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNED = ["global", "lstm", "dgl", "phformer", "diffassemble", "garf"]


def run_pytest() -> None:
    import pytest

    code = pytest.main(["-q", os.path.join(REPO_ROOT, "tests")])
    if code != 0:
        raise SystemExit(f"pytest exited with {code}")


def smoke_learned(method: str, data_root: str, runs_root: str, epochs: int,
                  n_points: int, val_scenes: int, device: torch.device) -> dict:
    from hylofrac.baselines.train import (PHYSICAL_BATCH, build_model, train,
                                          validate)
    from hylofrac.baselines.common import count_parameters

    model = build_model(method).to(device)
    out_dir = os.path.join(runs_root, f"smoke_{method}")
    train(model, method, data_root, out_dir, epochs=epochs, lr=1e-3,
          physical_batch=PHYSICAL_BATCH[method], accum=32 // PHYSICAL_BATCH[method],
          n_points=n_points, seed=0, device=device, num_workers=0)
    val = validate(model, data_root, "val", n_points, device,
                   num_scenes=val_scenes)
    report = {"params": count_parameters(model), **val}
    print(f"[smoke:{method}] " + json.dumps(report), flush=True)
    return report


def smoke_geometric(data_root: str, n_points: int, val_scenes: int) -> dict:
    from hylofrac.baselines.geometric import predict_scene
    from hylofrac.baselines.common import walk_scenes
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.metrics import scene_metrics

    rows = []
    for scene_id, scene_dir in walk_scenes(data_root, ["val"])[:val_scenes]:
        sample = SceneSample(scene_dir, scene_id, n_points=n_points)
        r, t = predict_scene(sample.pcs)
        rows.append(scene_metrics(sample, r, t))
    keys = [k for k in rows[0] if k not in ("n_parts", "scene", "part")]
    report = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("[smoke:geometric] " + json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="packaged dataset root")
    parser.add_argument("--runs-root", default=os.path.join(REPO_ROOT, "runs"),
                        help="directory for checkpoints and logs (default: repo runs/)")
    parser.add_argument("--methods", nargs="+", default=LEARNED,
                        choices=LEARNED)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--val-scenes", type=int, default=None,
                        help="only score the first N val scenes "
                             "(default: the whole val split)")
    parser.add_argument("--no-pytest", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    import hylofrac  # noqa: F401  (import check)

    os.makedirs(args.runs_root, exist_ok=True)
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available()
                          else "cpu")
    print(f"[smoke] torch={torch.__version__} device={device} "
          f"data={args.data} runs={args.runs_root}", flush=True)
    if not args.no_pytest:
        run_pytest()

    summary: dict[str, dict] = {}
    for method in args.methods:
        summary[method] = smoke_learned(method, args.data, args.runs_root,
                                        args.epochs, args.n_points,
                                        args.val_scenes, device)
    summary["geometric"] = smoke_geometric(args.data, args.n_points,
                                           args.val_scenes)

    print("\n=== smoke summary ===", flush=True)
    for method, metrics in summary.items():
        print(f"{method:12s} " + "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)


if __name__ == "__main__":
    main()
