"""Unified inference entry point: model checkpoint or geometric baseline
-> submissions in the unified format (docs/benchmark.md, section 5).

Usage::

    hylofrac-predict --method global --data dataset/hylofrac \\
        --ckpt runs/global/model_final.pt --out preds
    hylofrac-predict --method geometric --data dataset/hylofrac --out preds
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from hylofrac.baselines.common import MAX_PARTS, quat_to_rotmat, walk_scenes
from hylofrac.eval.loader import SceneSample
from hylofrac.eval.submit import write_submission

GEOMETRIC = "geometric"


def load_checkpoint(ckpt_path: str, device: torch.device
                    ) -> tuple[torch.nn.Module, Dict[str, object]]:
    from hylofrac.baselines.train import build_model

    meta_path = os.path.splitext(ckpt_path)[0] + ".meta.json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model" in state_dict:
        # training checkpoints are payloads (model + optimizer + scheduler +
        # epoch, see train.save_checkpoint); the bare state_dict is what
        # load_state_dict wants.  Accepting both keeps legacy files working.
        state_dict = state_dict["model"]
    model = build_model(meta["method"]).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


def predict_learned(model: torch.nn.Module, sample: SceneSample,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from hylofrac.baselines.common import rotmat_to_quat_scalar_first
    from hylofrac.baselines.garf_net import GarfPoseNet
    from hylofrac.baselines.rpf_net import RpfPoseNet

    pcs = torch.from_numpy(sample.all_pcs).float()  # [P + n_foreign, N, 3]
    n_total = len(pcs)
    n = sample.n_parts
    pad = MAX_PARTS - n_total
    pcs = torch.nn.functional.pad(pcs.unsqueeze(0), (0, 0, 0, 0, 0, pad))
    valids = torch.zeros(1, MAX_PARTS)
    valids[0, :n_total] = 1.0
    predict = getattr(model, "predict", None)
    with torch.no_grad():
        if predict is not None:
            if isinstance(model, (GarfPoseNet, RpfPoseNet)):
                # GARF and RPF pin their anchor (the largest fragment by
                # surface area) to the GT pose during inference, following
                # the reference protocol; the anchor is moved to index 0
                # for the sampler
                anchor = int(np.argmax(sample.areas))
                anchor_q = torch.from_numpy(
                    rotmat_to_quat_scalar_first(sample.gt_R[anchor])).float()
                anchor_t = torch.from_numpy(sample.gt_t[anchor]).float()
                order = [anchor] + [i for i in range(n) if i != anchor]
                inv = np.argsort(order)
                quat, trans = predict(pcs[:, order].to(device),
                                      valids[:, order].to(device),
                                      anchor_q.unsqueeze(0).to(device),
                                      anchor_t.unsqueeze(0).to(device))
                # predictions come back ordered like the permuted input
                quat, trans = quat[:, inv], trans[:, inv]
            else:
                quat, trans = predict(pcs.to(device), valids.to(device))
        else:
            quat, trans = model(pcs.to(device), valids.to(device))
    # keep only the GT fragments (foreign distractors carry no GT pose)
    quat = quat[0, :n].cpu()
    trans = trans[0, :n].cpu().numpy()
    r = quat_to_rotmat(quat).numpy()
    return r, trans


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for a baseline.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="submission directory")
    parser.add_argument("--ckpt", default="", help="checkpoint (learned methods)")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--direct-adjacency", action="store_true",
                        help="phformer: write the adjacency graph predicted "
                             "by the adjacency head (T2 direct channel) "
                             "instead of deriving it from the poses")
    args = parser.parse_args()

    if args.direct_adjacency and args.method != "phformer":
        raise SystemExit("--direct-adjacency is only implemented for phformer")
    if args.method == GEOMETRIC:
        from hylofrac.baselines.geometric import predict_scene
    else:
        if not args.ckpt:
            raise SystemExit(f"--ckpt is required for method {args.method!r}")
        device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available()
                              else "cpu")
        model, state = load_checkpoint(args.ckpt, device)
        print(f"[predict] method={state['method']} epoch={state.get('epoch')} "
              f"device={device}", flush=True)

    n_scenes = 0
    for split in args.splits:
        for scene_id, scene_dir in walk_scenes(args.data, [split]):
            sample = SceneSample(scene_dir, scene_id)
            if args.method == GEOMETRIC:
                r, t = predict_scene(sample.pcs)
            else:
                r, t = predict_learned(model, sample, device)
            fragments = {name: {"R": r[i], "t": t[i]}
                         for i, name in enumerate(sample.names)}
            adjacency = None
            if args.direct_adjacency:
                n = sample.n_parts
                scores = model.adj_scores[0, :n, :n]   # [n, n] probabilities
                pairs = (scores > 0.5).triu(1).nonzero(as_tuple=False)
                adjacency = [(sample.names[int(i)], sample.names[int(j)])
                             for i, j in pairs.tolist()]
            write_submission(os.path.join(args.out, f"{scene_id}.json"),
                             scene_id, args.method, fragments,
                             adjacency=adjacency)
            n_scenes += 1
            print(f"[ok] {scene_id}: n={sample.n_parts}", flush=True)
    print(f"[done] scenes={n_scenes} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
