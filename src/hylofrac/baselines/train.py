"""Unified training entry point for the learned baselines.

Usage::

    hylofrac-train --method dgl --data dataset/hylofrac --out runs/dgl \\
        [--epochs 200] [--lr 1e-3] [--batch 8] [--accum 4]

The physical batch and gradient accumulation follow docs/baselines.md
(effective batch size 32 for every method). Checkpoints and a training log
are written under ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from hylofrac.baselines.common import (AssemblyDataset, GeometricLoss,
                                       walk_scenes)
from hylofrac.baselines.dgl_net import DglPoseNet
from hylofrac.baselines.diffassemble_net import DiffAssemblePoseNet
from hylofrac.baselines.garf_net import GarfPoseNet
from hylofrac.baselines.global_net import GlobalPoseNet
from hylofrac.baselines.lstm_net import LstmPoseNet
from hylofrac.baselines.phformer_net import PhformerPoseNet
from hylofrac.baselines.rpf_net import RpfPoseNet

MODEL_REGISTRY = {
    "global": GlobalPoseNet,
    "lstm": LstmPoseNet,
    "dgl": DglPoseNet,
    "phformer": PhformerPoseNet,
    "diffassemble": DiffAssemblePoseNet,
    "garf": GarfPoseNet,
    "rpf": RpfPoseNet,
}

# physical batch per method (docs/baselines.md); effective batch = 32
PHYSICAL_BATCH = {"global": 8, "lstm": 8, "dgl": 8, "phformer": 4,
                  "diffassemble": 2, "garf": 8, "rpf": 4}
ACCUMULATION = {name: 32 // batch for name, batch in PHYSICAL_BATCH.items()}

# Upper bound on the validation loader pool. One worker holds one whole
# scene: a SceneSample keeps every fragment mesh of its scene alive and
# packaged scenes reach a few GB, so the pool size multiplies peak memory.
# Validation is CPU-bound mesh loading, so this caps the memory rather than
# the training loader, which only keeps small point-cloud tensors per batch.
VAL_MAX_WORKERS = 4


def build_model(method: str) -> nn.Module:
    if method not in MODEL_REGISTRY:
        raise ValueError(f"method {method!r} not implemented "
                         f"(available: {sorted(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[method]()


class _ValSceneBuilder:
    """Yields ``(scene_id, SceneSample)`` pairs for the validation loader.

    Mesh loading plus the protocol sampling is the expensive, purely CPU part
    of validation, so it runs in DataLoader worker processes while the main
    process does the model forward pass and the metrics. The scene list, the
    scene ids and the per-scene seeding are unchanged, so the scores are the
    same as with the serial loop.
    """

    def __init__(self, scenes: List[Tuple[str, str]], n_points: int) -> None:
        self.scenes = list(scenes)
        self.n_points = n_points

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> Tuple[str, object]:
        from hylofrac.eval.loader import SceneSample
        scene_id, scene_dir = self.scenes[index]
        return scene_id, SceneSample(scene_dir, scene_id, n_points=self.n_points)


def _first_item(batch):
    """collate_fn: keep the single element of each batch untouched."""
    return batch[0]


def validate(model: nn.Module, data_root: str, split: str, n_points: int,
             device: torch.device, num_scenes: Optional[int] = None,
             num_workers: int = 0) -> Dict[str, float]:
    """Evaluate the model on a split with the full HyloFrac metric set
    (docs/baselines.md).

    Every scene of the split is scored; pass ``num_scenes`` to score only the
    first N scenes (smoke runs). Scene preparation runs in data-loader
    processes (purely CPU work), the forward pass and the metrics stay on the
    main process/device. Predictions go through the unified inference path
    (hylofrac.baselines.predict) and metrics through the evaluator
    (hylofrac.eval.metrics), the same code used for the reported results.

    The loader pool is capped at :data:`VAL_MAX_WORKERS` with one prefetched
    batch per worker, and a scene whose prediction is not finite is skipped
    and counted in ``val_skipped``: a diverged model must not be able to take
    the run down with an exception from the metrics.
    """
    from hylofrac.baselines.predict import predict_learned
    from hylofrac.eval.metrics import scene_metrics

    model.eval()
    rows: List[Dict[str, object]] = []
    scenes = walk_scenes(data_root, [split])
    if num_scenes is not None:
        scenes = scenes[:num_scenes]
    n_val_workers = min(VAL_MAX_WORKERS, max(0, num_workers))
    print(f"[val] scenes={len(scenes)} workers={n_val_workers}", flush=True)
    loader = torch.utils.data.DataLoader(
        _ValSceneBuilder(scenes, n_points), batch_size=1, shuffle=False,
        num_workers=n_val_workers, collate_fn=_first_item,
        prefetch_factor=1 if n_val_workers else None)
    skipped = 0
    with torch.no_grad():
        for scene_id, sample in loader:
            pred_r, pred_t = predict_learned(model, sample, device)
            if not (np.isfinite(pred_r).all() and np.isfinite(pred_t).all()):
                skipped += 1
                print(f"[val] {scene_id}: non-finite prediction, skipped",
                      flush=True)
                continue
            rows.append(scene_metrics(sample, pred_r, pred_t))
    model.train()
    if not rows:
        return {"val_skipped": float(skipped)} if skipped else {}
    keys = [k for k in rows[0] if k not in ("n_parts", "scene", "part")]
    out = {f"val_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    if skipped:
        out["val_skipped"] = float(skipped)
    return out


def train(model: nn.Module, method: str, data_root: str, out_dir: str,
          epochs: int, lr: float, physical_batch: int, accum: int,
          n_points: int, seed: int, device: torch.device,
          num_workers: int = 0, gpu_throttle: float = 0.0,
          resume_path: Optional[str] = None) -> None:
    torch.manual_seed(seed)
    random.seed(seed)  # LSTM teacher forcing
    dataset = AssemblyDataset(data_root, "train", n_points=n_points,
                              with_adjacency=method == "phformer")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=physical_batch, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = GeometricLoss()
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"[train] resumed from {resume_path} @ epoch {start_epoch}",
              flush=True)

    step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            if gpu_throttle > 0 and device.type == "cuda":
                torch.cuda.synchronize()
            t_step = time.perf_counter()
            pcs = batch["pcs"].to(device)
            quat_gt = batch["quat"].to(device)
            trans_gt = batch["trans"].to(device)
            valids = batch["valids"].to(device)
            step_loss = getattr(model, "loss_step", None)
            if step_loss is not None:
                # methods with their own training objective (diffusion /
                # flow matching) define loss_step; GARF and RPF condition
                # on the scene's anchor fragment
                if method in ("garf", "rpf"):
                    anchor = torch.zeros_like(valids)
                    idx = batch["anchor"].to(device)
                    anchor[torch.arange(anchor.shape[0], device=device),
                           idx] = 1.0
                    losses = step_loss(pcs, quat_gt, trans_gt, valids,
                                       anchor=anchor)
                else:
                    losses = step_loss(pcs, quat_gt, trans_gt, valids)
            else:
                quat, trans = model(pcs, valids)
                losses = loss_fn(pcs, quat, trans, quat_gt, trans_gt, valids)
                total = losses["total"]
                aux = getattr(model, "aux_loss", None)
                if aux is not None:
                    total = total + aux(batch["adj"].to(device),
                                        batch["rel_quat"].to(device),
                                        batch["rel_trans"].to(device), valids)
                losses["total"] = total
            losses["total"].backward()
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(losses["total"].item())
            epoch_steps += 1
            if gpu_throttle > 0 and device.type == "cuda":
                # keep the average SM utilisation near gpu_throttle
                torch.cuda.synchronize()
                dt = time.perf_counter() - t_step
                time.sleep(dt * (1.0 / gpu_throttle - 1.0))
        if step % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        record = {"epoch": epoch, "train_loss": epoch_loss / max(epoch_steps, 1),
                  "lr": scheduler.get_last_lr()[0]}
        periodic = epoch % 10 == 0 or epoch == epochs - 1
        if periodic:
            record.update(validate(model, data_root, "val", n_points, device,
                                   num_workers=num_workers))
        save_checkpoint(model, method, epoch, out_dir, name="model_latest",
                        optimizer=optimizer, scheduler=scheduler)
        if periodic:
            save_checkpoint(model, method, epoch, out_dir,
                            optimizer=optimizer, scheduler=scheduler)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def save_checkpoint(model: nn.Module, method: str, epoch: int,
                   out_dir: str, name: Optional[str] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler=None) -> str:
    stem = name if name else f"model_ep{epoch:04d}"
    path = os.path.join(out_dir, f"{stem}.pt")
    payload = {"model": model.state_dict(), "method": method, "epoch": epoch}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    with open(os.path.join(out_dir, f"{stem}.meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"method": method, "epoch": epoch}, fh)
    return path


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None) -> int:
    """Restore model (and optionally optimizer/scheduler) state; returns the
    next epoch to run. Accepts both the full payload and legacy raw
    state_dict checkpoints."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        return int(payload.get("epoch", -1)) + 1
    model.load_state_dict(payload)
    # legacy raw state_dict: recover the epoch from the file name if present
    import re
    m = re.search(r"model_ep(\d+)", os.path.basename(path))
    return int(m.group(1)) + 1 if m else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HyloFrac baseline.")
    parser.add_argument("--method", required=True, choices=sorted(PHYSICAL_BATCH))
    parser.add_argument("--data", required=True, help="packaged dataset root")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=0,
                        help="physical batch (default per method, docs/baselines.md)")
    parser.add_argument("--accum", type=int, default=0,
                        help="gradient accumulation (default per method)")
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8,
                        help="data-loader worker processes")
    parser.add_argument("--throttle", type=float, default=0.0,
                        help="target average GPU SM utilisation in (0,1]; "
                             "0 disables throttling (full speed)")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="resume from a checkpoint path, or omit the "
                             "value ('auto') to use the newest checkpoint in "
                             "--out")
    args = parser.parse_args()

    batch = args.batch or PHYSICAL_BATCH[args.method]
    accum = args.accum or ACCUMULATION[args.method]
    if batch * accum != 32:
        raise SystemExit(f"physical batch x accumulation must equal 32 "
                         f"(got {batch} x {accum})")
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    model = build_model(args.method).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] method={args.method} device={device} "
          f"batch={batch} accum={accum} params={n_params} "
          f"throttle={args.throttle}", flush=True)
    resume_path = args.resume
    if resume_path == "auto":
        latest = os.path.join(args.out, "model_latest.pt")
        if os.path.exists(latest):
            resume_path = latest
        else:
            eps = sorted(glob.glob(os.path.join(args.out, "model_ep*.pt")))
            resume_path = eps[-1] if eps else None
    train(model, args.method, args.data, args.out, args.epochs, args.lr,
          batch, accum, args.n_points, args.seed, device,
          num_workers=args.workers, gpu_throttle=args.throttle,
          resume_path=resume_path)
    final_path = save_checkpoint(model, args.method, args.epochs - 1,
                                args.out, name="model_final")
    print(f"[done] -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
