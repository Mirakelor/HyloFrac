"""Train/val/test splitting for HyloFrac.

The archive is keyed by fracture object: a scene lives under
``<split>/<object_id>/<variant>/``, where ``object_id`` is the MorphoSource
media identifier of one object (a cranium or a mandible).  Several objects can
belong to the same physical specimen (USNM number), and those objects must stay
in the same split, otherwise a method could be evaluated on a bone whose other
half it has already seen.  ``--level specimen`` (the default) enforces that by
grouping objects through ``provenance.physical_object_title`` in each scene's
``metadata.json``; ``--level object`` reproduces the older behaviour, which
splits directory names and therefore does **not** guarantee specimen-level
separation.

Usage::

    hylofrac-split --data dataset/hylofrac --out splits.json      # specimen level
    hylofrac-split --data dataset/hylofrac --level object --out s.json
    hylofrac-split --specimens objects.txt --out splits.json      # object level

The released ``splits.json`` is the definition of the split used by the
benchmark: it was produced independently, and re-running this tool gives a
different (but equally specimen-level) assignment.  Do not overwrite the
released file when comparing results across models.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

SPLITS = ("train", "val", "test")


def collect_from_data(data_root: str) -> List[str]:
    """Object ids of an exported dataset tree (``<split>/<object_id>/...``)."""
    objects: List[str] = []
    for split in SPLITS:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for entry in sorted(os.listdir(split_dir)):
            if os.path.isdir(os.path.join(split_dir, entry)) and entry not in objects:
                objects.append(entry)
    return sorted(objects)


def collect_specimen_of_object(data_root: str) -> Tuple[Dict[str, str], int]:
    """Map every object id to its physical specimen id.

    The specimen id is read from ``provenance.physical_object_title`` of the
    first scene that belongs to the object (for example
    ``USNM:MAMM:USNM 153798``).  An object whose metadata is missing or carries
    no such field is kept as its own group; the number of those objects is
    returned so that the caller can warn about it.
    """
    mapping: Dict[str, str] = {}
    n_unresolved = 0
    for obj in collect_from_data(data_root):
        resolved = None
        for split in SPLITS:
            obj_dir = os.path.join(data_root, split, obj)
            if not os.path.isdir(obj_dir):
                continue
            for variant in sorted(os.listdir(obj_dir)):
                meta_path = os.path.join(obj_dir, variant, "metadata.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path, encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    continue
                title = (meta.get("provenance") or {}).get("physical_object_title")
                if title:
                    resolved = str(title)
                    break
            if resolved:
                break
        if resolved:
            mapping[obj] = resolved
        else:
            mapping[obj] = obj
            n_unresolved += 1
    return mapping, n_unresolved


def make_splits(specimens: List[str], train: float, val: float,
                seed: int) -> Dict[str, str]:
    """Assign each group to train/val/test with the given proportions."""
    if not 0.0 < train < 1.0 or not 0.0 < val < 1.0 or train + val >= 1.0:
        raise ValueError("train and val must be positive and sum below 1")
    rng = random.Random(seed)
    order = sorted(specimens)
    rng.shuffle(order)
    n_train = int(round(len(order) * train))
    n_val = int(round(len(order) * val))
    mapping: Dict[str, str] = {}
    for i, specimen in enumerate(order):
        if i < n_train:
            mapping[specimen] = "train"
        elif i < n_train + n_val:
            mapping[specimen] = "val"
        else:
            mapping[specimen] = "test"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign fracture objects to train/val/test splits. With "
                    "--data and the default --level specimen, all objects of "
                    "one physical specimen (USNM number) receive the same "
                    "split, which is what the released splits.json does.")
    parser.add_argument("--specimens", default="",
                        help="text file with one object id per line (object "
                             "level: objects are not grouped by specimen)")
    parser.add_argument("--data", default="",
                        help="exported dataset root to collect object ids from")
    parser.add_argument("--level", choices=("specimen", "object"),
                        default="specimen",
                        help="grouping level used with --data (default: specimen)")
    parser.add_argument("--out", required=True, help="output splits.json path")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    if args.specimens:
        with open(args.specimens, encoding="utf-8") as fh:
            objects = [line.strip() for line in fh if line.strip()]
        print("[warn] --specimens splits object ids directly; objects of one "
              "physical specimen may end up in different splits. Use --data "
              "with --level specimen to enforce specimen-level separation.")
        mapping = make_splits(objects, args.train, args.val, args.seed)
        n_groups = len(mapping)
    elif args.data:
        if args.level == "specimen":
            specimen_of, n_unresolved = collect_specimen_of_object(args.data)
            if n_unresolved:
                print(f"[warn] {n_unresolved} object(s) have no "
                      f"provenance.physical_object_title and were treated as "
                      f"their own specimen")
            by_specimen = make_splits(sorted(set(specimen_of.values())),
                                      args.train, args.val, args.seed)
            mapping = {obj: by_specimen[spec] for obj, spec in specimen_of.items()}
            n_groups = len(by_specimen)
        else:
            print("[warn] --level object splits directory names; objects of one "
                  "physical specimen may end up in different splits")
            objects = collect_from_data(args.data)
            mapping = make_splits(objects, args.train, args.val, args.seed)
            n_groups = len(mapping)
    else:
        raise SystemExit("provide --specimens or --data")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, sort_keys=True)

    counts: Dict[str, int] = {s: 0 for s in SPLITS}
    for split in mapping.values():
        counts[split] += 1
    print(f"[done] groups={n_groups} objects={len(mapping)} "
          f"splits={counts} -> {args.out}")
    print("[note] the released splits.json is the benchmark definition; a "
          "regenerated split is different and must not replace it")


if __name__ == "__main__":
    main()
