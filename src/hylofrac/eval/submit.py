"""Unified submission format for HyloFrac baselines.

Every method (geometric or learned) converts its raw predictions into one
JSON file per scene; the evaluator only consumes this format, so all methods
are measured with exactly the same protocol (docs/benchmark.md, section 5).

Format::

    {
      "scene": "000102166_v00",
      "method": "dgl",
      "fragments": {
        "frag_00": {"R": [[r00, r01, r02], [r10, ...], [r20, ...]],
                    "t": [x, y, z]},
        ...
      },
      "adjacency": [["frag_00", "frag_02"], ...]   # optional (T2 direct)
    }

``R`` is the 3x3 rotation (row-major) mapping fragment-local coordinates to
world coordinates; ``t`` is the translation in normalized units. The
optional ``adjacency`` lists the predicted touching pairs directly; when it
is absent the evaluator derives the graph from the predicted poses.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class SubmissionError(ValueError):
    """Raised when a submission file violates the format."""


def validate_submission(path: str) -> Tuple[bool, str]:
    """Return (ok, error_message) for a submission file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable json: {exc}"
    return _validate_dict(data)


def _validate_dict(data: object) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "top level must be an object"
    scene = data.get("scene")
    if not isinstance(scene, str) or not scene:
        return False, "missing or invalid 'scene'"
    fragments = data.get("fragments")
    if not isinstance(fragments, dict) or not fragments:
        return False, "'fragments' must be a non-empty object"
    for name, pose in fragments.items():
        if not isinstance(name, str):
            return False, f"fragment key must be a string, got {name!r}"
        if not isinstance(pose, dict):
            return False, f"{name}: pose must be an object"
        try:
            rmat = np.asarray(pose["R"], dtype=np.float64)
            trans = np.asarray(pose["t"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False, f"{name}: missing or malformed R/t"
        if rmat.shape != (3, 3) or trans.shape != (3,):
            return False, f"{name}: R must be 3x3 and t length 3"
        if not np.all(np.isfinite(rmat)) or not np.all(np.isfinite(trans)):
            return False, f"{name}: non-finite values"
        if not np.allclose(rmat @ rmat.T, np.eye(3), atol=1e-4):
            return False, f"{name}: R is not a rotation matrix"
    adjacency = data.get("adjacency")
    if adjacency is not None:
        if not isinstance(adjacency, list):
            return False, "'adjacency' must be a list of fragment-name pairs"
        for pair in adjacency:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(n, str) for n in pair)):
                return False, f"'adjacency' entry must be [name_i, name_j], got {pair!r}"
    return True, ""


def load_submission(path: str) -> Tuple[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load and validate a submission; raises SubmissionError on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"{path}: unreadable json: {exc}") from exc
    ok, error = _validate_dict(data)
    if not ok:
        raise SubmissionError(f"{path}: {error}")
    fragments = {}
    for name, pose in data["fragments"].items():
        fragments[name] = {
            "R": np.asarray(pose["R"], dtype=np.float64),
            "t": np.asarray(pose["t"], dtype=np.float64),
        }
    return data["scene"], fragments


def read_adjacency(path: str) -> Optional[List[Tuple[str, str]]]:
    """Return the optional predicted adjacency pairs of a validated
    submission, or None when the field is absent."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    adjacency = data.get("adjacency")
    if not isinstance(adjacency, list):
        return None
    return [tuple(pair) for pair in adjacency if isinstance(pair, list)]


def write_submission(path: str, scene: str, method: str,
                     fragments: Dict[str, Dict[str, np.ndarray]],
                     adjacency: Optional[List[Tuple[str, str]]] = None) -> None:
    """Write a submission file; ``fragments`` values carry R (3x3) and t (3).

    ``adjacency`` optionally lists the predicted touching fragment pairs
    (T2 direct channel); the evaluator derives the graph from the poses
    when it is omitted.
    """
    payload = {
        "scene": scene,
        "method": method,
        "fragments": {
            name: {"R": pose["R"].tolist(), "t": pose["t"].tolist()}
            for name, pose in sorted(fragments.items())
        },
    }
    if adjacency:
        payload["adjacency"] = [list(pair) for pair in sorted(adjacency)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
