"""Subset reporting for HyloFrac evaluations.

Metrics are aggregated over the overall set and over the 8 subsets formed by
morphology (cranium / mandible) x difficulty (2-5 / 6-20 / 21-50 / 51-100
fragments), plus morphology-only groups. Difficulty is derived from the
scene's real fragment count; morphology comes from the scene metadata.

Reports can be written as Markdown tables or CSV.
"""

from __future__ import annotations

import csv
from typing import Dict, List

METRIC_KEYS = ["pa", "rmse_r_deg", "rmse_t", "sym_cd", "qpos",
               "adj_prec", "adj_rec", "adj_f1"]
METRIC_LABELS = {
    "pa": "PA", "rmse_r_deg": "RMSE(R deg)", "rmse_t": "RMSE(T)",
    "sym_cd": "SymCD", "qpos": "Qpos", "adj_prec": "adjP",
    "adj_rec": "adjR", "adj_f1": "adjF1",
}


def difficulty_bucket(n_parts: int) -> str:
    """Map a fragment count to its difficulty bucket name."""
    if n_parts <= 5:
        return "easy_2_5"
    if n_parts <= 20:
        return "medium_6_20"
    if n_parts <= 50:
        return "hard_21_50"
    return "extreme_51_100"


def aggregate(scene_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Group per-scene metric dicts into subset rows (mean per subset).

    Each input dict carries the METRIC_KEYS plus ``part`` (morphology tag).
    """
    groups: Dict[str, List[Dict[str, object]]] = {}
    for result in scene_results:
        part = str(result.get("part", "unknown"))
        bucket = difficulty_bucket(int(result["n_parts"]))
        for subset in ("overall", part, f"{part}|{bucket}"):
            groups.setdefault(subset, []).append(result)

    rows: List[Dict[str, object]] = []
    for subset in sorted(groups):
        members = groups[subset]
        row: Dict[str, object] = {"subset": subset, "n_scenes": len(members)}
        for metric in METRIC_KEYS:
            values = [float(m[metric]) for m in members if metric in m]
            row[metric] = sum(values) / len(values) if values else float("nan")
        rows.append(row)
    return rows


def write_markdown(rows: List[Dict[str, object]], out_path: str) -> None:
    header = ["subset", "scenes"] + [METRIC_LABELS[k] for k in METRIC_KEYS]
    templates = {"subset": "{}", "n_scenes": "{}", "pa": "{:.4f}",
                 "rmse_r_deg": "{:.3f}", "rmse_t": "{:.4f}", "sym_cd": "{:.4f}",
                 "qpos": "{:.4f}", "adj_prec": "{:.4f}", "adj_rec": "{:.4f}",
                 "adj_f1": "{:.4f}"}
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            cells = [templates[key].format(row[key])
                     for key in ["subset", "n_scenes"] + METRIC_KEYS]
            fh.write("| " + " | ".join(cells) + " |\n")


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    fieldnames = ["subset", "n_scenes"] + METRIC_KEYS
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_report(rows: List[Dict[str, object]], out_path: str) -> None:
    if out_path.endswith(".csv"):
        write_csv(rows, out_path)
    else:
        write_markdown(rows, out_path)
    print(f"report -> {out_path}")
