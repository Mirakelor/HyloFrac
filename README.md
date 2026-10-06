# HyloFrac: Physically simulated fracture scenes of gibbon crania and mandibles for 3D fragment reassembly

HyloFrac is a dataset and evaluation benchmark for 3D reassembly of fractured
objects, built from real CT specimens of gibbons (family **Hylobatidae**:
*Hylobates* and *Nomascus*) fractured with the RigidFractureLab simulator
([github.com/Linxu-Fan/RigidFractureLab](https://github.com/Linxu-Fan/RigidFractureLab),
an MPM brittle-fracture simulator following Fan et al. 2022, *Simulating
Brittle Fracture with Material Points*, ACM TOG).

This repository contains the toolchain released with the paper:

- **dataset/** — build the dataset from simulator outputs (packaging,
  ground-truth annotation, splits, quality checks);
- **eval/** — the evaluation framework (data protocol, metrics,
  submission format, subset reporting);
- **baselines/** — reference methods (geometric and learned);
- **vis/** — visualization of assemblies and errors.

## Dataset summary

| Property | Value |
|---|---|
| Source | CT surface models of gibbon crania and mandibles from USNM (Smithsonian National Museum of Natural History) specimens via MorphoSource |
| Fracture | RigidFractureLab MPM brittle fracture (impact-driven, crack propagation) |
| Objects | 154 fracture objects (76 cranium, 78 mandible) from 111 USNM specimens; 14 named *Hylobates* taxa + *Hylobates sp.* + *Nomascus concolor*; up to 10 impact variants each |
| Scenes | 1,517 fracture scenes, 2–100 fragments per scene |
| Ground truth | fragment SE(3) poses, adjacency graph, interface (contact surface) of every adjacent pair, simulator result metadata |
| Licensing | Released material: CC-BY 4.0. Source surface models (third-party, not covered by that licence): MorphoSource Standard agreement with CommercialUsePermitted / 3DPrintingLimited / OnAnyRepository, copyright statement No Known Copyright (NKC) — all 154 objects comply. Code: MIT |

## Data release layout

The archive is released in compact binary containers (xz-compressed) because
the ascii meshes of the full dataset exceed a terabyte. Each scene is:

```
dataset/<split>/<object_id>/<variant>/
  intact.hfm.xz               complete mesh, canonical pose
  fragments/frag_XX.hfm.xz    fragments in local coordinates
  gt/transforms.json          per-fragment SE(3) (assembly frame)
  gt/adjacency.json           adjacency graph with interface areas
  gt/contact_faces/*.hfc.xz   interface of each adjacent pair, as the face
                              indices of the parent fragment
  metadata.json               simulator bookkeeping and provenance
```

`.hfm.xz` stores vertices as 16-bit integers quantised over the bounding box of
that mesh (a few 1e-6 of the scene scale) followed by the exact triangle
indices; `.hfc.xz` stores, per edge, the faces of the parent fragment that form
the interface, which is lossless. Convert a scene back to the ascii layout that
the tools below read with:

```bash
python tools/hf_pack.py unpack --src <split>/<specimen>/<variant> --dst OUT
```

Archive root: `ATTRIBUTION` (per-object sources and reuse terms),
`splits.json` (specimen to split), `configurations.json` (the 1,540 sampled
impact configurations), `README.md`, and `tools/hf_pack.py`.

The impact configuration that produced a scene is the entry with the same
`<specimen>_<variant>` identifier in `configurations.json`; in `metadata.json`,
`n_objects` counts the simulator's rigid bodies and is `fragments + 2` (the
specimen body and the passive ground), so the fragment count is `fragments`.

## Evaluation overview

Three tasks: **T1 reassembly** (predict per-fragment SE(3) from unordered
fragment point clouds), **T2 adjacency** (predict the adjacency graph),
**T3 robustness** (missing / eroded / mixed fragments). Metrics are computed
by the evaluator in `eval/`; see
[docs/benchmark.md](docs/benchmark.md) for the full protocol.

## Quick start

```bash
pip install -e .

# 1. Split specimens into train / val / test (no leakage; every variant of a
#    specimen stays in one split). specimens.txt lists one specimen id per line.
hylofrac-split    --specimens specimens.txt --out splits.json

# 2. Package the dataset from raw simulator outputs
hylofrac-export   --batchs batch0 ... --split-file splits.json --part-file parts.json --provenance-file provenance.json --out dataset/hylofrac
hylofrac-annotate --scenes ... --split-file splits.json --out dataset/hylofrac

# 3. Run a baseline and evaluate
hylofrac-predict  --method geometric --data dataset/hylofrac --out preds
hylofrac-evaluate --data dataset/hylofrac --pred preds --out report.md

# 4. Train a learned baseline
hylofrac-train    --method dgl --data dataset/hylofrac --out runs/dgl
```

## Reference

If you use HyloFrac, please cite:

```bibtex
@misc{hylofrac,
  title     = {HyloFrac: Physically simulated fracture scenes of gibbon crania and mandibles for 3D fragment reassembly},
  author    = {Rong, Z. and Wang, Z. and Mo, K. and Shi, X. and You, J. and Wei, P.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23074709},
  url       = {https://github.com/ORG/HyloFrac}
}
```

## License

- Code: MIT (see [LICENSE](LICENSE)); the MIT licence does not extend to the
  dataset or to the source surface models.
- Released material: CC-BY 4.0 - the fracture geometry, annotations, splits,
  configuration list and documentation produced for this dataset.
- Source surface models: third-party, **not** covered by CC-BY 4.0; obtained
  from MorphoSource under the per-object terms recorded in `ATTRIBUTION`
  (MorphoSource Standard agreement: commercial use permitted, archival of
  published derivatives on any repository, 3D printing limited; No Known
  Copyright). Users must comply with those terms as well.
  See [docs/dataset_card.md](docs/dataset_card.md).
