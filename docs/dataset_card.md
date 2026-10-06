# HyloFrac Dataset Card

This card follows the Datasheets for Datasets framework
([Gebru et al., 2021](https://arxiv.org/abs/1803.09010)): the seven sections
below correspond to its template (Motivation, Composition, Collection,
Preprocessing, Uses, Distribution, Maintenance).

## Motivation

- **Why**: 3D reassembly of fractured objects matters for archaeology,
  paleontology, forensics and art restoration, but progress is limited by
  the data: procedurally fractured synthetic shapes lack physical realism,
  while real fractured objects are scarce, small in scale, and usually come
  without a known complete original or exact per-fragment ground truth. This
  dataset provides physically simulated fractures of real anatomical CT
  specimens together with exact ground truth (SE(3) poses, adjacency graph
  and the interface of every adjacent pair) at a scale that supports
  learning-based methods.
- **Who**: this dataset accompanies the HyloFrac benchmark paper.

## Composition

- 154 fracture objects: CT surface models of gibbon crania (76) and mandibles
  (78), family Hylobatidae — 14 named *Hylobates* taxa plus *Hylobates sp.*
  and *Nomascus concolor* — from 111 specimens of the Smithsonian National
  Museum of Natural History (USNM), downloaded from MorphoSource.
- Each object is fractured up to 10 times with distinct impact
  configurations sampled from a four-tier recipe (drop height 1.5-15,
  50,000-119,000 material points, shatter parameter 0.3-1.0, reference stress
  404-1,995, in the normalised scale of the scenes; the tiers target 2-5,
  6-20, 21-50 and 51-100 fragments), giving 1,517 scenes with 2-100 fragments
  each and a median of 20. All 1,540 sampled configurations, including the 23
  that produced no fragments and therefore no scene, are listed in
  `configurations.json`.
- Fragments are rigid bodies after fracture; the base split has no erosion or
  missing pieces (the T3 task introduces controlled degradations, see
  [benchmark.md](benchmark.md), section 1).

## Collection

- Meshes: CT-derived surface models downloaded from MorphoSource (media
  downloads; per-object metadata recorded in the archive).
- Fracture: RigidFractureLab
  ([github.com/Linxu-Fan/RigidFractureLab](https://github.com/Linxu-Fan/RigidFractureLab)),
  an MPM brittle-fracture simulator (Fan et al. 2022, ACM TOG);
  impact-driven fracture with crack propagation over 1,000 MPM steps and
  rigid-body dynamics after fracture.

## Preprocessing

Source meshes are processed in this order: first repaired to watertight,
then centered at the origin and normalized so the bounding-box diagonal
equals 1. Simulation inputs are then decimated to 100,000 faces. The tools
used per step:

| Step | Tool |
|---|---|
| vertex welding (tolerance 1e-5) | in-house NumPy implementation |
| hole filling / mesh repair | `trimesh` (fill_holes) and `pymeshfix` (MeshFix) |
| centering and diagonal normalization | in-house implementation on `trimesh`/NumPy |
| decimation to 100,000 faces | `igl::decimate` (libigl) |
| fracture simulation | RigidFractureLab ([github.com/Linxu-Fan/RigidFractureLab](https://github.com/Linxu-Fan/RigidFractureLab)) |
| packaging and annotation | the dataset tools of this repository (`hylofrac.dataset`) |

The simulator fractures a decimated input (at most 100,000 faces) and
outputs fragment meshes at its own reconstruction resolution (per fragment:
median ≈ 4×10^4 vertices, up to ≈ 4×10^6); the packaged fragments keep that
resolution.

Each scene is packaged with the intact mesh, fragments in local
coordinates, GT SE(3) transforms (assembly frame: the frame where all
fragments are born in place), the GT adjacency graph with interface areas, the
interface of each adjacent pair, and simulator result metadata. The archive is
released in compact xz-compressed binary containers (1,517 scenes, 35.7 GB;
about 1.1 TB in ascii), with `tools/hf_pack.py` to convert a scene back to the
ascii layout. See [benchmark.md](benchmark.md), section 6, and the dataset
README for the layout and the containers.

## Uses

- Benchmarking 3D reassembly methods (see [benchmark.md](benchmark.md));
  training and evaluation splits are provided by specimen to prevent
  leakage. `splits.json` assigns 74, 19 and 18 of the 111 physical specimens
  (103, 25 and 26 fracture objects; 1,013, 247 and 257 scenes) to training,
  validation and test, and both morphologies appear in every split.
- Intended for research on geometric reassembly, pose estimation and
  adjacency reasoning. The data are synthetic fractures of bone-shaped
  surfaces and are not suitable for medical diagnosis, surgical planning,
  forensic evidence, or any application where physical fidelity of the
  fracture process itself must be relied upon.

## Distribution

- Dataset released on Zenodo: DOI [10.5281/zenodo.23074709](https://doi.org/10.5281/zenodo.23074709).
- Code on GitHub (MIT): [github.com/ORG/HyloFrac](https://github.com/ORG/HyloFrac),
  mirrored at Anonymous GitHub for peer review.
- Released material licence: CC-BY 4.0 - the fracture geometry, annotations,
  splits, configuration list and documentation produced for this dataset.
- Source surface models: third-party and **not** covered by CC-BY 4.0. All 154
  are CT surface models of gibbon crania and mandibles from USNM (Smithsonian
  National Museum of Natural History) specimens via MorphoSource, under the
  MorphoSource Standard agreement with `CommercialUsePermitted` +
  `3DPrintingLimited` + `OnAnyRepository` and the copyright statement
  `No Known Copyright` (NKC). Per-object provenance is listed in `ATTRIBUTION`
  in the dataset archive, and users must comply with those terms as well.

## Maintenance

- The dataset is maintained by the HyloFrac authors. Corrections and
  additions are released as versioned updates on Zenodo; issues and feedback
  are tracked through the GitHub repository. Each release records the set of
  objects and scenes it contains, so results remain reproducible across
  versions.
