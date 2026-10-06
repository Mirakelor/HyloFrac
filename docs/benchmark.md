# HyloFrac Benchmark Protocol

HyloFrac is a benchmark for 3D reassembly of fractured bone specimens:
154 fracture objects (76 cranium, 78 mandible) from 111 USNM gibbon
(Hylobatidae) specimens, each fractured up to 10 times with the
RigidFractureLab MPM brittle-fracture simulator
([github.com/Linxu-Fan/RigidFractureLab](https://github.com/Linxu-Fan/RigidFractureLab);
Fan et al. 2022, *Simulating Brittle Fracture with Material Points*, ACM
TOG). This yields 1,517 scenes with 2-100 fragments each.

## 1. Tasks

| Task | Input | Output |
|---|---|---|
| T1 Reassembly | per-fragment point cloud (1,000 points, uniformly sampled) with an independent random rotation per fragment | per-fragment SE(3) |
| T2 Adjacency | fragment pairs / point clouds | adjacency graph (derived from a T1 solution or predicted directly) |
| T3 Robustness | degraded scenes under one of three independent degradations (below) | same as T1/T2, reported per degradation type and level |

The three T3 degradations are applied **independently** (one per run, never
combined) so that robustness to each degradation is measured separately.
Levels are fractions of the normalized scale (diagonal = 1). Degradations
are applied by the data loader (`hylofrac.eval.loader`, seeded from the
scene id), so methods consume the same fragment-point-cloud interface as
T1/T2; a T1 submission can be scored under every T3 level unchanged.

| T3a Missing | T3b Erosion | T3c Foreign |
|---|---|---|
| L1: 10% of fragments removed | depth 0.02 | 1 distractor fragment |
| L2: 20% of fragments removed | depth 0.05 | 2 distractor fragments |
| L3: 30% of fragments removed | depth 0.1 | 3 distractor fragments |

- **T3a Missing** removes the smallest fragments first (by surface area,
  at least two fragments kept) and prunes the GT poses and adjacency graph
  accordingly; methods are not told which fragments are missing.
- **T3b Erosion** displaces the surface vertices of every fragment inward
  along their normals by the given depth plus uniform noise of the same
  magnitude, before point sampling; GT poses and adjacency are unchanged.
- **T3c Foreign** injects distractor fragments taken from other specimens
  of the same morphology and split. They appear in the method input but
  carry no GT entry: per-fragment metrics are computed on the true
  fragments only, while the assembly-level metrics (Qpos, Sym Chamfer)
  penalize methods that assemble the distractors into the scene.

## 2. Metrics

Metrics are computed by the evaluator in `hylofrac/eval`. Definitions:

| Metric | Definition | Task |
|---|---|---|
| PA@0.01 | per-fragment mean bidirectional *squared* Chamfer between the prediction- and GT-transformed point cloud < 0.01 counts as correct; averaged over the fragments of a scene (per-scene mean) | T1 |
| RMSE(R deg) | root mean square of the **geodesic** rotation error between predicted and GT rotations | T1 |
| RMSE(T) | sqrt(mean(\|\|Δt\|\|²)) in normalized units | T1 |
| Sym Chamfer | bidirectional mean distance between the predicted assembly and the GT assembly point clouds | T1 |
| Qpos | after rigid alignment on the fragment with the largest volume (anchor), volume-weighted mean per-fragment overlap of voxelized fragment surfaces (voxel pitch 0.005, normalized units) | T1 |
| Qprecision / Qrecall / F1 | adjacency consistency against the **GT adjacency graph**, volume-weighted | T2 |

## 3. Protocol

- **Scale**: every scene is normalized so its bounding-box diagonal equals 1;
  the PA threshold 0.01 refers to this scale.
- **Fragment range**: the full range 2-100 is evaluated (easy, medium, hard
  and extreme tiers are all represented).
- **Anchor**: PA / RMSE / Chamfer are per-fragment independent; Qpos aligns on
  the fragment with the largest volume.
- **Data loader protocol**: fragment meshes in assembled pose → 1,000 surface
  points per fragment; translation GT = the fragment centroid relative to the
  anchor's centroid, rotation GT = the fragment's observation rotation
  (uniform random SO(3) per fragment), both relative to the anchor fragment
  (fragment 0, section 4), whose GT pose is the identity. Evaluation is
  deterministic: sampling and observation rotations are seeded from the scene
  id, so a scene is scored against identical inputs on every run. Training
  uses the same protocol but re-draws both each epoch as data augmentation.
- **Splits**: by physical specimen - 74 / 19 / 18 of the 111 USNM specimens,
  which is 103 / 25 / 26 fracture objects and 1,013 / 247 / 257 scenes in
  training / validation / test; all fracture objects of one specimen stay in the same
  split.
- **Subset reporting**: metrics are reported for the overall set and for
  the **7 non-empty subsets = morphology (cranium / mandible) × difficulty
  (2-5 / 6-20 / 21-50 / 51-100 fragments)** (mandible has no
  51-100-fragment scenes).

## 4. Baselines

| Method | Type | Reference |
|---|---|---|
| RANSAC-ICP | geometric: FPFH + RANSAC global registration + ICP refinement | this work, built on [Open3D](https://www.open3d.org/) |
| Global | per-part + whole-cloud PointNet + MLP pose regression | [Schor et al., ICCV 2019](https://arxiv.org/abs/1811.07441) + [Li et al., AAAI 2020](https://arxiv.org/abs/1906.06693) |
| LSTM | PointNet + GRU seq2seq pose regression | [Wu et al., CVPR 2020](https://arxiv.org/abs/1911.10949) |
| DGL | PointNet + dynamic graph network, 3 rounds | [Huang et al., NeurIPS 2020](https://arxiv.org/abs/2006.07793) |
| PHFormer | proxy-level hybrid transformer with adjacency-aware pose estimation | [Cui et al., AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/27905) |
| DiffAssemble | graph diffusion for joint pose denoising | [Scarpellini et al., CVPR 2024](https://arxiv.org/abs/2402.19302) |
| GARF | SE(3) flow matching with anchored reference fragment | [Li et al., ICCV 2025](https://arxiv.org/abs/2504.05400) |
| RPF | rectified point flow over the fragment point clouds | [Sun et al., NeurIPS 2025](https://arxiv.org/abs/2506.05282) |

All learned baselines are trained with the unified protocol above
(`MAX_PARTS = 100`). Training settings per method are documented in
[baselines.md](baselines.md).

## 5. Submission format

Every baseline converts its raw predictions into one JSON file per scene:

```json
{
  "scene": "000102166_v00",
  "method": "dgl",
  "fragments": {
    "frag_00": {"R": [[...3x3...]], "t": [x, y, z]},
    "frag_01": {"R": [[...3x3...]], "t": [x, y, z]}
  }
}
```

`R` maps fragment local coordinates to world coordinates; `t` is in
normalized units. The evaluator only consumes this format, so every method is
measured with the same protocol.

## 6. Dataset layout

The archive is released in compact binary containers (xz-compressed): the
ascii meshes of the full dataset exceed a terabyte, while the same scenes fit
in about 36 GB in the released form. One scene directory is:

```
dataset/<split>/<object_id>/<variant>/
  intact.hfm.xz               complete mesh, canonical pose
  fragments/frag_XX.hfm.xz    fragments in local coordinates
  gt/transforms.json          per-fragment SE(3) from the assembly frame
                              (the frame where all fragments are born in
                              place, before the post-fracture dynamics)
  gt/adjacency.json           adjacency graph with interface areas
  gt/contact_faces/*.hfc.xz   interface of each adjacent pair: the indices of
                              the parent fragment faces that form it
  metadata.json               simulator bookkeeping and provenance
```

The two containers are documented in the dataset README. To get the ascii
layout that this repository reads, unpack a scene first:

```bash
python tools/hf_pack.py unpack --src <split>/<object_id>/<variant> --dst OUT
```

`<object_id>` is the MorphoSource media identifier of one fracture object; the
archive is keyed by media identifier and not by USNM specimen number, which is
recorded in `metadata.json` under `provenance.physical_object_title`.  (The
unpack utility ships with the dataset archive as `tools/hf_pack.py`; it is not
part of this repository.) `metadata.json` also carries `fragments`
(the fragment count) and `n_objects` (the simulator's rigid bodies, which is
`fragments + 2` because the scene contains the breakable specimen body and the
passive ground). The impact configuration of a scene is the entry with the same
`<object_id>_<variant>` identifier in `configurations.json`.
