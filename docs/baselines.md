# Baselines: Implementation Notes and Training Settings

HyloFrac evaluates eight baselines — the geometric RANSAC-ICP method and
seven learned methods ([benchmark.md](benchmark.md), section 4). The
geometric baseline needs no training and is run directly with
`hylofrac-predict --method geometric`; the sections below cover the learned
baselines. All learned baselines share the HyloFrac data protocol
([benchmark.md](benchmark.md), section 3): assembled-pose fragment meshes,
1,000 surface points per fragment, per-fragment random observation rotation,
translation / rotation GT relative to the anchor fragment (fragment 0),
fragment range 2-100
(`MAX_PARTS = 100`). Ground-truth rotations are given as rotation
matrices; internally the training tensors use scalar-first unit
quaternions. Each method's rotation parameterization follows its
entry below.

## Validation

Models are validated with the full HyloFrac metric set
([benchmark.md](benchmark.md), section 2): PA@0.01, RMSE(R) as the geodesic
rotation error, RMSE(T), symmetric Chamfer, Qpos, and adjacency P/R/F1
against the GT graph, computed by the evaluator in `hylofrac/eval`.

## Unified training configuration

All learned baselines are trained with the same schedule and data protocol;
the effective batch size is 32 for every method (physical batch x gradient
accumulation, per method below).

| Setting | Value |
|---|---|
| epochs | 200 |
| optimizer | Adam, lr 1e-3, cosine decay to 0 |
| weight decay | 0 |
| validation | every 10 epochs |
| effective batch size | 32 (physical batch x accumulation, below) |
| shared loss weights (Global / LSTM / DGL / PHFormer) | translation L2 1.0, rotation cosine 0.2, rotated-part point L2 1.0, per-part Chamfer 10.0, whole-shape Chamfer 10.0 |
| PHFormer auxiliary weights | adjacency MSE 1.0, relative rotation cosine 0.2, relative translation L2 1.0 |
| DiffAssemble loss weights | diffusion: translation L2 1.0, rotation cosine 0.2, assembly Chamfer 10.0 |
| GARF loss | flow-matching velocity MSE (zero target on the anchor) |
| RPF loss | rectified-flow velocity MSE (zero target on the anchor) |

| Method | physical batch x accumulation | architecture |
|---|---|---|
| Global | 8 x 4 | per-part PointNet + whole-cloud PointNet + MLP pose head |
| LSTM | 8 x 4 | PointNet encoder + GRU seq2seq (bidirectional encoder, hidden 256) |
| DGL | 8 x 4 | PointNet encoder + dynamic graph network, 3 rounds |
| PHFormer | 4 x 8 | PointNet2-SSG proxies + hybrid transformer (4 heads) |
| DiffAssemble | 2 x 16 | VN-DGCNN encoder + transformer GNN denoiser |
| GARF | 8 x 4 | blocked point-transformer encoder + denoiser transformer |
| RPF | 4 x 8 | blocked point-transformer encoder + fragment-token flow transformer with a point-wise velocity head |

## Methods

- **RANSAC-ICP** — geometric registration of each fragment against the
  growing assembly: FPFH feature matching, RANSAC global registration and
  point-to-plane ICP refinement, greedily adding the best-scoring pair per
  iteration. Fragment 0 is the anchor and is submitted with the identity
  pose; no training. Built on [Open3D](https://www.open3d.org/).
- **Global** — a PointNet encoder shared by the fragments plus a second
  PointNet over the whole scene cloud; an MLP regresses each fragment pose
  from its feature and the global feature. The architecture follows
  CompoNet, Schor et al., *Learning to Generate the Unseen by Part
  Synthesis and Composition*, ICCV 2019
  ([arXiv:1811.07441](https://arxiv.org/abs/1811.07441)), and PAGENet,
  Li et al., *Learning Part Generation and Assembly for Structure-aware
  Shape Synthesis*, AAAI 2020
  ([arXiv:1906.06693](https://arxiv.org/abs/1906.06693)). It is used as a
  reassembly baseline in the Breaking Bad benchmark
  ([project page](https://breaking-bad-dataset.github.io/)); the
  reference implementation follows the DGL codebase
  ([code](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly)).
- **LSTM** — a PointNet encoder feeding a GRU sequence-to-sequence model
  (two-layer bidirectional encoder, hidden 256 per direction;
  autoregressive decoder with teacher forcing) that regresses the
  fragment poses in fragment order. The sequence-to-sequence design
  follows PQ-Net, Wu et al., *PQ-Net: A Generative Part Seq2Seq Network
  for 3D Shapes*, CVPR 2020
  ([arXiv:1911.10949](https://arxiv.org/abs/1911.10949)).
- **DGL** — a dynamic graph network of 3 rounds with per-round parameters:
  each round re-estimates edge relations from the current fragment poses,
  propagates messages over the weighted graph, and regresses poses.
  Geometric-equivalence node merging is disabled because fragments are
  geometrically unique. Huang et al., *Generative 3D Part Assembly via
  Dynamic Graph Learning*, NeurIPS 2020
  ([arXiv:2006.07793](https://arxiv.org/abs/2006.07793),
  [code](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly)).
- **PHFormer** — proxy-level hybrid transformer with adjacency-aware
  hierarchical pose estimation; the intra-fragment self-attention layers
  use a sinusoidal position encoding of the proxy centers in voxel coordinates.
  Cui et al., *PHFormer: Multi-Fragment Assembly Using Proxy-Level Hybrid
  Transformer*, AAAI 2024
  ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/27905),
  [code](https://github.com/521piglet/PHFormer)).
- **DiffAssemble** — graph diffusion over fragment poses with an equivariant
  VN-DGCNN encoder; the denoiser is a transformer GNN over the fully
  connected fragment graph, and inference runs deterministic DDIM sampling
  over the 300 diffusion steps (ratio 10). Scarpellini et al., *DiffAssemble: A
  Unified Graph-Diffusion Model for 2D and 3D Reassembly*, CVPR 2024
  ([arXiv:2402.19302](https://arxiv.org/abs/2402.19302),
  [code](https://github.com/iit-pavis/diffassemble)).
- **GARF** — SE(3) flow matching with an anchored reference fragment (the
  fragment of a scene with the largest surface area is the anchor, pinned
  to its GT pose during inference; sigma is integrated from 1 to 0 in 20
  Euler steps), with a PointTransformerV3-style blocked point-transformer
  encoder.
  Li et al., *GARF: Learning Generalizable 3D Reassembly for Real-World
  Fractures*, ICCV 2025
  ([arXiv:2504.05400](https://arxiv.org/abs/2504.05400),
  [code](https://github.com/ai4ce/GARF)).
- **RPF** — rectified point flow: a conditional flow transports noisy
  points to their assembled positions, the fragment identity conditions
  every point of that fragment, and the fragment poses are recovered by a
  rigid fit of each transported fragment to its observation cloud. The
  anchor fragment is pinned to its GT pose during inference and the flow
  is integrated from t = 1 to 0 in 50 Euler steps. Sun et al.,
  *Rectified Point Flow: Generic Point Cloud Pose Estimation*,
  NeurIPS 2025 ([arXiv:2506.05282](https://arxiv.org/abs/2506.05282),
  [code](https://github.com/GradientSpaces/Rectified-Point-Flow)).
