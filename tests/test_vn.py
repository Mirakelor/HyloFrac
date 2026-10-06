"""Tests for the equivariant (vector-neuron) backbone."""

import numpy as np
import torch

# small footprint for shared workstations (16-thread torch defaults can
# momentarily allocate GBs of arena memory under memory pressure)
torch.set_num_threads(2)


def _random_rotation(rng: np.random.Generator) -> torch.Tensor:
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    x, y, z, w = q
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return torch.from_numpy(r).float()


def test_vn_linear_equivariance():
    from hylofrac.baselines.vn_layers import VNLinear

    rng = np.random.default_rng(0)
    layer = VNLinear(4, 8)
    r = _random_rotation(rng)
    x = torch.randn(3, 4, 3)
    # rotating the input rotates the output the same way (row-vector frames:
    # v' = v @ R, so layer(x @ R) == layer(x) @ R)
    assert torch.allclose(layer(x @ r), layer(x) @ r, atol=1e-5)


def test_vn_encoder_equivariance():
    from hylofrac.baselines.vn_layers import VNDGCNNEncoder

    rng = np.random.default_rng(1)
    encoder = VNDGCNNEncoder()
    encoder.eval()
    r = _random_rotation(rng)
    pcs = torch.randn(1, 2, 64, 3)
    with torch.no_grad():
        code = encoder(pcs)
        code_rot = encoder(pcs @ r)
    # equivariance: code(R x) == R code(x), with row-vector convention;
    # the flat code is a stack of 256 3-dim vectors, rotate per vector
    code_v = code.reshape(1, 2, -1, 3)
    err = (code_rot.reshape_as(code_v) - code_v @ r).norm() / code_v.norm()
    # The kNN grouping selects discrete neighbors, so near-tie distances can
    # flip a neighbor at floating-point precision (errors ~1e-2), while a
    # broken non-equivariant layer fails at O(1); the bound separates the two.
    assert err < 5e-2


def test_vn_encoder_shape():
    from hylofrac.baselines.vn_layers import VNDGCNNEncoder

    encoder = VNDGCNNEncoder()
    code = encoder(torch.randn(1, 3, 50, 3))
    assert code.shape == (1, 3, 768)
