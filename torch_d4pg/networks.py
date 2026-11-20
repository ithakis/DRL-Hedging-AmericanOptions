"""Actor and quantile critic in PyTorch.

Layer layout follows the original acme/Sonnet networks of run.py (make_quantile_networks), so the
paper's method description stays valid:

    LayerNormMLP(sizes) = Linear(sizes[0]) -> LayerNorm -> tanh -> [Linear -> ELU] for sizes[1:]
                          (acme.tf.networks.LayerNormMLP, activate_final=True,
                           w_init = VarianceScaling(1/3, fan_in, uniform))
    policy  = LayerNormMLP(256,256,256) -> NearZeroInitializedLinear(1) -> tanh   (action in [-1, 1])
    critic  = concat(obs, action) -> LayerNormMLP(512,512,256) -> Linear(n_quantiles)

The policy works in acme's canonical action space [-1, 1]; the environment action (hedge fraction
in [0, 1]) is (u + 1) / 2, as done by acme's CanonicalSpecWrapper in the original run.py.
The structure (separate Actor / Critic modules, target copies, Adam) follows the swing D4PG
(DRL-Swing-Options/src/networks.py); its swing-specific parts (profitability gate, HHK input
layer) are not used here.
"""

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def _variance_scaling_uniform_(w: torch.Tensor, scale: float) -> None:
    fan_in = w.shape[1]
    limit = math.sqrt(3.0 * scale / fan_in)
    nn.init.uniform_(w, -limit, limit)


def _truncated_normal_fan_in_(w: torch.Tensor, scale: float) -> None:
    # tf VarianceScaling(scale, fan_in, truncated_normal): stddev = sqrt(scale/fan_in)/0.8796
    std = math.sqrt(scale / w.shape[1]) / 0.87962566103423978
    nn.init.trunc_normal_(w, 0.0, std, -2 * std, 2 * std)


class LayerNormMLP(nn.Module):
    def __init__(self, in_dim: int, sizes: Sequence[int]):
        super().__init__()
        sizes = list(sizes)
        self.first = nn.Linear(in_dim, sizes[0])
        self.norm = nn.LayerNorm(sizes[0], eps=1e-5)  # snt.LayerNorm default eps 1e-5
        rest = []
        d = sizes[0]
        for h in sizes[1:]:
            rest.append(nn.Linear(d, h))
            d = h
        self.rest = nn.ModuleList(rest)
        self.out_dim = d
        for lin in [self.first, *self.rest]:
            _variance_scaling_uniform_(lin.weight, 1.0 / 3.0)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        x = torch.tanh(self.norm(self.first(x)))
        for lin in self.rest:
            x = torch.nn.functional.elu(lin(x))
        return x


class ObsNormalizer(nn.Module):
    """Fixed robust scaling x_hat = clip((x - median) / IQR, -clip, clip), as the swing D4PG's
    RobustNormalizer. Median and IQR come from the TRAINING paths only (train_d4pg.obs_stats).
    Identity when center = 0, scale = 1, clip = 0."""

    def __init__(self, obs_dim: int, center=None, scale=None, clip: float = 10.0):
        super().__init__()
        c = torch.zeros(obs_dim) if center is None else torch.as_tensor(center, dtype=torch.float32)
        sc = torch.ones(obs_dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32)
        self.register_buffer("center", c.clone())
        self.register_buffer("scale", sc.clone())
        self.clip = float(clip) if center is not None else 0.0

    def forward(self, x):
        x = (x - self.center) / self.scale
        return x.clamp(-self.clip, self.clip) if self.clip > 0 else x


class Actor(nn.Module):
    """Deterministic policy, output u in [-1, 1] (env action = (u + 1) / 2)."""

    def __init__(self, obs_dim: int, sizes: Sequence[int] = (256, 256, 256), norm: ObsNormalizer = None):
        super().__init__()
        self.norm_in = norm if norm is not None else ObsNormalizer(obs_dim)
        self.body = LayerNormMLP(obs_dim, sizes)
        self.head = nn.Linear(self.body.out_dim, 1)
        _truncated_normal_fan_in_(self.head.weight, 1e-4)  # acme NearZeroInitializedLinear
        nn.init.zeros_(self.head.bias)

    def preact(self, obs):
        return self.head(self.body(self.norm_in(obs)))

    def forward(self, obs):
        return torch.tanh(self.preact(obs))


def quantile_grid(interval: float = 0.01, tail: bool = False):
    """Quantile levels and MID probabilities exactly as agent/distributional.py (99 levels).
    tail=True: denser levels in the loss tail (0.25% steps below 10%, 2% steps above; 84 levels)."""
    q = np.arange(interval, 1.0, interval)
    if tail:
        q = np.r_[np.arange(0.0025, 0.0999, 0.0025), np.arange(0.10, 0.9901, 0.02)]
    left = q - np.insert(q[:-1], 0, 0.0)
    right = np.insert(q[1:], len(q) - 1, 1.0) - q
    return q, (left + right) / 2


class QuantileCritic(nn.Module):
    """Returns the quantile values Z(s, a) of shape (batch, n_quantiles)."""

    def __init__(self, obs_dim: int, sizes: Sequence[int] = (512, 512, 256), quantile_interval: float = 0.01,
                 norm: ObsNormalizer = None, tail_quantiles: bool = False):
        super().__init__()
        self.norm_in = norm if norm is not None else ObsNormalizer(obs_dim)
        q, p = quantile_grid(quantile_interval, tail_quantiles)
        self.register_buffer("quantiles", torch.tensor(q, dtype=torch.float32))
        self.register_buffer("probs", torch.tensor(p, dtype=torch.float32))
        self.body = LayerNormMLP(obs_dim + 1, sizes)
        self.head = nn.Linear(self.body.out_dim, len(q))
        _truncated_normal_fan_in_(self.head.weight, 1.0)  # snt.Linear default: TruncatedNormal(1/sqrt(fan_in))
        nn.init.zeros_(self.head.bias)

    def forward(self, obs, action):
        return self.head(self.body(torch.cat([self.norm_in(obs), action], dim=-1)))
