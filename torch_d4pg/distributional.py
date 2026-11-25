"""Quantile-regression loss and risk objectives, ported from agent/distributional.py.

All formulas are kept as in the TensorFlow code (QuantileDistribution with MID probabilities,
99 quantile levels 0.01..0.99, each with probability 0.01):

    meanstd(Z) = mean(Z) - std_coef * std(Z)      mean = plain average of the 99 values,
                                                  var  = sum_i p_i (Z_i - mean)^2  (p_i = 0.01)
    var(Z)     = value at quantile level 1 - threshold (0.05 is a grid point -> Z_5)
    cvar(Z)    = sum_{i: cdf_i <= 1 - threshold} p_i Z_i   (NOT divided by 1 - threshold; as before)

Loss ('qr-huber'): Huber(k=1) quantile regression between the target values
z_t = r + d * Z'(s', a') and the online values Z(s, a), averaged over target and online quantiles.
"""

import numpy as np
import torch
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def _qh_kernel(z_pred, z_t, tau, k):
    """loss[b] = mean_{i,j} |tau_j - 1[d<0]| huber_k(d) / k,  d = z_t[b,i] - z_pred[b,j];
    grad[b,j] = d loss[b] / d z_pred[b,j]."""
    B, N = z_pred.shape
    Nt = z_t.shape[1]
    loss = np.zeros(B, dtype=np.float32)
    grad = np.zeros((B, N), dtype=np.float32)
    scale = 1.0 / (k * Nt * N)
    for b in prange(B):
        acc = 0.0
        for j in range(N):
            zp = z_pred[b, j]
            tj = tau[j]
            gacc = 0.0
            for i in range(Nt):
                d = z_t[b, i] - zp
                w = tj - 1.0 if d < 0 else tj
                if w < 0:
                    w = -w
                ad = d if d >= 0 else -d
                if ad < k:
                    acc += w * 0.5 * d * d
                    gacc -= w * d
                else:
                    acc += w * k * (ad - 0.5 * k)
                    gacc -= w * (k if d > 0 else -k)
            grad[b, j] = gacc * scale
        loss[b] = acc * scale
    return loss, grad


@njit(cache=True, fastmath=False, parallel=True)
def _qh_kernel_sorted(z_pred, z_t, tau, k):
    """Same loss and gradient as _qh_kernel, in O((N + N') log N') per row instead of O(N N').

    For one online value zp, the targets split into 4 groups by d = t - zp:  t < zp - k (linear, w = 1 - tau),
    zp - k <= t < zp (quadratic, w = 1 - tau), zp <= t < zp + k (quadratic, w = tau), t >= zp + k (linear, w = tau).
    (At |d| = k the quadratic and linear pieces agree in value and slope, and d = 0 adds nothing, so the
    tie side does not matter.)  With the targets sorted and prefix sums of 1, t, t^2, each group's sum
    of losses and of d/dzp is closed form.  float64 accumulation on values centred at the row mean."""
    B, N = z_pred.shape
    Nt = z_t.shape[1]
    loss = np.zeros(B, dtype=np.float32)
    grad = np.zeros((B, N), dtype=np.float32)
    scale = 1.0 / (k * Nt * N)
    for b in prange(B):
        c = 0.0
        for i in range(Nt):
            c += z_t[b, i]
        c /= Nt
        t = np.empty(Nt, dtype=np.float64)
        for i in range(Nt):
            t[i] = z_t[b, i] - c
        t.sort()
        S1 = np.zeros(Nt + 1)
        S2 = np.zeros(Nt + 1)
        for i in range(Nt):
            S1[i + 1] = S1[i] + t[i]
            S2[i + 1] = S2[i] + t[i] * t[i]
        acc = 0.0
        for j in range(N):
            zp = z_pred[b, j] - c
            tj = tau[j]
            i1 = np.searchsorted(t, zp - k)   # t < zp - k
            i2 = np.searchsorted(t, zp)       # t < zp
            i3 = np.searchsorted(t, zp + k)   # t < zp + k
            # A: linear, d < -k
            nA, s1A = i1, S1[i1]
            lA = k * (nA * (zp - 0.5 * k) - s1A)
            gA = k * nA
            # B: quadratic, -k <= d < 0
            nB, s1B, s2B = i2 - i1, S1[i2] - S1[i1], S2[i2] - S2[i1]
            lB = 0.5 * (s2B - 2.0 * zp * s1B + nB * zp * zp)
            gB = nB * zp - s1B
            # C: quadratic, 0 <= d < k
            nC, s1C, s2C = i3 - i2, S1[i3] - S1[i2], S2[i3] - S2[i2]
            lC = 0.5 * (s2C - 2.0 * zp * s1C + nC * zp * zp)
            gC = nC * zp - s1C
            # D: linear, d >= k
            nD, s1D = Nt - i3, S1[Nt] - S1[i3]
            lD = k * (s1D - nD * (zp + 0.5 * k))
            gD = -k * nD
            acc += (1.0 - tj) * (lA + lB) + tj * (lC + lD)
            grad[b, j] = ((1.0 - tj) * (gA + gB) + tj * (gC + gD)) * scale
        loss[b] = acc * scale
    return loss, grad


FAST_LOSS = True  # module switch: False = the original O(N N') kernel


class _QuantileHuber(torch.autograd.Function):
    """Fused forward/backward of quantile_huber_loss_reference (numba kernel, same values).

    The (N', B, N) pairwise tensor of the reference version dominated the learner step;
    here it is never materialised, and d/dz_pred = -w * clip(d, -k, k) / k is computed directly.
    """

    @staticmethod
    def forward(ctx, z_pred, z_t, quantiles, k):
        f = _qh_kernel_sorted if FAST_LOSS else _qh_kernel
        loss, grad = f(z_pred.detach().numpy(), z_t.detach().numpy(), quantiles.detach().numpy(), k)
        ctx.save_for_backward(torch.from_numpy(grad))
        return torch.from_numpy(loss)

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return grad * g.unsqueeze(-1), None, None, None


def quantile_huber_loss(z_pred, quantiles, r, d, z_next, k: float = 1.0):
    """z_pred (B, N), quantiles (N,), r (B,1), d (B,1), z_next (B, N) -> per-sample loss (B,).
    z_next must not require grad (target network)."""
    z_t = (r + d * z_next).detach()
    return _QuantileHuber.apply(z_pred, z_t, quantiles, float(k))


def quantile_huber_loss_reference(z_pred, quantiles, r, d, z_next, k: float = 1.0):
    """Direct port of agent/distributional.QuantileLoss (huber), used by the tests."""
    z_t = r + d * z_next                                  # (B, N')
    diff = z_t.t().unsqueeze(-1) - z_pred                 # (N', B, N)
    absd = diff.abs()
    huber = torch.where(absd < k, 0.5 * diff.pow(2), k * (absd - 0.5 * k)) / k
    w = (quantiles - (diff.detach() < 0).to(diff.dtype)).abs()
    return (huber * w).mean(dim=(0, 2))


class RiskObjective:
    def __init__(self, name: str, quantiles: torch.Tensor, probs: torch.Tensor, threshold: float = 0.95,
                 std_coef: float = 1.645, smooth: float = 0.0):
        self.name = name
        self.std_coef = std_coef
        self.probs = probs
        q_level = torch.tensor(1.0 - threshold, dtype=quantiles.dtype)
        cdf = torch.cumsum(probs, 0)
        self.cvar_mask = (cdf <= q_level).to(probs.dtype)
        # VaR: quantile projection of agent/distributional.py; exact grid point -> pick it,
        # otherwise linear interpolation between the two neighbours.
        hit = torch.nonzero(quantiles == q_level).flatten()
        if len(hit) == 1:
            self.var_w = torch.zeros_like(quantiles)
            self.var_w[hit[0]] = 1.0
        else:
            j = int(torch.searchsorted(quantiles, q_level).clamp(1, len(quantiles) - 1))
            lam = float((q_level - quantiles[j - 1]) / (quantiles[j] - quantiles[j - 1]))
            self.var_w = torch.zeros_like(quantiles)
            self.var_w[j - 1], self.var_w[j] = 1 - lam, lam
        if smooth > 0:  # triangular kernel of half-width `smooth` around the VaR level (denser actor gradient)
            w = torch.clamp(1.0 - (quantiles - q_level).abs() / smooth, min=0.0) * probs
            self.var_w = w / w.sum()
        if name not in ("meanstd", "var", "cvar", "mean"):
            raise ValueError(f"unknown objective {name}")

    def __call__(self, z):
        if self.name == "meanstd":
            mean = z.mean(-1)
            var = (self.probs * (z - mean.unsqueeze(-1)).pow(2)).sum(-1)
            return mean - self.std_coef * var.sqrt()
        if self.name == "var":
            return (z * self.var_w).sum(-1)
        if self.name == "mean":
            return z.mean(-1)
        return (z * self.probs * self.cvar_mask).sum(-1)
