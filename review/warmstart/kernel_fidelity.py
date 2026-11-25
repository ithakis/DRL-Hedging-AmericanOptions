"""Fidelity test of the quadrature target (plan step 1).

For transitions of the validation market under random hedge fractions, compare the quadrature mean and
std of the one-step reward (and of the next portfolio gamma / vega) with nested Monte Carlo: n_mc draws
of the SAME QE step from the same state (pseudo-random normals/uniforms instead of the market's Sobol).
Pass rule (fixed in the plan): |mean bias| < 10% of the reward std, std ratio in [0.9, 1.1].

    PYTHONPATH=. python review/warmstart/kernel_fidelity.py --table_dir ../DRL-Hedging-retrain/pricing_tables
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import train_d4pg as T
from review.retrain.settings import BLOCKS
from torch_d4pg.hedge_env import HedgingEnv
from torch_d4pg.heston_kernel import HestonKernel, book_arrivals


def qe_mc(K, S, vol, n, rng):
    """n pseudo-random draws of the market's QE step from (S, vol): returns S1, vol1, w (B, n)."""
    v = vol ** 2
    E, k, th, xi = K.E, K.kappa, K.theta, K.xi
    mm = th + (v - th) * E
    s2 = v * xi ** 2 * E / k * (1 - E) + th * xi ** 2 / (2 * k) * (1 - E) ** 2
    psi = s2 / mm ** 2
    B = len(v)
    Z1, Z2, U = rng.standard_normal((B, n)), rng.standard_normal((B, n)), rng.uniform(size=(B, n))
    v1 = np.zeros((B, n)); K0s = np.zeros(B)
    lo = psi <= K.psiC
    ps = psi[lo]; b2 = 2 / ps - 1 + np.sqrt(2 / ps * (2 / ps - 1)); b = np.sqrt(b2); a = mm[lo] / (1 + b2)
    v1[lo] = a[:, None] * (b[:, None] + Z1[lo]) ** 2
    K0s[lo] = -(K.A * b2 * a) / (1 - 2 * K.A * a) + 0.5 * np.log(1 - 2 * K.A * a)
    hi = ~lo
    if hi.any():
        ps = psi[hi]; pp = (ps - 1) / (ps + 1); beta = (1 - pp) / mm[hi]
        v1[hi] = np.where(U[hi] <= pp[:, None], 0.0, np.log((1 - pp[:, None]) / (1 - U[hi])) / beta[:, None])
        K0s[hi] = -np.log(pp + beta * (1 - pp) / (beta - K.A))
    K0s -= (K.K1 + 0.5 * K.K3) * v
    lnS1 = (np.log(S)[:, None] + K.drift + K0s[:, None] + K.K1 * v[:, None] + K.K2 * v1
            + np.sqrt(K.K3 * v[:, None] + K.K4 * v1) * Z2)
    return np.exp(lnS1), np.sqrt(v1), np.full((B, n), 1.0 / n)


def moments(r, w):
    m = (r * w).sum(1)
    return m, np.sqrt(np.maximum((w * (r - m[:, None]) ** 2).sum(1), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_dir", default="../DRL-Hedging-retrain/pricing_tables")
    ap.add_argument("--market_dir", default="../DRL-Hedging-retrain/markets")
    ap.add_argument("--spread", type=float, default=0.01)
    ap.add_argument("--n_paths", type=int, default=64)
    ap.add_argument("--n_mc", type=int, default=4096)
    ap.add_argument("--out", default="review/results/warmstart/kernel_fidelity.json")
    args = ap.parse_args()
    a = T.get_args(BLOCKS["B"]["flags"] + ["--obj_func", "var", "--spread", "0.0", "--market_dir", args.market_dir])
    u = T.make_utils(argparse.Namespace(**{**vars(a), "spread": 0.0}), a.val_seed, a.n_val)
    u.table_dir = args.table_dir
    m = T.get_market(a, a.val_seed, a.n_val)
    _, _, c = book_arrivals(u)
    rng = np.random.default_rng(0)
    res = {}
    kernels = {mx: HestonKernel(u, m, c, True, m_x=mx) for mx in (2, 3, 4, 5)}
    K0 = kernels[3]
    env = HedgingEnv(m, args.spread, True)
    p = rng.choice(env.P, args.n_paths, replace=False)
    obs = env.reset(p)
    rows = {mx: [] for mx in kernels}
    done = False
    while not done:
        t = env.t
        f = rng.uniform(0, 1, len(p))
        lo, hi = env.bounds()
        x = lo + f * (hi - lo)
        cost = -np.abs(args.spread * x * env.b["h_price"][:, t, t])
        _, _, done, _ = env.step(f)
        n = env.n.copy()
        tt = np.full(len(p), t)
        # nested Monte Carlo of the same QE step
        orig = K0.nodes
        K0.nodes = lambda S, vol: qe_mc(K0, S, vol, args.n_mc, rng)
        r_mc, o_mc, _, w_mc = K0.transition(p, tt, n, cost)
        K0.nodes = orig
        mmc, smc = moments(r_mc, w_mc)
        gmc, _ = moments(o_mc[..., 1], w_mc)
        for mx, K in kernels.items():
            r, o, _, w = K.transition(p, tt, n, cost)
            mk, sk = moments(r, w)
            gk, _ = moments(o[..., 1], w)
            # MC standard error of the nested mean, to tell bias from MC noise
            rows[mx].append(np.c_[mk - mmc, smc, sk / np.maximum(smc, 1e-12), smc / np.sqrt(args.n_mc),
                                  (gk - gmc) / (np.abs(gmc) + 1e-9)])
    for mx, rr in rows.items():
        x = np.concatenate(rr)
        rel_bias = np.abs(x[:, 0]) / np.maximum(x[:, 1], 1e-12)
        res[mx] = dict(nodes=mx * mx, transitions=len(x),
                       mean_abs_bias=float(np.mean(np.abs(x[:, 0]))), mean_reward_std=float(np.mean(x[:, 1])),
                       median_rel_bias=float(np.median(rel_bias)), p95_rel_bias=float(np.quantile(rel_bias, 0.95)),
                       median_std_ratio=float(np.median(x[:, 2])), p05_std_ratio=float(np.quantile(x[:, 2], 0.05)),
                       p95_std_ratio=float(np.quantile(x[:, 2], 0.95)), mc_se_mean=float(np.mean(x[:, 3])),
                       median_rel_gamma_bias=float(np.median(np.abs(x[:, 4]))))
        res[mx]["pass"] = bool(res[mx]["p95_rel_bias"] < 0.10 and 0.9 <= res[mx]["median_std_ratio"] <= 1.1)
        print(mx, json.dumps(res[mx]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
