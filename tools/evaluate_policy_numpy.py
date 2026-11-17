#!/usr/bin/env python
"""Evaluate an exported hedging policy with numpy only (no torch, no QuantLib, no repo code).

    python evaluate_policy_numpy.py --policy policy.npz --market market_test.npz --spread 0.01

policy.npz : written by train_d4pg.py (actor weights + meta_json)
market.npz : written by train_d4pg.py / tools/export_market.py (paths, prices and Greeks of the
             liability and of the hedge options; see environment/market.py for the keys)

The environment below is a copy of torch_d4pg/hedge_env.py (the test suite checks they agree):
state [S, portfolio gamma, hedge-option gamma, (portfolio vega, hedge-option vega)], action
a in [0, 1] mapped to n = low + a (high - low) contracts of the new ATM hedge option, between
0 and the gamma-neutral (and vega-neutral) amounts; the stock keeps the book delta neutral;
reward = P&L minus spread x |premium traded|.  Metrics: loss = -episode P&L, VaR95 = 95% quantile
of the loss, CVaR95 = mean loss at or beyond VaR95; bootstrap standard errors over paths.
"""

import argparse
import json

import numpy as np


# ------------------------------------------------------------------------------- policy
class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z[k] for k in z.files}
        self.meta = json.loads(str(self.p["meta_json"]))

    def __call__(self, obs):
        p = self.p
        if "obs_center" in p:
            obs = (obs - p["obs_center"]) / p["obs_scale"]
            if float(p["obs_clip"]) > 0:
                obs = np.clip(obs, -float(p["obs_clip"]), float(p["obs_clip"]))
        h = obs @ p["W0"] + p["b0"]
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        h = (h - mu) / np.sqrt(var + float(p["ln_eps"])) * p["ln_gamma"] + p["ln_beta"]
        h = np.tanh(h)
        for i in range(1, int(p["n_hidden"])):
            h = h @ p[f"W{i}"] + p[f"b{i}"]
            h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))  # ELU
        u = np.tanh(h @ p["W_out"] + p["b_out"])
        return (u[:, 0] + 1.0) / 2.0  # hedge fraction in [0, 1]


# ------------------------------------------------------------------------------- environment
def run_episodes(policy, m, spread, vega_obs, return_actions=False):
    S = np.asarray(m["S"], dtype=float)
    P, T1 = S.shape
    T = T1 - 1
    n = np.zeros((P, T1))
    pnl = np.zeros(P)
    cost = np.zeros(P)
    actions = np.zeros((P, T))
    for t in range(T):
        act = ~m["h_inactive"][:, t, :]
        G = m["l_gamma"][:, t] + np.sum(n * m["h_gamma"][:, t, :] * act, axis=1)
        V = m["l_vega"][:, t] + np.sum(n * m["h_vega"][:, t, :] * act, axis=1)
        g_h, v_h = m["h_gamma"][:, t, t], m["h_vega"][:, t, t]
        cols = [S[:, t], G, g_h] + ([V, v_h] if vega_obs else [])
        obs = np.stack(cols, axis=1).astype(np.float32).astype(float)
        a = np.clip(policy(obs), 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_g = np.nan_to_num(-G / g_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
            lo, hi = np.minimum(0.0, n_g), np.maximum(0.0, n_g)
            if vega_obs:
                n_v = -V / v_h
                lo, hi = np.minimum(lo, n_v), np.maximum(hi, n_v)
        x = lo + a * (hi - lo)
        n[:, t] = x
        actions[:, t] = a
        c = np.abs(spread * x * m["h_price"][:, t, t])
        act_next = ~m["h_inactive"][:, t + 1, :]
        pos = -(m["l_delta"][:, t] + np.sum(n * m["h_delta"][:, t, :] * act, axis=1))
        pnl += -c + (m["l_next"][:, t] - m["l_price"][:, t])
        pnl += np.sum(n * (m["h_price"][:, t + 1, :] - m["h_price"][:, t, :]) * act_next, axis=1)
        pnl += pos * (S[:, t + 1] - S[:, t])
        cost += c
    return (pnl, cost, actions) if return_actions else (pnl, cost)


# ------------------------------------------------------------------------------- metrics
def var_cvar(pnl, alpha=0.95):
    loss = -np.asarray(pnl)
    v = np.quantile(loss, alpha, axis=-1)
    tail = loss >= v[..., None] if loss.ndim > 1 else loss >= v
    return v, (loss * tail).sum(-1) / tail.sum(-1)


def summarize(pnl, cost, n_boot=1000, seed=1, std_coef=1.645):
    P = len(pnl)
    bs = pnl[np.random.default_rng(seed).integers(0, P, (n_boot, P))]
    v, c = var_cvar(pnl)
    bv, bc = var_cvar(bs)
    return dict(
        n_paths=P,
        mean_pnl=pnl.mean(), mean_pnl_se=pnl.std(ddof=1) / np.sqrt(P),
        std_pnl=pnl.std(), std_pnl_se=bs.std(1).std(ddof=1),
        VaR95=v, VaR95_se=bv.std(ddof=1), CVaR95=c, CVaR95_se=bc.std(ddof=1),
        meanstd=pnl.mean() - std_coef * pnl.std(),
        meanstd_se=(bs.mean(1) - std_coef * bs.std(1)).std(ddof=1),
        mean_cost=cost.mean(), mean_cost_se=cost.std(ddof=1) / np.sqrt(P),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--market", required=True)
    ap.add_argument("--spread", type=float, default=None, help="option cost (default: the one in policy meta)")
    ap.add_argument("--vega_obs", type=int, default=None, help="default: the one in policy meta")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out_csv", default=None, help="optional: write per-path P&L and cost")
    args = ap.parse_args()

    pol = NumpyPolicy(args.policy)
    spread = pol.meta["spread"] if args.spread is None else args.spread
    vega_obs = bool(pol.meta["vega_obs"] if args.vega_obs is None else args.vega_obs)
    z = np.load(args.market)
    m = {k: z[k] for k in z.files}
    pnl, cost = run_episodes(pol, m, spread, vega_obs)
    res = summarize(pnl, cost, args.n_boot)
    print(f"policy: {args.policy}\nmarket: {args.market}  spread={spread}  vega_obs={vega_obs}")
    for k in ["mean_pnl", "std_pnl", "VaR95", "CVaR95", "meanstd", "mean_cost"]:
        print(f"  {k:10s} {res[k]:10.4f}  (se {res[k + '_se']:.4f})")
    if args.out_csv:
        np.savetxt(args.out_csv, np.c_[pnl, cost], delimiter=",", header="pnl,cost", comments="")


if __name__ == "__main__":
    main()
