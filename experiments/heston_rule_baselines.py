"""Cost-aware rule baselines under Heston (referee comment: baseline too weak).

Rules (one new ATM hedge option per day, like the RL agent; stock leg keeps delta neutral):
    n_gamma = contracts that make portfolio gamma zero   (full Delta-Gamma baseline)
    n_vega  = contracts that make portfolio vega zero    (full Delta-Vega baseline)
    (a) fraction rule:       n = clip(a * n_gamma + b * n_vega, [low, high])
    (b) fraction + band:     same, but no option trade when |n| < w contracts
    [low, high] = [min(0, n_gamma, n_vega), max(0, n_gamma, n_vega)] is the RL agent's action range.

Parameters are tuned on validation paths (one seed) and reported on fresh test paths
(another seed).  Risk = VaR95 / CVaR95 of the episode loss (-P&L).  Gains are
benchmark - rule (positive = rule is better), paired bootstrap standard errors.

The episode P&L is computed with arrays taken from the environment's own MainPortfolio
(``--check_env`` replays MainPortfolio.step and checks the vectorised P&L matches).

Costs:  option cost = spread * |premium traded| (as in Portfolio.add)
        delta cost  = c_delta * |change in stock position| * S_t (new, default 0)
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from absl import flags

from environment.utils import Utils

# Trading.py reads FLAGS.vega_obs in get_state; define it when used as a script.
if "vega_obs" not in flags.FLAGS:
    flags.DEFINE_boolean("vega_obs", True, "")
flags.FLAGS(["heston_rule_baselines"])
from environment.Trading import MainPortfolio  # noqa: E402

SPREADS = [0.005, 0.01, 0.02]
DELTA_COSTS_BP = [0, 1, 2, 5, 10]
A_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)
B_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)
W_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]  # band, in hedge-option contracts


def paper_utils(seed, n_paths, backend="table", **heston):
    p = dict(init_vol=0.3, kappa=1.0, theta=0.09, volvol=0.3, rho=-0.7)
    p.update(heston)
    return Utils(
        S0=10.0, K=10.0, init_ttm=30, r=0.0, mu=0.0, q=0.0, spread=0.0,
        ttms=[60], poisson_rate=1.0, moneyness_mean=1.0, moneyness_std=0.0,
        num_conts_to_add=-1, contract_size=100, hed_ttm=30, hed_type="American",
        stochastic_process="Heston", time_to_simulate=30, num_sim=n_paths, frq=1,
        TradingDaysPerYear=252, numerical_accuracy="low", n_jobs=-1, np_seed=seed,
        action_low=0.0, action_high=1.0, pricing_backend=backend, **p,
    )


def build_market(utils):
    """Arrays from the environment's MainPortfolio (same random stream as TradingEnv)."""
    np.random.seed(utils.seed)  # TradingEnv.__init__ seeds again before MainPortfolio
    mp = MainPortfolio(utils)
    liab = mp.liab_port.options
    L = liab[0].num_contract * liab[0].contract_size
    hed = mp.hed_port.options  # (P, n_hedge) Option objects
    P, H = hed.shape
    stack = lambda attr: np.stack([[getattr(o, attr) for o in row] for row in hed]).transpose(0, 2, 1)
    m = dict(
        S=mp.a_price, V=mp.vol,
        l_price=L * np.stack([o.price_path for o in liab]),
        l_next=L * np.stack([o.next_price_path for o in liab]),
        l_delta=L * np.stack([o.delta_path for o in liab]),
        l_gamma=L * np.stack([o.gamma_path for o in liab]),
        l_vega=L * np.stack([o.vega_path for o in liab]),
        # hedge options per contract: (P, time, issue)
        h_price=stack("price_path") * utils.contract_size,
        h_delta=stack("delta_path") * utils.contract_size,
        h_gamma=stack("gamma_path") * utils.contract_size,
        h_vega=stack("vega_path") * utils.contract_size,
        h_inactive=stack("inactive"),
    )
    return m, mp


def simulate(m, a, b, w, mode="rule"):
    """Vectorised episode P&L.  Returns pnl (before costs), option cost base, delta-trade base."""
    S = m["S"]
    P, T1 = S.shape
    T = T1 - 1  # number of hedging steps (environment stops at t = num_period - 1)
    n = np.zeros((P, T1))  # contracts held of each issued hedge option
    pnl = np.zeros(P)
    opt_base = np.zeros(P)
    dlt_base = np.zeros(P)
    pos_prev = np.zeros(P)
    for t in range(T):
        act = ~m["h_inactive"][:, t, :]
        G = m["l_gamma"][:, t] + np.sum(n * m["h_gamma"][:, t, :] * act, axis=1)
        Vg = m["l_vega"][:, t] + np.sum(n * m["h_vega"][:, t, :] * act, axis=1)
        g_h, v_h = m["h_gamma"][:, t, t], m["h_vega"][:, t, t]
        n_g = np.nan_to_num(-G / g_h)
        n_v = -Vg / v_h
        lo = np.minimum(0, np.minimum(n_g, n_v))
        hi = np.maximum(0, np.maximum(n_g, n_v))
        if mode == "gamma":
            x = n_g
        elif mode == "vega":
            x = n_v
        elif mode == "delta":
            x = np.zeros(P)
        else:
            x = np.clip(a * n_g + b * n_v, lo, hi)
            x = np.where(np.abs(x) < w, 0.0, x)
        n[:, t] = x
        opt_base += np.abs(x * m["h_price"][:, t, t])
        act_next = ~m["h_inactive"][:, t + 1, :]
        pos = -(m["l_delta"][:, t] + np.sum(n * m["h_delta"][:, t, :] * act, axis=1))
        dlt_base += np.abs(pos - pos_prev) * S[:, t]
        pos_prev = pos
        pnl += m["l_next"][:, t] - m["l_price"][:, t]
        pnl += np.sum(n * (m["h_price"][:, t + 1, :] - m["h_price"][:, t, :]) * act_next, axis=1)
        pnl += pos * (S[:, t + 1] - S[:, t])
    return pnl, opt_base, dlt_base


def total_pnl(res, spread, dcost_bp):
    pnl, ob, db = res
    return pnl - spread * ob - dcost_bp * 1e-4 * db


def var_cvar(pnl, alpha=0.95):
    loss = -np.asarray(pnl)
    v = np.quantile(loss, alpha, axis=-1)
    tail = loss >= v[..., None] if loss.ndim > 1 else loss >= v
    c = (loss * tail).sum(-1) / tail.sum(-1)
    return v, c


def boot_idx(P, B, seed=0):
    return np.random.default_rng(seed).integers(0, P, (B, P))


# ------------------------------------------------------------------------------------------------
def check_env(utils, n_check=16):
    """Replay MainPortfolio.step for a few episodes and compare with the vectorised P&L."""
    m, mp = build_market(utils)
    out = []
    for mode, (a, b, w) in [("gamma", (1, 0, 0)), ("vega", (0, 1, 0)), ("rule", (0.6, 0.3, 0.2))]:
        vec = simulate(m, a, b, w, mode)
        for ep in range(n_check):
            mp.reset(ep)
            mp.liab_port.add(ep, 0, utils.num_conts_to_add)
            tot, cost, pos_prev, dbase = 0.0, 0.0, 0.0, 0.0

            class R:  # StepResult stand-in
                pass

            for t in range(mp.a_price.shape[1] - 1):
                G, V = mp.get_gamma(t), mp.get_vega(t)
                ho = mp.hed_port.options[ep, t]
                n_g = -G / ho.gamma_path[t] / utils.contract_size
                n_v = -V / ho.vega_path[t] / utils.contract_size
                lo, hi = min(0, n_g, n_v), max(0, n_g, n_v)
                x = {"gamma": n_g, "vega": n_v}.get(mode, float(np.clip(a * n_g + b * n_v, lo, hi)))
                if mode == "rule" and abs(x) < w:
                    x = 0.0
                cost += abs(x * ho.price_path[t] * utils.contract_size)
                r, _ = mp.step(x, t, R())
                tot += r
                dbase += abs(mp.underlying.position - pos_prev) * mp.a_price[ep, t]
                pos_prev = mp.underlying.position
            out.append(dict(mode=mode, ep=ep, env_pnl=tot, vec_pnl=vec[0][ep], env_cost=cost, vec_cost=vec[1][ep],
                            env_dbase=dbase, vec_dbase=vec[2][ep]))
    df = pd.DataFrame(out)
    df["max_abs_diff"] = np.maximum.reduce([(df.env_pnl - df.vec_pnl).abs(), (df.env_cost - df.vec_cost).abs(),
                                            (df.env_dbase - df.vec_dbase).abs()])
    return df


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_paths", type=int, default=8192)
    ap.add_argument("--val_seed", type=int, default=2025)
    ap.add_argument("--test_seed", type=int, default=654)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="review/results/heston_rule_baselines")
    ap.add_argument("--check_env", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.check_env:
        df = check_env(paper_utils(args.val_seed, 16))
        df.to_csv(out / "env_check.csv", index=False)
        print(df.groupby("mode").max_abs_diff.max())
        return

    t0 = time.time()
    val, _ = build_market(paper_utils(args.val_seed, args.n_paths))
    test, _ = build_market(paper_utils(args.test_seed, args.n_paths))
    t_build = time.time() - t0
    print(f"built validation + test markets in {t_build:.0f}s")

    cells = list(itertools.product(SPREADS, DELTA_COSTS_BP))
    # ---------------- tuning on validation paths ----------------
    rows = []
    grid = [(a, b, 0.0) for a in A_GRID for b in B_GRID] + \
           [(a, b, w) for a in A_GRID for b in B_GRID for w in W_GRID if w > 0]
    for k, (a, b, w) in enumerate(grid):
        res = simulate(val, a, b, w)
        pnls = np.stack([total_pnl(res, s, d) for s, d in cells])
        v, c = var_cvar(pnls)
        for i, (s, d) in enumerate(cells):
            rows.append(dict(a=a, b=b, w=w, spread=s, dcost_bp=d, VaR95=v[i], CVaR95=c[i], mean=pnls[i].mean()))
        if k % 200 == 0:
            print(f"  grid {k}/{len(grid)}  {time.time() - t0:.0f}s")
    tune = pd.DataFrame(rows)
    tune.to_csv(out / "validation_grid.csv", index=False)

    # ---------------- test ----------------
    P = test["S"].shape[0]
    bidx = boot_idx(P, args.n_boot, seed=1)
    bench = {k: simulate(test, 0, 0, 0, k) for k in ["gamma", "vega", "delta"]}
    sel_rows, res_rows = [], []
    for (s, d) in cells:
        g = tune[(tune.spread == s) & (tune.dcost_bp == d)]
        bpnl = {k: total_pnl(r, s, d) for k, r in bench.items()}
        for metric in ["VaR95", "CVaR95"]:
            for rule, gg in [("(a) fraction", g[g.w == 0]), ("(b) fraction + band", g)]:
                best = gg.loc[gg[metric].idxmin()]
                sel_rows.append(dict(spread=s, dcost_bp=d, objective=metric, rule=rule, a=best.a, b=best.b,
                                     w=best.w, val_value=best[metric]))
                rp = total_pnl(simulate(test, best.a, best.b, best.w), s, d)
                mi = 0 if metric == "VaR95" else 1
                r_val = var_cvar(rp)[mi]
                r_se = var_cvar(rp[bidx])[mi].std(ddof=1)
                row = dict(spread=s, dcost_bp=d, metric=metric, rule=rule, a=best.a, b=best.b, w=best.w,
                           rule_value=r_val, rule_se=r_se, rule_mean_pnl=rp.mean())
                for k, name in [("gamma", "DG"), ("vega", "DV"), ("delta", "D")]:
                    bp = bpnl[k]
                    b_val = var_cvar(bp)[mi]
                    gain_boot = var_cvar(bp[bidx])[mi] - var_cvar(rp[bidx])[mi]
                    row[f"{name}_value"] = b_val
                    row[f"{name}_se"] = var_cvar(bp[bidx])[mi].std(ddof=1)
                    row[f"gain_vs_{name}"] = b_val - r_val
                    row[f"gain_vs_{name}_se"] = gain_boot.std(ddof=1)
                    row[f"gain_vs_{name}_pct"] = 100 * (b_val - r_val) / b_val
                    row[f"{name}_mean_pnl"] = bp.mean()
                res_rows.append(row)
    pd.DataFrame(sel_rows).to_csv(out / "selected_params.csv", index=False)
    res = pd.DataFrame(res_rows)
    res.to_csv(out / "test_results.csv", index=False)
    meta = dict(n_paths=args.n_paths, val_seed=args.val_seed, test_seed=args.test_seed, n_boot=args.n_boot,
                grid_size=len(grid), seconds=time.time() - t0, market_build_seconds=t_build)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(res[res.dcost_bp == 0].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
