"""Richer rule baseline (Sven: "is the gain from extra parameters?").

Rule (b) with blend weights and band that depend on time to horizon: separate (a, b, w) for the first
and the second half of the 30 days (6 parameters). Tuned on VALIDATION terminal VaR95 by coordinate
search over the same grids as rule (b) (a, b in 0..1 step 0.1, w in W_GRID), starting from the tuned
rule (b) and then from random grid points (restarts), with a budget of about 1,100 rollouts per cost (rule (b)'s full grid is 11 x 11 x 9 = 1,089).
Writes review/results/warmstart/richer_rule_params.json. No test paths are used.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from review.retrain import analysis as A
from review.retrain.settings import COSTS

OUT = Path("review/results/warmstart")
BUDGET = 1100
T_SPLIT = 15


def decider(p):
    (a1, b1, w1), (a2, b2, w2) = p[:3], p[3:]

    def f(t, obs, n_g, n_v, lo, hi):
        a, b, w = (a1, b1, w1) if t < T_SPLIT else (a2, b2, w2)
        lo2 = np.minimum(0, np.minimum(n_g, n_v))
        hi2 = np.maximum(0, np.maximum(n_g, n_v))
        x = np.clip(a * n_g + b * n_v, lo2, hi2)
        return np.where(np.abs(x) < w, 0.0, x)
    return f


def tune_rule_b(val, c):
    """Rule (b) full grid (as review/retrain/analysis.py) on validation terminal VaR95."""
    best = None
    for aa, bb, ww in itertools.product(A.A_GRID, A.A_GRID, A.W_GRID):
        v = float(A.var_cvar(A.net(A.rollout(val, A.rule_decider("rule", aa, bb, ww), True), c))[0])
        if best is None or v < best[0] - 1e-12:
            best = (v, float(aa), float(bb), float(ww))
    return dict(a=best[1], b=best[2], w=best[3], VaR95=best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pricing_backend", default="table")
    ap.add_argument("--costs", default=",".join(str(c) for c in COSTS))
    ap.add_argument("--retune_rule_b", action="store_true", help="re-tune rule (b) on this market first")
    args = ap.parse_args()
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", args.pricing_backend,
                           "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
    val = A.T.get_market(a, a.val_seed, a.n_val)
    costs = [float(x) for x in args.costs.split(",")]
    base = {f"{c}_0": {"rule_b_VaR95": tune_rule_b(val, c) if args.retune_rule_b
                       else json.load(open("review/results/retrain/baselines_B.json"))[f"{c}_0"]["rule_b_VaR95"]}
            for c in costs}
    grids = [list(A.A_GRID), list(A.A_GRID), list(A.W_GRID)] * 2
    out = {}
    for c in costs:
        rb = base[f"{c}_0"]["rule_b_VaR95"]
        cache = {}

        def score(p):
            p = tuple(round(float(x), 4) for x in p)
            if p not in cache:
                r = A.rollout(val, decider(p), True)
                cache[p] = float(A.var_cvar(A.net(r, c))[0])
            return cache[p]
        rng = np.random.default_rng(0)

        def descend(cur):
            best = score(cur)
            improved = True
            while improved and len(cache) < BUDGET:
                improved = False
                for i, g in enumerate(grids):
                    for v in g:
                        if len(cache) >= BUDGET:
                            break
                        cand = list(cur)
                        cand[i] = v
                        s = score(cand)
                        if s < best - 1e-9:
                            best, cur, improved = s, cand, True
            return best, cur
        start = score([rb["a"], rb["b"], rb["w"]] * 2)
        best, cur = descend([rb["a"], rb["b"], rb["w"]] * 2)
        # use the rest of the budget: coordinate descent from random grid points (restarts)
        while len(cache) < BUDGET:
            b2, c2 = descend([float(rng.choice(g)) for g in grids])
            if b2 < best - 1e-9:
                best, cur = b2, c2
        out[str(c)] = dict(rule_b=base[f"{c}_0"]["rule_b_VaR95"], early=dict(a=cur[0], b=cur[1], w=cur[2]), late=dict(a=cur[3], b=cur[4], w=cur[5]),
                           val_VaR95=best, rule_b_val_VaR95=start, evaluations=len(cache))
        print(c, out[str(c)], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"richer_rule_params_{args.pricing_backend}_{args.costs}.json", "w"), indent=2,
              default=float)


if __name__ == "__main__":
    main()
