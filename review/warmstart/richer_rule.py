"""Richer rule baseline (Sven: "is the gain from extra parameters?").

Rule (b) with blend weights and band that depend on time to horizon: separate (a, b, w) for the first
and the second half of the 30 days (6 parameters). Tuned on VALIDATION terminal VaR95 by coordinate
search over the same grids as rule (b) (a, b in 0..1 step 0.1, w in W_GRID), starting from the tuned
rule (b), with a budget of about 1,100 rollouts per cost (rule (b)'s full grid is 11 x 11 x 9 = 1,089).
Writes review/results/warmstart/richer_rule_params.json. No test paths are used.
"""
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


def main():
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets"])
    val = A.T.get_market(a, a.val_seed, a.n_val)
    base = json.load(open("review/results/retrain/baselines_B.json"))
    grids = [list(A.A_GRID), list(A.A_GRID), list(A.W_GRID)] * 2
    out = {}
    for c in COSTS:
        rb = base[f"{c}_0"]["rule_b_VaR95"]
        cache = {}

        def score(p):
            p = tuple(round(float(x), 4) for x in p)
            if p not in cache:
                r = A.rollout(val, decider(p), True)
                cache[p] = float(A.var_cvar(A.net(r, c))[0])
            return cache[p]
        cur = [rb["a"], rb["b"], rb["w"]] * 2
        best = score(cur)
        start = best
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
        out[str(c)] = dict(early=dict(a=cur[0], b=cur[1], w=cur[2]), late=dict(a=cur[3], b=cur[4], w=cur[5]),
                           val_VaR95=best, rule_b_val_VaR95=start, evaluations=len(cache))
        print(c, out[str(c)], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "richer_rule_params.json", "w"), indent=2)


if __name__ == "__main__":
    main()
