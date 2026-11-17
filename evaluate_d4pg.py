#!/usr/bin/env python
"""Evaluate a trained policy (policy_best.npz) on a market built from any parameters.

Takes the same market flags as train_d4pg.py, so it can be used for the robustness table
(e.g. --volvol 0.4) and to export a market file for tools/evaluate_policy_numpy.py.

  python evaluate_d4pg.py --policy logs_torch/<run>/policy_best.npz --process Heston --volvol 0.4 \
      --test_seed 654 --n_test 8192 [--export_market market_volvol04.npz]

A new Heston parameter set needs its own price table: it is built once on first use
(about 15 min on 4 cores) and cached in pricing_tables/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import evaluate_policy_numpy as npe  # noqa: E402
import train_d4pg  # noqa: E402
from environment.market import save_market_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--export_market", default=None)
    ap.add_argument("--out_json", default=None)
    own, rest = ap.parse_known_args()
    pol = npe.NumpyPolicy(own.policy)
    defaults = ["--spread", str(pol.meta["spread"]), "--vega_obs", str(pol.meta["vega_obs"]),
                "--process", pol.meta.get("process", "Heston")]
    a = train_d4pg.get_args(defaults + rest)
    m = train_d4pg.get_market(a, a.test_seed, a.n_test)
    pnl, cost = npe.run_episodes(pol, m, a.spread, a.vega_obs)
    res = {k: float(v) for k, v in npe.summarize(pnl, cost, a.n_boot).items()}
    res.update(policy=own.policy, test_seed=a.test_seed, spread=a.spread,
               market={k: getattr(a, k) for k in ["process", "S0", "init_vol", "kappa", "theta", "volvol", "rho",
                                                    "r", "mu", "lambda_v"]})
    print(json.dumps(res, indent=2))
    if own.out_json:
        json.dump(res, open(own.out_json, "w"), indent=2)
    if own.export_market:
        save_market_npz(m, own.export_market)


if __name__ == "__main__":
    main()
