"""End-to-end check: same paths, environment built with the original QuantLib pricer vs the table.

Compares per-path episode P&L of the Delta, Delta-Gamma and Delta-Vega baselines and the
market build time.  Usage: python experiments/compare_backends.py --n_paths 32
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.heston_rule_baselines import build_market, paper_utils, simulate, total_pnl

ap = argparse.ArgumentParser()
ap.add_argument("--n_paths", type=int, default=32)
ap.add_argument("--seed", type=int, default=2025)
ap.add_argument("--out", default="review/results/backend_comparison")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

markets, secs = {}, {}
for backend in ["table", "quantlib"]:
    t0 = time.time()
    markets[backend], _ = build_market(paper_utils(args.seed, args.n_paths, backend=backend))
    secs[backend] = time.time() - t0
rows = []
for mode in ["delta", "gamma", "vega"]:
    for spread in [0.0, 0.01]:
        p = {b: total_pnl(simulate(markets[b], 0, 0, 0, mode), spread, 0) for b in markets}
        d = p["table"] - p["quantlib"]
        rows.append(dict(strategy=mode, spread=spread, mean_pnl_quantlib=p["quantlib"].mean(),
                         mean_pnl_table=p["table"].mean(), std_pnl_quantlib=p["quantlib"].std(),
                         max_abs_path_diff=np.abs(d).max(), mean_abs_path_diff=np.abs(d).mean()))
df = pd.DataFrame(rows)
df.to_csv(out / "pnl_comparison.csv", index=False)
for k in ["l_price", "l_delta", "l_gamma", "l_vega"]:
    print(k, "max abs diff liability", np.abs(markets["table"][k] - markets["quantlib"][k]).max(),
          "max abs value", np.abs(markets["quantlib"][k]).max())
json.dump(dict(n_paths=args.n_paths, seconds=secs), open(out / "timing.json", "w"), indent=2)
print(secs)
print(df.to_string(index=False))
