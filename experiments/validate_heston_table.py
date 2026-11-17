"""Validate the Heston American put table against direct QuantLib solves.

Random points (not grid nodes), random strikes (tests the S/K homogeneity),
with extra points near maturity and deep in / out of the money.
Errors are for one option with the drawn strike, in natural units:
price and vega in currency, delta per share, gamma per currency unit.

Usage:
    python experiments/validate_heston_table.py --n_points 4000 --seed 7
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from environment.heston_table import HestonAmericanPutTable, HestonTableConfig, heston_point

QTY = ["price", "delta", "gamma", "vega"]


def sample_points(n, rng, cfg):
    n_bulk = n // 2
    n_short = n // 6
    n_itm = n // 6
    n_otm = n - n_bulk - n_short - n_itm
    tau = np.r_[rng.integers(1, cfg.max_ttm + 1, n_bulk), rng.integers(1, 4, n_short),
                rng.integers(1, cfg.max_ttm + 1, n_itm + n_otm)]
    log_m = np.r_[rng.uniform(np.log(0.75), np.log(1.3), n_bulk),
                  rng.uniform(np.log(0.9), np.log(1.1), n_short),
                  rng.uniform(np.log(0.55), np.log(0.75), n_itm),
                  rng.uniform(np.log(1.3), np.log(1.9), n_otm)]
    vol = rng.uniform(0.08, 0.6, n)
    K = rng.uniform(5.0, 20.0, n)
    region = np.array(["bulk"] * n_bulk + ["near maturity (1-3d)"] * n_short
                      + ["deep ITM (m<0.75)"] * n_itm + ["deep OTM (m>1.3)"] * n_otm)
    return pd.DataFrame(dict(tau=tau, m=np.exp(log_m), vol=vol, K=K, region=region))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_points", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--out", default="review/results/table_validation")
    args = ap.parse_args()

    cfg = HestonTableConfig()
    table = HestonAmericanPutTable.load_or_build(cfg, n_jobs=args.n_jobs)
    rng = np.random.default_rng(args.seed)
    df = sample_points(args.n_points, rng, cfg)

    ref = Parallel(n_jobs=args.n_jobs, batch_size=50)(
        delayed(heston_point)(float(r.m * r.K), float(r.K), int(r.tau), float(r.vol), cfg)
        for r in df.itertuples()
    )
    ref = np.array(ref, dtype=float)
    # the original environment vega (tiny forward bump), to show its FD-mesh noise
    cfg_fwd = dataclasses.replace(cfg, vega_mode="env_forward")
    fwd = Parallel(n_jobs=args.n_jobs, batch_size=50)(
        delayed(heston_point)(float(r.m * r.K), float(r.K), int(r.tau), float(r.vol), cfg_fwd)
        for r in df.itertuples()
    )
    df["vega_envfwd"] = np.array(fwd, dtype=float)[:, 3]
    tab = np.array(table.evaluate(df.m * df.K, df.K, df.tau, df.vol)).T
    for i, q in enumerate(QTY):
        df[f"{q}_ql"] = ref[:, i]
        df[f"{q}_tab"] = tab[:, i]
        df[f"{q}_err"] = tab[:, i] - ref[:, i]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "points.csv", index=False)

    rows = []
    for region, g in [("all", df)] + list(df.groupby("region")):
        for q in QTY:
            e = g[f"{q}_err"].abs()
            scale = g[f"{q}_ql"].abs().max()
            rows.append(dict(region=region, quantity=q, n=len(g), max_abs_err=e.max(), mean_abs_err=e.mean(),
                             max_abs_value=scale, max_err_over_max_value=e.max() / scale if scale > 0 else np.nan))
        e = (g["vega_tab"] - g["vega_envfwd"]).abs()
        rows.append(dict(region=region, quantity="vega vs original fwd-bump vega", n=len(g), max_abs_err=e.max(),
                         mean_abs_err=e.mean(), max_abs_value=g["vega_ql"].abs().max(),
                         max_err_over_max_value=e.max() / g["vega_ql"].abs().max()))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    json.dump(table.n_out_of_grid, open(out / "out_of_grid.json", "w"))


if __name__ == "__main__":
    main()
