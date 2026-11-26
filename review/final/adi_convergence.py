"""Sven item 3: convergence of the pricing table actually used for all final numbers (adi_trading: single-sweep ADI,
modified Craig-Sneyd + Ikonen-Toivanen, maturity tau/252) for price, Delta, Gamma AND Vega.

The same 400 random off-grid points (S/K, trading days to maturity, vol; stratified as review/speed/table_convergence.py)
are evaluated on ADI tables of increasing resolution; the reference is a 2x finer table (800 x 400, 16 steps/day).
Errors per unit strike: max, 99th percentile and mean of |error|.
    PYTHONPATH=. python review/final/adi_convergence.py"""
import argparse
import time
from pathlib import Path

import numpy as np

from environment.heston_adi import ADIGrid, build_table_adi
from environment.heston_table import HestonAmericanPutTable, HestonTableConfig
from review.speed.table_convergence import sample_points

NAMES = ["Price", "Delta", "Gamma", "Vega"]
GRIDS = [("100 x 50, 2 steps/day", ADIGrid(ns=100, nv=50, steps_per_day=2)),
         ("200 x 100, 4 steps/day", ADIGrid(ns=200, nv=100, steps_per_day=4)),
         ("400 x 200, 8 steps/day (used)", ADIGrid(ns=400, nv=200, steps_per_day=8))]
REF = ("800 x 400, 16 steps/day (reference)", ADIGrid(ns=800, nv=400, steps_per_day=16))


def main():
    from review.retrain.settings import train_args
    import train_d4pg as T
    args = T.get_args(train_args("B", 0.02, 1, out="/tmp/_conv"))
    u = T.make_utils(argparse.Namespace(**{**vars(args), "spread": 0.0}), args.market_seed, 8)
    cfg = HestonTableConfig.from_utils(u)
    pts = sample_points(400, cfg.max_ttm, np.random.default_rng(2026))
    S = np.array([p[1] for p in pts]); tau = np.array([p[2] for p in pts]); vol = np.array([p[3] for p in pts])
    reg = np.array([p[0] for p in pts])
    vals, secs = {}, {}
    for name, g in GRIDS + [REF]:
        t0 = time.time()
        tab = HestonAmericanPutTable(build_table_adi(cfg, g, verbose=False, quantlib_maturity=False))
        secs[name] = time.time() - t0
        vals[name] = np.array(tab.evaluate(S, np.ones_like(S), tau, vol)).T  # (n, 4)
        print(name, f"{secs[name]:.0f}s", flush=True)
    ref = vals[REF[0]]
    scale = np.abs(ref).max(0)
    lines = ["# Convergence of the adi_trading pricing table (price, Delta, Gamma, Vega)", "",
             f"{len(pts)} random off-grid points ({len(pts) // 4} per region; S/K, trading days to maturity 1-{cfg.max_ttm}, "
             f"vol 0.12-0.6), maturity tau/252. Reference: ADI table on a 2x finer grid ({REF[0].split(' (')[0]}). "
             "Errors per unit strike, |error| max / 99th percentile / mean. Largest |value| in the sample: "
             + ", ".join(f"{n} {s:.3g}" for n, s in zip(NAMES, scale)) + ". Build time (one core): "
             + ", ".join(f"{k.split(' (')[0]} {v:.0f}s" for k, v in secs.items()) + ".", ""]
    for region in ["all", "bulk", "near maturity", "deep ITM", "deep OTM"]:
        mk = np.ones(len(pts), bool) if region == "all" else reg == region
        lines += [f"## {region} ({mk.sum()} points)", "", "| Grid | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|"]
        for name, _ in GRIDS:
            e = np.abs(vals[name] - ref)[mk]
            lines.append(f"| {name} | " + " | ".join(
                f"{e[:, q].max():.1e} / {np.quantile(e[:, q], 0.99):.1e} / {e[:, q].mean():.1e}" for q in range(4)) + " |")
        lines.append("")
    out = Path("review/results/final")
    out.mkdir(parents=True, exist_ok=True)
    (out / "adi_convergence.md").write_text("\n".join(lines) + "\n")
    np.savez_compressed(out / "adi_convergence.npz", S=S, tau=tau, vol=vol, **{k.split(" (")[0].replace(" ", ""): v for k, v in vals.items()})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
