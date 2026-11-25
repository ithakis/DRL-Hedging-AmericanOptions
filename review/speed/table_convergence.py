"""Convergence check of the Heston pricing table with the existing QuantLib solver (referee numerics point).

At random (not grid) points in the region the paths visit, compare against direct QuantLib solves
(same engine and vega bump as the table, Utils._American_Option_Heston via heston_point) on finer grids:
    table (50-point engine, interpolated)  vs  direct 50 / 100 / 200 / 400-point solves.
"direct 50 vs 400" = engine discretisation error; "table vs direct 50" = interpolation error;
"table vs 400" = total error.  The single-sweep ADI table (fine grid, QuantLib maturity convention) is
reported as a second, independent check.  Run on the Mac (parallel over points):

    PYTHONPATH=. python review/speed/table_convergence.py --table_dir ../DRL-Hedging-retrain/pricing_tables --n 400 --jobs 4
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from environment.heston_table import HestonAmericanPutTable, HestonTableConfig, heston_point

NAMES = ["price", "delta", "gamma", "vega"]
LEVELS = [50, 100, 200, 400]


def sample_points(n, max_ttm, rng):
    """Stratified: bulk, near maturity, deep ITM, deep OTM (as the earlier table validation)."""
    k = n // 4
    pts = []
    for region in ("bulk", "near maturity", "deep ITM", "deep OTM"):
        for _ in range(k):
            if region == "bulk":
                m, tau = rng.uniform(0.75, 1.3), int(rng.integers(1, max_ttm + 1))
            elif region == "near maturity":
                m, tau = rng.uniform(0.9, 1.1), int(rng.integers(1, 4))
            elif region == "deep ITM":
                m, tau = rng.uniform(0.55, 0.75), int(rng.integers(1, max_ttm + 1))
            else:
                m, tau = rng.uniform(1.3, 1.9), int(rng.integers(1, max_ttm + 1))
            pts.append((region, float(m), tau, float(rng.uniform(0.12, 0.6))))  # vol range the paths visit
    return pts


def solve(pt, cfg):
    region, m, tau, vol = pt
    out = []
    for N in LEVELS:
        c = dataclasses.replace(cfg, N_gridpoints=N)
        out.append(heston_point(m, 1.0, tau, vol, c))  # S = m, K = 1 (prices per unit strike)
    return np.array(out)  # (levels, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_dir", default="../DRL-Hedging-retrain/pricing_tables")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="review/results/speed")
    a = ap.parse_args()

    from review.retrain.settings import train_args
    import train_d4pg as T
    args = T.get_args(train_args("B", 0.02, 1, out="/tmp/_conv"))
    u = T.make_utils(argparse.Namespace(**{**vars(args), "spread": 0.0}), args.market_seed, 8)
    cfg = HestonTableConfig.from_utils(u)
    tab = HestonAmericanPutTable.load_or_build(cfg, table_dir=a.table_dir)
    rng = np.random.default_rng(2026)
    pts = sample_points(a.n, cfg.max_ttm, rng)

    t0 = time.time()
    ref = np.array(Parallel(n_jobs=a.jobs)(delayed(solve)(p, cfg) for p in pts))  # (n, levels, 4)
    secs = time.time() - t0
    S = np.array([p[1] for p in pts])
    tau = np.array([p[2] for p in pts])
    vol = np.array([p[3] for p in pts])
    tbl = np.array(tab.evaluate(S, np.ones_like(S), tau, vol)).T  # (n, 4)
    adi = None
    try:
        from environment.heston_adi import ADIGrid, build_table_adi
        adi_tab = HestonAmericanPutTable(build_table_adi(cfg, ADIGrid(ns=400, nv=200, steps_per_day=8), verbose=False))
        adi = np.array(adi_tab.evaluate(S, np.ones_like(S), tau, vol)).T
    except Exception as e:  # second check only
        print("ADI check skipped:", e)

    regions = sorted(set(p[0] for p in pts), key=["bulk", "near maturity", "deep ITM", "deep OTM"].index)
    reg = np.array([p[0] for p in pts])
    best = ref[:, -1]
    rows = [("Direct QuantLib, 50-point engine (the table's engine)", ref[:, 0]),
            ("Direct QuantLib, 100-point engine ('high' accuracy)", ref[:, 1]),
            ("Direct QuantLib, 200-point engine", ref[:, 2]),
            ("Table (50-point engine, bicubic interpolation): total error", tbl)]
    if adi is not None:
        rows.append(("Single-sweep ADI table, 400 x 200, 8 steps/day (independent check)", adi))
    scale = np.abs(best).max(axis=0)
    lines = ["# Pricing-table convergence check (QuantLib FdHestonVanillaEngine, Modified Craig-Sneyd)", "",
             f"{len(pts)} random off-grid points (S/K, days to maturity, vol), {len(pts) // 4} per region, vol in "
             f"[0.12, 0.6]. Reference: direct QuantLib solve with a 400-point engine (400 x 400 x 200, "
             f"same vega bump). Errors per unit strike, max / mean abs. Largest |value|: "
             + ", ".join(f"{n} {s:.3g}" for n, s in zip(NAMES, scale)) + f". Solves took {secs:.0f}s.", ""]
    for region in ["all"] + regions:
        mk = np.ones(len(pts), bool) if region == "all" else reg == region
        lines += [f"## {region} ({mk.sum()} points)", "", "| Against the 400-point solve | " + " | ".join(NAMES) + " |",
                  "|---|---|---|---|---|"]
        for name, v in rows:
            e = np.abs(v - best)[mk]
            lines.append(f"| {name} | " + " | ".join(f"{e[:, q].max():.1e} / {e[:, q].mean():.1e}" for q in range(4)) + " |")
        e_int = np.abs(tbl - ref[:, 0])[mk]
        lines.append("| Interpolation only (table vs direct 50-point) | " +
                     " | ".join(f"{e_int[:, q].max():.1e} / {e_int[:, q].mean():.1e}" for q in range(4)) + " |")
        lines.append("")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_convergence.md").write_text("\n".join(lines) + "\n")
    np.savez_compressed(out / "table_convergence.npz", S=S, tau=tau, vol=vol, ref=ref, table=tbl,
                        adi=adi if adi is not None else np.zeros(0), levels=np.array(LEVELS))
    json.dump({"config": dataclasses.asdict(cfg), "seconds": secs}, open(out / "table_convergence.json", "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
