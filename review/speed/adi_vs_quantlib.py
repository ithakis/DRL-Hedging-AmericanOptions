"""Single-sweep ADI table vs the current QuantLib table, and convergence of both.  Run on the Mac:

    PYTHONPATH=. python review/speed/adi_vs_quantlib.py --table_dir ../DRL-Hedging-retrain/pricing_tables

1. Build ADI tables on three grids (coarse / medium / fine) for the parameters of the main Heston setup
   and time them (vs 364,500 QuantLib solves, about 14 min).
2. Reference = the fine ADI grid.  Errors at all table nodes (price, delta, gamma, vega), by region,
   for the QuantLib table and the coarser ADI tables.
3. Independent check: direct QuantLib solves with a fine engine (N_gridpoints 200) at random nodes.
Writes review/results/speed/adi_vs_quantlib.md (+ .json) and saves the medium ADI table next to the
QuantLib one as heston_american_put_adi_<key>_<grid>.npz (the training code does NOT use it).
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from environment.heston_adi import ADIGrid, build_table_adi
from environment.heston_table import HestonAmericanPutTable, HestonTableConfig, heston_point

GRIDS = {
    "coarse": ADIGrid(ns=100, nv=50, steps_per_day=2),
    "medium": ADIGrid(ns=200, nv=100, steps_per_day=4),
    "fine": ADIGrid(ns=400, nv=200, steps_per_day=8),
}
NAMES = ["price", "delta", "gamma", "vega"]


def regions(log_m, taus, vols):
    """Boolean masks (n_tau, n_m, n_vol) for the table nodes."""
    T, M, V = np.meshgrid(taus, np.arange(log_m.shape[1]), vols, indexing="ij")
    X = log_m[:, :, None] * np.ones_like(V)
    m = np.exp(X)
    on_paths = (V >= 0.09) & (V <= 0.6)
    return {
        "all": np.ones_like(m, bool),
        "bulk (m 0.75-1.3, vol 0.09-0.6)": (m >= 0.75) & (m <= 1.3) & on_paths,
        "near maturity (1-3 d, m 0.9-1.1)": (T <= 3) & (m >= 0.9) & (m <= 1.1) & on_paths,
        "deep ITM (m < 0.75)": (m < 0.75) & on_paths,
    }


def err_table(vals, ref, masks):
    rows = []
    for name, mk in masks.items():
        e = np.abs(vals - ref)  # (n_tau, 4, n_m, n_vol)
        row = [name]
        for q in range(4):
            eq = e[:, q][mk]
            row.append(f"{eq.max():.1e} / {eq.mean():.1e}")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_dir", default="../DRL-Hedging-retrain/pricing_tables")
    ap.add_argument("--n_check", type=int, default=150, help="random nodes for the fine QuantLib check")
    ap.add_argument("--out", default="review/results/speed")
    a = ap.parse_args()

    # the main Heston setup (same parameters the training uses for its table)
    from review.retrain.settings import train_args
    import train_d4pg as T
    args = T.get_args(train_args("B", 0.02, 1, out="/tmp/_adi"))
    u = T.make_utils(argparse.Namespace(**{**vars(args), "spread": 0.0}), args.market_seed, 8)
    cfg = HestonTableConfig.from_utils(u)
    ql = HestonAmericanPutTable.load_or_build(cfg, table_dir=a.table_dir)
    print("config:", cfg)

    tabs, secs = {}, {}
    for name, g in GRIDS.items():
        t0 = time.time()
        tabs[name] = build_table_adi(cfg, g)
        secs[name] = time.time() - t0
    ref = tabs["fine"]["values"]
    assert np.allclose(tabs["fine"]["log_m"], ql.log_m) and np.allclose(tabs["fine"]["vols"], ql.vols)
    masks = regions(ql.log_m, ql.taus, ql.vols)

    lines = ["# Single-sweep ADI table vs QuantLib table", "",
             f"Parameters: {json.dumps(dataclasses.asdict(cfg))}", "",
             "## Build time", "", "| Table | Seconds |", "|---|---|",
             "| QuantLib (364,500 single-point solves, 4 cores) | about 840 (14 min, logged when built) |"]
    for name, g in GRIDS.items():
        lines.append(f"| ADI {name} ({g.ns} x {g.nv}, {g.steps_per_day} steps/day, 3 sweeps, 1 core) | {secs[name]:.1f} |")
    lines += ["", "## Error vs the fine ADI grid (max / mean abs, per unit strike, at all table nodes)", ""]
    for name, vals in (("QuantLib table", ql.values), ("ADI coarse", tabs["coarse"]["values"]),
                       ("ADI medium", tabs["medium"]["values"])):
        lines += [f"### {name}", "", "| Region | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|"]
        lines += ["| " + " | ".join(r) + " |" for r in err_table(vals, ref, masks)]
        lines.append("")

    # independent check: a fine direct QuantLib solve at random nodes
    rng = np.random.default_rng(0)
    fine_cfg = dataclasses.replace(cfg, N_gridpoints=200)
    rows = []
    t0 = time.time()
    for _ in range(a.n_check):
        k = int(rng.integers(0, len(ql.taus)))
        i = int(rng.integers(10, ql.log_m.shape[1] - 10))
        j = int(rng.integers(3, len(ql.vols) - 3))
        tau, x, vol = int(ql.taus[k]), ql.log_m[k, i], ql.vols[j]
        refp = np.array(heston_point(float(np.exp(x)), 1.0, tau, float(vol), fine_cfg))
        rows.append(np.abs(np.stack([ql.values[k, :, i, j], tabs["medium"]["values"][k, :, i, j],
                                     ref[k, :, i, j]]) - refp[None, :]))
    rows = np.array(rows)  # (n, 3, 4)
    lines += [f"## Error vs direct QuantLib solves with a 200-point engine ({a.n_check} random nodes, "
              f"{time.time() - t0:.0f}s)", "", "| Table | " + " | ".join(f"{n} max / mean" for n in NAMES) + " |",
              "|---|---|---|---|---|"]
    for t, name in enumerate(["QuantLib table (50-point engine)", "ADI medium", "ADI fine"]):
        lines.append(f"| {name} | " + " | ".join(f"{rows[:, t, q].max():.1e} / {rows[:, t, q].mean():.1e}"
                                                  for q in range(4)) + " |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adi_vs_quantlib.md").write_text("\n".join(lines) + "\n")
    json.dump({"seconds": secs, "config": dataclasses.asdict(cfg)}, open(out / "adi_vs_quantlib.json", "w"), indent=2)
    g = GRIDS["medium"]
    np.savez_compressed(Path(a.table_dir) / f"heston_american_put_adi_{cfg.key()}_{g.ns}x{g.nv}x{g.steps_per_day}.npz",
                        **tabs["medium"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
