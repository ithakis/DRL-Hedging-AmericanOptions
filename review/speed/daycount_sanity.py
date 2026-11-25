"""Day-count sanity check (no training): ADI table with maturity tau/252 (adi_trading, the paths' clock) vs
the same ADI table with QuantLib's calendar-day maturity (adi_qldays) vs the current QuantLib table, at ATM points.

    PYTHONPATH=. python review/speed/daycount_sanity.py --table_dir ../DRL-Hedging-retrain/pricing_tables
"""
import argparse

import numpy as np

from environment.heston_table import HestonAmericanPutTable, HestonTableConfig

ap = argparse.ArgumentParser()
ap.add_argument("--table_dir", default="../DRL-Hedging-retrain/pricing_tables")
a = ap.parse_args()
from review.retrain.settings import train_args
import train_d4pg as T
args = T.get_args(train_args("B", 0.02, 1, out="/tmp/_dc"))
u = T.make_utils(argparse.Namespace(**{**vars(args), "spread": 0.0}), args.market_seed, 8)
cfg = HestonTableConfig.from_utils(u)
tabs = {"QuantLib table": HestonAmericanPutTable.load_or_build(cfg, table_dir=a.table_dir),
        "adi_qldays": HestonAmericanPutTable.load_or_build_adi(cfg, table_dir=a.table_dir, quantlib_maturity=True),
        "adi_trading": HestonAmericanPutTable.load_or_build_adi(cfg, table_dir=a.table_dir, quantlib_maturity=False)}
lines = ["| tau (days) | m = S/K | " + " | ".join(f"{k} price" for k in tabs) + " | "
         + " | ".join(f"{k} gamma" for k in tabs) + " |", "|---" * (2 + 2 * len(tabs)) + "|"]
for tau in (1, 5, 30):
    for m in (0.97, 1.0, 1.03):
        r = {k: t.evaluate(np.array([m]), np.array([1.0]), np.array([float(tau)]), np.array([0.3])) for k, t in tabs.items()}
        lines.append(f"| {tau} | {m} | " + " | ".join(f"{float(v[0][0]):.5f}" for v in r.values()) + " | "
                     + " | ".join(f"{float(v[2][0]):.3f}" for v in r.values()) + " |")
print("Per unit strike, vol 0.3. 365/252 = 1.448: adi_qldays prices tau trading days as round(tau*365/252) calendar days.\n")
print("\n".join(lines))
