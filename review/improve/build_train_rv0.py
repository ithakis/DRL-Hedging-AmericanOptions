"""Build the training market (seed 1234, 40k paths, adi_trading) with per-path initial vol log-uniform on [lo, hi]."""
import sys, time
from review.retrain import analysis as A
a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                       "--table_dir", "../DRL-Hedging-retrain/pricing_tables", "--n_train_paths", "40000",
                       "--train_init_vol_range", sys.argv[1]])
t = time.time(); m = A.T.get_market(a, a.market_seed, 40000)
import numpy as np
print("paths", m["S"].shape, f"{time.time() - t:.0f}s", "V0 quantiles", np.quantile(m["V"][:, 0], [0, .25, .5, .75, 1]).round(3))
