"""Build (or load from the cache) an adi_trading Heston market for a given seed and path count.
    PYTHONPATH=. python review/improve/build_market.py SEED N"""
import sys
import time
from review.retrain import analysis as A

a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                       "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
t = time.time()
m = A.T.get_market(a, int(sys.argv[1]), int(sys.argv[2]))
print("paths", m["S"].shape, f"{time.time() - t:.0f}s")
