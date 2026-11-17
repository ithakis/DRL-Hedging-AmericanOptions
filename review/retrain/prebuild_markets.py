"""Build (or load) every market and pricing table the retrain needs, before any training starts.

Parallel training runs read the cached markets (memory-mapped); building them here first avoids
several runs building the same market at once. Uses all cores.

    PYTHONPATH=. python review/retrain/prebuild_markets.py --blocks A B --robustness
"""
import argparse
import time

import train_d4pg as T
from review.retrain.settings import BLOCKS, BUMPS, with_bump

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", nargs="+", default=["A", "B"])
ap.add_argument("--robustness", action="store_true", help="also the 10 bumped Heston test markets")
args = ap.parse_args()

t00 = time.time()
for blk in args.blocks:
    a = T.get_args(BLOCKS[blk]["flags"] + ["--obj_func", "var", "--spread", "0.0"])
    for seed, n in [(a.val_seed, a.n_val), (a.test_seed, a.n_test), (a.market_seed, a.n_train_paths)]:
        t0 = time.time()
        T.get_market(a, seed, n)
        print(f"[{blk}] seed {seed}, {n} paths: {time.time() - t0:.1f}s", flush=True)

if args.robustness:
    base = BLOCKS["B"]["flags"]
    for flag, v in BUMPS:
        a = T.get_args(with_bump(base, flag, v) + ["--obj_func", "var", "--spread", "0.0"])
        t0 = time.time()
        T.get_market(a, a.test_seed, a.n_test)
        print(f"[robustness {flag}={v}] test market: {time.time() - t0:.1f}s", flush=True)
print(f"all markets ready in {(time.time() - t00) / 60:.1f} min")
