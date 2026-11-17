"""Market arrays for the hedging environment, with an on-disk cache.

The arrays are the same ones the original ``MainPortfolio`` builds (same calls, same random
stream), stored as plain numpy arrays so that the PyTorch trainer, the vectorised environment
and the standalone numpy evaluator can use them without the ``Option`` objects.

Keys (P paths, T1 = time_to_simulate + 1 dates):
    S, V                     (P, T1)      spot and volatility
    l_price, l_delta,
    l_gamma, l_vega          (P, T1)      liability portfolio, already x num_contract x contract_size
    l_next                   (P, T1-1)    liability value at t+1 with the positions of t
    h_price, h_delta,
    h_gamma, h_vega          (P, T1, T1)  hedge option issued at date j, seen at date t: [p, t, j],
                                          per contract (x contract_size)
    h_inactive               (P, T1, T1)  True if option j is not alive at date t
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

KEYS = ["S", "V", "l_price", "l_next", "l_delta", "l_gamma", "l_vega",
        "h_price", "h_delta", "h_gamma", "h_vega", "h_inactive"]
CACHE_VERSION = 1


def build_market(utils):
    """Build the market arrays in the same order as ``MainPortfolio.__init__``."""
    np.random.seed(utils.seed)  # TradingEnv.__init__ seeds before MainPortfolio is built
    a_price, vol = utils.init_env()
    liab = utils.agg_poisson_dist(a_price, vol)
    hed = utils.atm_hedges(a_price, vol)  # (P, n_hedge) Option objects, views into 3-d arrays

    L = np.array([o.num_contract * o.contract_size for o in liab], dtype=float)
    if not np.allclose(L, L[0]):
        raise ValueError("liability options with different sizes are not supported")
    L = L[0]
    cs = utils.contract_size

    def hedge_array(attr):
        base = getattr(hed[0, 0], attr).base  # (P, time, issue) array created in atm_hedges
        if base is not None and base.shape == (hed.shape[0], hed.shape[1], hed.shape[1]):
            return base
        return np.stack([[getattr(o, attr) for o in row] for row in hed]).transpose(0, 2, 1)

    m = dict(
        S=np.asarray(a_price, dtype=float),
        V=np.broadcast_to(np.asarray(vol, dtype=float), a_price.shape).copy(),
        l_price=L * np.stack([o.price_path for o in liab]),
        l_next=L * np.stack([o.next_price_path for o in liab]),
        l_delta=L * np.stack([o.delta_path for o in liab]),
        l_gamma=L * np.stack([o.gamma_path for o in liab]),
        l_vega=L * np.stack([o.vega_path for o in liab]),
        h_price=hedge_array("price_path") * cs,
        h_delta=hedge_array("delta_path") * cs,
        h_gamma=hedge_array("gamma_path") * cs,
        h_vega=hedge_array("vega_path") * cs,
        h_inactive=hedge_array("inactive").copy(),
    )
    # the liability is a single synthetic option; the original env ignores its inactive flag
    # only when it is always active, so check it here
    if any(np.any(o.inactive[1:]) for o in liab):
        raise ValueError("liability becomes inactive inside the episode; not supported by the vectorised env")
    del hed, liab
    return m


def market_key(utils):
    params = utils.get_params()
    params.pop("n_jobs", None)  # does not change the numbers
    params["_cache_version"] = CACHE_VERSION
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_or_build_market(utils, cache_dir="markets", mmap=True, verbose=True):
    """Return the market arrays, building them once per parameter set.

    Arrays are stored as .npy files and opened with ``mmap_mode='r'``, so parallel runs on the
    same machine share one copy in memory.
    """
    folder = Path(cache_dir) / f"{utils.stochastic_process}_{utils.num_sim}_seed{utils.seed}_{market_key(utils)}"
    done = folder / "meta.json"
    if not done.exists():
        t0 = time.time()
        m = build_market(utils)
        folder.mkdir(parents=True, exist_ok=True)
        for k in KEYS:
            np.save(folder / f"{k}.npy", m[k])
        meta = dict(params=utils.get_params(), build_seconds=time.time() - t0, cache_version=CACHE_VERSION)
        json.dump(meta, open(done, "w"), indent=2, default=str)
        if verbose:
            print(f"market built in {meta['build_seconds']:.1f}s -> {folder}")
        if not mmap:
            return m
    elif verbose:
        print(f"market loaded from {folder}")
    return {k: np.load(folder / f"{k}.npy", mmap_mode="r" if mmap else None) for k in KEYS}


def save_market_npz(m, path, float32=False):
    """Single-file market for the standalone numpy evaluator."""
    out = {}
    for k in KEYS:
        a = np.asarray(m[k])
        out[k] = a.astype(np.float32) if (float32 and a.dtype == np.float64) else a
    np.savez_compressed(path, **out)
