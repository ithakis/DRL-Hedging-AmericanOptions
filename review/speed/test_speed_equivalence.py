"""Speed changes must not change the numbers.  Run on the Mac (all compute is local):

    PYTHONPATH=. python review/speed/test_speed_equivalence.py --markets ../DRL-Hedging-retrain/markets \
        --table_dir ../DRL-Hedging-retrain/pricing_tables

1. Table: vectorised bicubic lookup vs the SciPy per-tau loop, on random points (also outside the grid).
2. Density target: the new HestonKernel.transition (one table pass per book) vs the original code
   (copied below), on random paths / days / holdings.  atm_fix=False must agree to rounding.
3. Timing of both.
"""
import argparse
import time

import numpy as np

from environment.heston_table import HestonAmericanPutTable


def reference_transition(self, p, t, n, cost):
    """HestonKernel.transition before the speed change (commit 7644745), sampled nodes only."""
    m, bk = self.m, self.bk
    S, vol = m["S"][p, t], m["V"][p, t]
    S1, vol1, w = self.nodes(S, vol)
    t1 = t + 1
    lp1 = self.L * bk.liab(p, t1, S1, vol1, t)[0]
    hp1 = bk.hedges(p, t1, S1, vol1, n, t)[0]
    h_now = np.sum(n * m["h_price"][p, t, :] * (np.arange(bk.T1)[None, :] <= t[:, None]), axis=1)
    pos = -(m["l_delta"][p, t] + np.sum(n * m["h_delta"][p, t, :] * (np.arange(bk.T1)[None, :] <= t[:, None]), axis=1))
    r = cost[:, None] + lp1 - m["l_price"][p, t][:, None] + hp1 - h_now[:, None] + pos[:, None] * (S1 - S[:, None])
    _, _, lg, lv = bk.liab(p, t1, S1, vol1, t1)
    _, _, hg, hv = bk.hedges(p, t1, S1, vol1, n, t)
    _, _, gh, vh = bk.table.evaluate(S1, S1, np.full(S1.shape, bk.hed_ttm), vol1)
    cols = [S1, self.L * lg + hg, bk.cs * gh]
    if self.vega_obs:
        cols += [self.L * lv + hv, bk.cs * vh]
    if self.extra_obs:
        cols += [np.broadcast_to((1.0 - t1 / self.T)[:, None], S1.shape), vol1]
    from torch_d4pg.hedge_env import HedgingEnv, neutral_amounts
    if HedgingEnv.rule_obs:
        cols += list(neutral_amounts(cols[1], cols[2], cols[3], cols[4]))
    obs1 = np.stack(cols, axis=-1)
    done = t1 == self.T
    obs1[done, :, 1:] = 0.0
    return r, obs1, done, w


def timeit(f, reps):
    f()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="../DRL-Hedging-retrain/markets")
    ap.add_argument("--table_dir", default="../DRL-Hedging-retrain/pricing_tables")
    ap.add_argument("--n_points", type=int, default=200_000)
    ap.add_argument("--skip_kernel", action="store_true")
    a = ap.parse_args()

    # ---------------------------------------------------------------- 1. table
    from review.retrain.settings import train_args
    import train_d4pg as T
    args = T.get_args(train_args("B", 0.02, 1, out="/tmp/_speedtest") + [
        "--market_dir", a.markets, "--n_train_paths", "40000", "--extra_obs", "--kernel_mx", "3",
        "--table_dir", a.table_dir, "--n_envs", "16"])
    u = T.make_utils(argparse.Namespace(**{**vars(args), "spread": 0.0}), args.market_seed, args.n_train_paths)
    u.table_dir = a.table_dir
    tab = u.heston_table()
    rng = np.random.default_rng(0)
    N = a.n_points
    K = rng.uniform(5, 20, N)
    S = K * np.exp(rng.uniform(np.log(0.45), np.log(2.2), N))   # some points outside the moneyness grid
    ttm = rng.integers(-1, int(tab.taus.max()) + 1, N)          # includes 0 (payoff) and -1 (zero)
    vol = rng.uniform(0.0, 1.1, N)                               # some outside the vol grid
    HestonAmericanPutTable.fast = False
    ref = np.array(tab.evaluate(S, K, ttm, vol))
    t_slow = timeit(lambda: tab.evaluate(S[:20000], K[:20000], ttm[:20000], vol[:20000]), 3)
    HestonAmericanPutTable.fast = True
    new = np.array(tab.evaluate(S, K, ttm, vol))
    t_fast = timeit(lambda: tab.evaluate(S[:20000], K[:20000], ttm[:20000], vol[:20000]), 3)
    err = np.abs(new - ref).max(axis=1)
    scale = np.abs(ref).max(axis=1)
    print(f"[table] max abs diff price/delta/gamma/vega: {err} (max |value| {scale})")
    print(f"[table] 20k points: SciPy loop {1e3 * t_slow:.1f} ms, vectorised {1e3 * t_fast:.1f} ms "
          f"-> {t_slow / t_fast:.1f}x")
    assert np.all(err <= 1e-10 * np.maximum(scale, 1.0)), "fast table differs from SciPy"

    if a.skip_kernel:
        return
    # ---------------------------------------------------------------- 2. density target
    from torch_d4pg.heston_kernel import HestonKernel, book_arrivals
    train_m = T.get_market(args, args.market_seed, args.n_train_paths)
    _, _, c_book = book_arrivals(u)
    kern = HestonKernel(u, train_m, c_book, True, m_x=3, extra_obs=True)
    B = 16
    worst = np.zeros(2)
    t_old = t_new = 0.0
    for trial in range(60):
        p = rng.choice(kern.bk.P, B, replace=False)
        t = np.full(B, int(rng.integers(0, kern.T)))
        n = rng.normal(0, 1, (B, kern.bk.T1)) * (np.arange(kern.bk.T1)[None, :] <= t[:, None])
        cost = -np.abs(rng.normal(0, 0.1, B))
        for fast in (False, True):
            HestonAmericanPutTable.fast = fast
            t0 = time.perf_counter()
            r0, o0, d0, w0 = reference_transition(kern, p, t, n, cost)
            t_old += (time.perf_counter() - t0) if not fast else 0.0
        HestonAmericanPutTable.fast = True
        t0 = time.perf_counter()
        r1, o1, d1, w1 = kern.transition(p, t, n, cost)
        t_new += time.perf_counter() - t0
        assert np.array_equal(d0, d1) and np.allclose(w0, w1, rtol=0, atol=0)
        worst = np.maximum(worst, [np.abs(r1 - r0).max() / (np.abs(r0).max() + 1e-12),
                                   np.abs(o1 - o0).max() / (np.abs(o0).max() + 1e-12)])
    print(f"[kernel] max relative diff reward {worst[0]:.2e}, next state {worst[1]:.2e}")
    print(f"[kernel] per transition (16 paths x 9 nodes): original {1e3 * t_old / 60:.1f} ms, "
          f"new {1e3 * t_new / 60:.1f} ms -> {t_old / t_new:.1f}x")
    assert worst.max() < 1e-9, "new transition differs from the original"
    # atm_fix: how much the next-state Greeks move when new arrivals are ATM at the node
    kern.atm_fix = True
    p = rng.choice(kern.bk.P, 256, replace=False)
    diffs = []
    for tt in range(1, kern.T - 1):
        t = np.full(256, tt)
        n = np.zeros((256, kern.bk.T1))
        _, oA, _, wA = kern.transition(p, t, n, np.zeros(256))
        kern.atm_fix = False
        _, oB, _, _ = kern.transition(p, t, n, np.zeros(256))
        kern.atm_fix = True
        diffs.append(np.abs(oA - oB).mean(axis=(0, 1)))
    print("[atm_fix] mean |change| of next-state features (S, gamma, ATM gamma, vega, ATM vega, ...):",
          np.round(np.mean(diffs, axis=0), 4))
    print("all equivalence checks passed")


if __name__ == "__main__":
    main()
