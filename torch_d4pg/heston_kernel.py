"""Expected (kernel) critic target for the Heston hedging task (idea 16; swing H1).

The sampled TD target uses ONE next step of the simulator. Here the next step is replaced by a small
Gauss-Hermite quadrature over (log S', v') given (S_t, v_t) under the P-dynamics, and every quantity of
the next step (liability and hedge-option values and Greeks) is recomputed with the same Heston pricing
table that built the market. The critic target is the mixture over nodes,
    Z_target(s, a) = sum_j w_j [ r_j(s, a) + Z(s'_j, pi(s'_j)) ]   (1-step, discount 1),
and since the quantile loss is linear in the target distribution, the loss is the w_j-weighted sum of the
per-node quantile-Huber losses (exact).

The liability book is not stored in the market cache (only its aggregates), so `book_arrivals`
replays the random draws of Utils.agg_poisson_dist (same seed, same order) without pricing, and
`check_book` verifies the reconstruction against the cached aggregates.
New liability options that arrive at t+1 are ATM at the node price (strike = S'_j, as in the simulator,
moneyness_std = 0); their number and signs are the ones realised on the path.
"""
import numpy as np

from environment.utils import Utils


# ---------------------------------------------------------------------------------------- book
def book_arrivals(utils: Utils):
    """Replays the market's random stream. Returns S, vol (P, T1) and the net signed number of
    liability options that arrive at each step c (P, T1) (step 0: the initial option, K = utils.K,
    ttm = init_ttm; later: ATM, ttm = ttms[0])."""
    assert len(utils.ttms) == 1 and utils.moneyness_std == 0.0, "only one liability ttm, ATM arrivals"
    np.random.seed(utils.seed)
    S, vol = utils.init_env()
    P, T1 = S.shape
    n_per_day = np.random.poisson(utils.poisson_rate, S.shape)
    max_n = n_per_day.max(axis=0)
    c = np.zeros((P, T1))
    for i in range(T1):
        k = 1 if i == 0 else int(max_n[i])
        np.random.choice(utils.ttms, (P, k))
        np.random.normal(utils.moneyness_mean, utils.moneyness_std, (P, k))
        if i == 0:
            c[:, 0] = 1.0
        else:
            sign = np.random.choice([-1.0, 1.0], (P, k))
            exist = n_per_day[:, i][:, None] > np.arange(k)
            c[:, i] = np.sum(sign * exist, axis=1)
    vol = np.broadcast_to(np.asarray(vol, dtype=float), S.shape)
    return np.asarray(S, float), np.array(vol), c


class Book:
    """Liability (Poisson arrivals) and hedge options of one market, priced with the Heston table."""

    def __init__(self, utils: Utils, market, c):
        self.u, self.m, self.c = utils, market, c
        self.table = utils.heston_table()
        L = market["l_price"]
        self.P, self.T1 = market["S"].shape
        self.cs = utils.contract_size
        # liability scale L (num_contract * contract_size of the synthetic option)
        self.L = None
        self.S_arr = np.asarray(market["S"])  # strike of an option that arrives at s (ATM) = S_s
        self.liab_ttm, self.init_ttm, self.K0, self.hed_ttm = utils.ttms[0], utils.init_ttm, utils.K, utils.hed_ttm

    def liab(self, p, t_eval, S, vol, upto):
        """Liability Greeks at time t_eval for state (S, vol), options that arrived at steps <= upto.
        p (B,), S/vol (B, M). Returns price, delta, gamma, vega (B, M), unscaled (per unit of c)."""
        B, M = S.shape
        s = np.arange(self.T1)
        t_eval = np.broadcast_to(np.asarray(t_eval), (B,))
        upto = np.broadcast_to(np.asarray(upto), (B,))
        K = np.where(s[None, :] == 0, self.K0, self.S_arr[p])                       # (B, T1)
        ttm = np.where(s == 0, self.init_ttm, self.liab_ttm)[None, :] - (t_eval[:, None] - s[None, :])  # (B, T1)
        c = self.c[p] * (s[None, :] <= upto[:, None]) * (ttm >= 0)                # (B, T1)
        keep = np.flatnonzero(np.abs(c).sum(0) > 0)
        out = [np.zeros((B, M)) for _ in range(4)]
        if len(keep) == 0:
            return out
        Kk, tk, ck = K[:, keep], np.maximum(ttm[:, keep], 0), c[:, keep]
        vals = self.table.evaluate(S[:, :, None], Kk[:, None, :], np.broadcast_to(tk[:, None, :], (B, M, len(keep))),
                                   vol[:, :, None])
        return [np.sum(v * ck[:, None, :], axis=2) for v in vals]

    def hedges(self, p, t_eval, S, vol, n, upto):
        """Hedge-option book (contracts n (B, T1) of the option issued at s) at t_eval: price/delta/gamma/vega (B, M)."""
        B, M = S.shape
        s = np.arange(self.T1)
        t_eval = np.broadcast_to(np.asarray(t_eval), (B,))
        upto = np.broadcast_to(np.asarray(upto), (B,))
        ttm = self.hed_ttm - (t_eval[:, None] - s[None, :])                         # (B, T1)
        w = n * (s[None, :] <= upto[:, None]) * (ttm >= 0)
        keep = np.flatnonzero(np.abs(w).sum(0) > 0)
        out = [np.zeros((B, M)) for _ in range(4)]
        if len(keep) == 0:
            return out
        Kk = self.S_arr[p][:, keep]
        tk = np.maximum(ttm[:, keep], 0)
        vals = self.table.evaluate(S[:, :, None], Kk[:, None, :], np.broadcast_to(tk[:, None, :], (B, M, len(keep))),
                                   vol[:, :, None])
        return [self.cs * np.sum(v * w[:, keep][:, None, :], axis=2) for v in vals]


def check_book(utils, market, c, n_paths=64, seed=0):
    """Max relative error of the reconstructed liability price / gamma / vega vs the cached market."""
    bk = Book(utils, market, c)
    rng = np.random.default_rng(seed)
    p = rng.choice(bk.P, n_paths, replace=False)
    errs = {}
    S, V = np.asarray(market["S"]), np.asarray(market["V"])
    L = None
    for t in range(bk.T1 - 1):
        pr, de, ga, ve = bk.liab(p, t, S[p, t][:, None], V[p, t][:, None], np.full(len(p), t))
        lp = np.asarray(market["l_price"])[p, t]
        if L is None:
            L = float(np.median(lp / pr[:, 0]))
        for name, mine, ref in (("price", pr, lp), ("gamma", ga, np.asarray(market["l_gamma"])[p, t]),
                                ("vega", ve, np.asarray(market["l_vega"])[p, t]), ("delta", de, np.asarray(market["l_delta"])[p, t])):
            e = np.max(np.abs(L * mine[:, 0] - ref)) / (np.max(np.abs(ref)) + 1e-12)
            errs[name] = max(errs.get(name, 0.0), float(e))
        nx = bk.liab(p, t + 1, S[p, t + 1][:, None], V[p, t + 1][:, None], np.full(len(p), t))[0]
        ln = np.asarray(market["l_next"])[p, t]
        errs["next"] = max(errs.get("next", 0.0), float(np.max(np.abs(L * nx[:, 0] - ln)) / np.max(np.abs(ln))))
    return L, errs


# ---------------------------------------------------------------------------------------- quadrature
def gh_nodes(m):
    """Gauss-Hermite nodes/weights for a standard normal (probabilists'), m points."""
    x, w = np.polynomial.hermite_e.hermegauss(m)
    return x, w / w.sum()


class HestonKernel:
    """One-step quadrature over (Z1, Z2) for the Heston P-dynamics (Euler on log S and v, v floored),
    then recompute reward and next state of the hedging env at each node.

    transition(p, t, n, cost) -> rewards (B, M), next_obs (B, M, d), done (B,), weights (M,)
      p: path index, t: step, n: contracts held of each issued hedge (B, T1) AFTER the trade at t,
      cost: option cost of the trade at t (B,) (<= 0).
    With nodes=None the realised next step of the path is used (M = 1): this reproduces the env exactly
    (test of the bookkeeping).
    """

    def __init__(self, utils, market, c, vega_obs=True, m_x=3, mu=None, v_floor=1e-4):
        self.bk = Book(utils, market, c)
        self.m = {k: np.asarray(market[k]) for k in ("S", "V", "l_price", "l_delta", "h_price", "h_delta")}
        self.L = -100.0 if utils.num_conts_to_add == -1 else None
        self.L = float(np.median(np.asarray(market["l_price"])[:64, 0] / self.bk.liab(
            np.arange(64), 0, np.asarray(market["S"])[:64, :1], np.asarray(market["V"])[:64, :1], np.zeros(64, int))[0][:, 0]))
        self.vega_obs = vega_obs
        self.T = self.bk.T1 - 1
        self.dt = 1.0 / utils.TradingDaysPerYear
        self.mu = utils.mu if mu is None else mu
        self.kappa, self.theta, self.xi, self.rho = utils.kappa, utils.theta, utils.volvol, utils.rho
        self.v_floor = v_floor
        z, w = gh_nodes(m_x)
        Z1, Z2 = np.meshgrid(z, z, indexing="ij")
        self.z1, self.z2 = Z1.ravel(), Z2.ravel()
        self.w = np.outer(w, w).ravel()

    def nodes(self, S, vol):
        v = vol[:, None] ** 2
        dt, sq = self.dt, np.sqrt(vol[:, None] ** 2 * self.dt)
        S1 = S[:, None] * np.exp((self.mu - 0.5 * v) * dt + sq * self.z1[None, :])
        v1 = v + self.kappa * (self.theta - v) * dt + self.xi * sq * (self.rho * self.z1 + np.sqrt(1 - self.rho ** 2) * self.z2)[None, :]
        return S1, np.sqrt(np.maximum(v1, self.v_floor))

    def transition(self, p, t, n, cost, realised=False):
        m, bk, B = self.m, self.bk, len(p)
        S, vol = m["S"][p, t], m["V"][p, t]
        if realised:
            S1, vol1, w = m["S"][p, t + 1][:, None], m["V"][p, t + 1][:, None], np.ones(1)
        else:
            (S1, vol1), w = self.nodes(S, vol), self.w
        t1 = t + 1
        # P&L over (t, t+1]: book held at t, valued at t+1 (arrivals at t+1 excluded, as l_next)
        lp1 = self.L * bk.liab(p, t1, S1, vol1, t)[0]
        hp1 = bk.hedges(p, t1, S1, vol1, n, t)[0]
        h_now = np.sum(n * m["h_price"][p, t, :] * (np.arange(bk.T1)[None, :] <= t[:, None]), axis=1)
        pos = -(m["l_delta"][p, t] + np.sum(n * m["h_delta"][p, t, :] * (np.arange(bk.T1)[None, :] <= t[:, None]), axis=1))
        r = cost[:, None] + lp1 - m["l_price"][p, t][:, None] + hp1 - h_now[:, None] + pos[:, None] * (S1 - S[:, None])
        # next state: book incl. arrivals at t+1, hedges held, and the new ATM hedge of t+1
        _, _, lg, lv = bk.liab(p, t1, S1, vol1, t1)
        _, _, hg, hv = bk.hedges(p, t1, S1, vol1, n, t)
        _, _, gh, vh = bk.table.evaluate(S1, S1, np.full(S1.shape, bk.hed_ttm), vol1)
        cols = [S1, self.L * lg + hg, bk.cs * gh]
        if self.vega_obs:
            cols += [self.L * lv + hv, bk.cs * vh]
        obs1 = np.stack(cols, axis=-1)
        done = t1 == self.T
        obs1[done, :, 1:] = 0.0
        return r, obs1, done, w
