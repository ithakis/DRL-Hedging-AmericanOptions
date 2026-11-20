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
    """Quadrature over the one-step law of the market's OWN simulator: the Andersen (2008) QE scheme
    with martingale correction (environment/utils.py get_sim_path_Heston, psiC = 1.5, gamma1 = gamma2 = 0.5).

      variance:  psi <= psiC: v' = a (b + Z1)^2      -> Gauss-Hermite in Z1 (m_x nodes)
                 psi >  psiC: v' = 0 w.p. p, else Exp -> one node at 0 (weight p) + (m_x - 1) Gauss-Laguerre nodes
      log-price: ln S' = ln S + (r + mu - q) dt + K0* + K1 v + K2 v' + sqrt(K3 v + K4 v') Z2  -> Gauss-Hermite in Z2
    So the nodes integrate exactly the law the training paths are drawn from (up to quadrature error);
    weights differ per transition in the psi > psiC regime, hence weights of shape (B, M).

    transition(p, t, n, cost) -> rewards (B, M), next_obs (B, M, d), done (B,), weights (B, M)
      p: path index, t: step, n: contracts held of each issued hedge (B, T1) AFTER the trade at t,
      cost: option cost of the trade at t (B,) (<= 0).
    realised=True uses the realised next step of the path (M = 1): reproduces the env exactly (bookkeeping test).
    """

    def __init__(self, utils, market, c, vega_obs=True, m_x=3, psiC=1.5, gamma1=0.5, gamma2=0.5):
        self.bk = Book(utils, market, c)
        self.m = {k: np.asarray(market[k]) for k in ("S", "V", "l_price", "l_delta", "h_price", "h_delta")}
        n0 = min(64, self.bk.P)
        self.L = float(np.median(self.m["l_price"][:n0, 0] / self.bk.liab(
            np.arange(n0), 0, self.m["S"][:n0, :1], self.m["V"][:n0, :1], 0)[0][:, 0]))
        self.vega_obs = vega_obs
        self.T = self.bk.T1 - 1
        lam = getattr(utils, "lambda_v", 0.0)
        kappa = utils.kappa - lam
        theta = utils.kappa * utils.theta / kappa
        xi, rho, dt = utils.volvol, utils.rho, utils.dt
        self.kappa, self.theta, self.xi, self.dt, self.psiC = kappa, theta, xi, dt, psiC
        self.drift = (utils.r + utils.mu - utils.q) * dt
        self.E = np.exp(-kappa * dt)
        self.K1 = (kappa * rho / xi - 0.5) * gamma1 * dt - rho / xi
        self.K2 = (kappa * rho / xi - 0.5) * gamma2 * dt + rho / xi
        self.K3 = (1 - rho ** 2) * gamma1 * dt
        self.K4 = (1 - rho ** 2) * gamma2 * dt
        self.A = self.K2 + 0.5 * self.K4
        self.m_x = m_x
        self.zh, self.wh = gh_nodes(m_x)                        # normal nodes (Z1 and Z2)
        xl, wl = np.polynomial.laguerre.laggauss(max(m_x - 1, 1))  # Exp(1) nodes
        self.xl, self.wl = xl, wl / wl.sum()
        self.M = m_x * m_x
        self.w = np.outer(self.wh, self.wh).ravel()  # weights in the psi <= psiC regime (most transitions)

    def nodes(self, S, vol):
        """(B,) -> S1, vol1, w, each (B, M) with M = m_x^2 (index = i_v * m_x + i_z2)."""
        v = vol ** 2
        E, k, th, xi = self.E, self.kappa, self.theta, self.xi
        mm = th + (v - th) * E
        s2 = v * xi ** 2 * E / k * (1 - E) + th * xi ** 2 / (2 * k) * (1 - E) ** 2
        psi = s2 / mm ** 2
        B, mx = len(v), self.m_x
        v1 = np.zeros((B, mx))
        wv = np.zeros((B, mx))
        K0s = np.zeros(B)
        lo = psi <= self.psiC
        if np.any(lo):
            ps = psi[lo]
            b2 = 2 / ps - 1 + np.sqrt(2 / ps * (2 / ps - 1))
            b = np.sqrt(b2)
            a = mm[lo] / (1 + b2)
            v1[lo] = a[:, None] * (b[:, None] + self.zh[None, :]) ** 2
            wv[lo] = self.wh[None, :]
            K0s[lo] = -(self.A * b2 * a) / (1 - 2 * self.A * a) + 0.5 * np.log(1 - 2 * self.A * a)
        hi = ~lo
        if np.any(hi):
            ps = psi[hi]
            pp = (ps - 1) / (ps + 1)
            beta = (1 - pp) / mm[hi]
            v1[hi, 0] = 0.0
            wv[hi, 0] = pp
            v1[hi, 1:] = (self.xl[None, :mx - 1] / beta[:, None])
            wv[hi, 1:] = (1 - pp)[:, None] * self.wl[None, :mx - 1]
            K0s[hi] = -np.log(pp + beta * (1 - pp) / (beta - self.A))
        K0s = K0s - (self.K1 + 0.5 * self.K3) * v
        # combine with Z2 nodes
        v1f = np.repeat(v1, mx, axis=1)                       # (B, M): i_v major
        z2 = np.tile(self.zh, mx)[None, :]
        w = np.repeat(wv, mx, axis=1) * np.tile(self.wh, mx)[None, :]
        lnS1 = (np.log(S)[:, None] + self.drift + K0s[:, None] + self.K1 * v[:, None] + self.K2 * v1f
                + np.sqrt(np.maximum(self.K3 * v[:, None] + self.K4 * v1f, 0.0)) * z2)
        return np.exp(lnS1), np.sqrt(v1f), w

    def transition(self, p, t, n, cost, realised=False):
        m, bk, B = self.m, self.bk, len(p)
        S, vol = m["S"][p, t], m["V"][p, t]
        if realised:
            S1, vol1, w = m["S"][p, t + 1][:, None], m["V"][p, t + 1][:, None], np.ones((len(p), 1))
        else:
            S1, vol1, w = self.nodes(S, vol)
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
