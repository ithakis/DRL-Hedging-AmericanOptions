"""American put under Heston on a full (S, v) grid with ONE backward sweep (ADI + Ikonen-Toivanen).

Why: the pricing table (environment/heston_table.py) is built with 364,500 single-point QuantLib solves
(60 maturities x 81 moneyness x 25 vols x 3 for the vega bump).  One PDE solve already gives the
price on the whole (S, v) grid, and one backward run over 60 trading days passes every shorter
maturity on the way.  With the strike homogeneity P(S, K) = K p(S/K) the full table needs 3 sweeps
(base and the two theta-bumps of the vega definition).

Scheme (as recommended for American options under Heston):
  * PDE in tau (time to maturity), K = 1:
        u_tau = 1/2 s^2 v u_ss + rho xi s v u_sv + 1/2 xi^2 v u_vv + (r - q) s u_s + kappa (theta - v) u_v - r u
  * non-uniform grids (in 't Hout & Foulon 2010): s dense around the strike, v dense near 0,
    second-order central differences (mixed derivative: product of first-difference stencils),
    boundaries: s = 0 Dirichlet (u = K), s = S_max and v = V_max Neumann (u_s = 0 / u_v = 0),
    v = 0: the PDE itself with a one-sided v-derivative;
  * Modified Craig-Sneyd ADI, theta = 1/3 (in 't Hout & Foulon 2010; Haentjens & in 't Hout 2015);
  * early exercise by Ikonen-Toivanen operator splitting (Haentjens & in 't Hout 2015):
        U~ = ADI step of u' = A u + lambda_n,   U_{n+1} = max(U~ - dt lambda_n, g),
        lambda_{n+1} = max(0, lambda_n + (g - U~) / dt);
  * damping: the first step of the first day is replaced by two implicit-Euler half steps (Rannacher).
Greeks: delta and gamma by finite differences on the grid; vega with the SAME definition as the
table (parallel shift h of sqrt(v0) and sqrt(theta), central, per 1 vol point).
"""
import json
import dataclasses
import time

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import splu


@dataclasses.dataclass(frozen=True)
class ADIGrid:
    ns: int = 200            # points in s (K = 1)
    nv: int = 100            # points in v
    steps_per_day: int = 4   # ADI steps per trading day
    s_max: float = 8.0
    v_max: float = 5.0
    c_s: float = 0.1         # s-grid concentration around the strike (smaller = denser), in units of K
    d_v: float = 0.01        # v-grid concentration near v = 0 (in units of v_max)
    mcs_theta: float = 1.0 / 3.0


def s_grid(g: ADIGrid):
    c = g.c_s
    xi = np.linspace(np.arcsinh(-1.0 / c), np.arcsinh((g.s_max - 1.0) / c), g.ns)
    s = 1.0 + c * np.sinh(xi)
    s[0] = 0.0
    return s


def v_grid(g: ADIGrid):
    d = g.d_v * g.v_max
    eta = np.linspace(0.0, np.arcsinh(g.v_max / d), g.nv)
    return d * np.sinh(eta)


def _central_weights(x):
    """Weights of the 3-point first and second derivative at interior points of a non-uniform grid.
    Returns (b_m, b_0, b_p), (d_m, d_0, d_p) for i = 1..n-2."""
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    b = (-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp)))
    d = (2.0 / (hm * (hm + hp)), -2.0 / (hm * hp), 2.0 / (hp * (hm + hp)))
    return b, d


def build_operators(s, v, kappa, theta, xi, rho, r, q):
    """Sparse A0 (mixed), A1 (s-direction), A2 (v-direction) on u.ravel() with index i * nv + j."""
    ns, nv = len(s), len(v)
    idx = lambda i, j: i * nv + j
    bs, ds = _central_weights(s)
    bv, dv = _central_weights(v)
    rows, cols, vals = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}

    def add(k, I, J, V):
        rows[k].append(np.ravel(I))
        cols[k].append(np.ravel(J))
        vals[k].append(np.ravel(V))

    jj = np.arange(nv)[None, :]
    # ---------------- A1: s-direction, rows i = 1..ns-1 (i = 0 Dirichlet: zero row)
    ii = np.arange(1, ns - 1)[:, None]
    S, V = s[1:-1][:, None], v[None, :]
    a_ss, a_s = 0.5 * S ** 2 * V, (r - q) * S
    for k, off in enumerate((-1, 0, 1)):
        coef = a_ss * ds[k][:, None] + a_s * bs[k][:, None]
        add(1, np.broadcast_to(idx(ii, jj), coef.shape), np.broadcast_to(idx(ii + off, jj), coef.shape), coef)
    i = ns - 1  # Neumann u_s = 0: u_ss = 2 (u_{i-1} - u_i) / h^2
    h = s[-1] - s[-2]
    a = 0.5 * s[-1] ** 2 * v / h ** 2 * 2.0
    add(1, idx(i, jj), idx(i - 1, jj), a)
    add(1, idx(i, jj), idx(i, jj), -a)
    ii_all = np.arange(1, ns)[:, None]
    add(1, np.broadcast_to(idx(ii_all, jj), (ns - 1, nv)), np.broadcast_to(idx(ii_all, jj), (ns - 1, nv)),
        np.full((ns - 1, nv), -0.5 * r))
    # ---------------- A2: v-direction, rows i >= 1
    ii = np.arange(1, ns)[:, None]
    J = np.arange(1, nv - 1)[None, :]
    Vi = v[1:-1][None, :]
    a_vv, a_v = 0.5 * xi ** 2 * Vi, kappa * (theta - Vi)
    for k, off in enumerate((-1, 0, 1)):
        coef = np.broadcast_to(a_vv * dv[k][None, :] + a_v * bv[k][None, :], (ns - 1, nv - 2))
        add(2, np.broadcast_to(idx(ii, J), coef.shape), np.broadcast_to(idx(ii, J + off), coef.shape), coef)
    # v = 0: kappa theta u_v, one-sided second-order
    h1, h2 = v[1] - v[0], v[2] - v[1]
    w = (-(2 * h1 + h2) / (h1 * (h1 + h2)), (h1 + h2) / (h1 * h2), -h1 / (h2 * (h1 + h2)))
    for k in range(3):
        add(2, idx(ii, 0), idx(ii, k), np.full(ii.shape, kappa * theta * w[k]))
    # v = v_max: Neumann u_v = 0
    h = v[-1] - v[-2]
    a = 0.5 * xi ** 2 * v[-1] * 2.0 / h ** 2
    add(2, idx(ii, nv - 1), idx(ii, nv - 2), np.full(ii.shape, a))
    add(2, idx(ii, nv - 1), idx(ii, nv - 1), np.full(ii.shape, -a))
    add(2, np.broadcast_to(idx(ii, jj), (ns - 1, nv)), np.broadcast_to(idx(ii, jj), (ns - 1, nv)),
        np.full((ns - 1, nv), -0.5 * r))
    # ---------------- A0: mixed derivative, interior in both directions
    ii = np.arange(1, ns - 1)[:, None]
    J = np.arange(1, nv - 1)[None, :]
    a_sv = rho * xi * s[1:-1][:, None] * v[1:-1][None, :]
    for k, oi in enumerate((-1, 0, 1)):
        for l, oj in enumerate((-1, 0, 1)):
            coef = a_sv * bs[k][:, None] * bv[l][None, :]
            add(0, np.broadcast_to(idx(ii, J), coef.shape), np.broadcast_to(idx(ii + oi, J + oj), coef.shape), coef)
    N = ns * nv
    mats = []
    for k in (0, 1, 2):
        mats.append(sp.csr_matrix((np.concatenate(vals[k]), (np.concatenate(rows[k]), np.concatenate(cols[k]))),
                                  shape=(N, N)))
    return mats


def solve_surface(kappa, theta, xi, rho, r, q, n_days, days_per_year=252, grid: ADIGrid = ADIGrid()):
    """Backward sweep.  Returns s (ns,), v (nv,), U (n_days + 1, ns, nv) with U[d] = price at tau = d days."""
    s, v = s_grid(grid), v_grid(grid)
    ns, nv = len(s), len(v)
    A0, A1, A2 = build_operators(s, v, kappa, theta, xi, rho, r, q)
    A = (A0 + A1 + A2).tocsc()
    N = ns * nv
    I = sp.identity(N, format="csc")
    dt = 1.0 / (days_per_year * grid.steps_per_day)
    th = grid.mcs_theta
    lu1 = splu((I - th * dt * A1).tocsc())
    lu2 = splu((I - th * dt * A2).tocsc())
    luE = splu((I - 0.5 * dt * A).tocsc())  # Rannacher half steps
    g = np.broadcast_to(np.maximum(1.0 - s, 0.0)[:, None], (ns, nv)).ravel().copy()
    U = g.copy()
    lam = np.zeros(N)
    out = np.empty((n_days + 1, ns, nv))
    out[0] = U.reshape(ns, nv)

    def it_update(Ubar, lam, h):
        Unew = np.maximum(Ubar - h * lam, g)
        lam = np.maximum(0.0, lam + (g - Ubar) / h)
        return Unew, lam

    for d in range(1, n_days + 1):
        for k in range(grid.steps_per_day):
            if d == 1 and k == 0:
                for _ in range(2):
                    Ubar = luE.solve(U + 0.5 * dt * lam)
                    U, lam = it_update(Ubar, lam, 0.5 * dt)
                continue
            AU0, AU1, AU2 = A0 @ U, A1 @ U, A2 @ U
            Y0 = U + dt * (AU0 + AU1 + AU2 + lam)
            Y1 = lu1.solve(Y0 - th * dt * AU1)
            Y2 = lu2.solve(Y1 - th * dt * AU2)
            Yh0 = Y0 + th * dt * (A0 @ Y2 - AU0)
            Yt0 = Yh0 + (0.5 - th) * dt * (A @ Y2 - (AU0 + AU1 + AU2))
            Yt1 = lu1.solve(Yt0 - th * dt * AU1)
            Ubar = lu2.solve(Yt1 - th * dt * AU2)
            U, lam = it_update(Ubar, lam, dt)
        out[d] = U.reshape(ns, nv)
    return s, v, out


def _delta_gamma(s, U):
    """dU/ds and d2U/ds2 on the grid (interior central, edges copied)."""
    b, d = _central_weights(s)
    De = np.empty_like(U)
    Ga = np.empty_like(U)
    De[1:-1] = b[0][:, None] * U[:-2] + b[1][:, None] * U[1:-1] + b[2][:, None] * U[2:]
    Ga[1:-1] = d[0][:, None] * U[:-2] + d[1][:, None] * U[1:-1] + d[2][:, None] * U[2:]
    De[0], De[-1], Ga[0], Ga[-1] = De[1], De[-2], Ga[1], Ga[-2]
    return De, Ga


def build_table_adi(cfg, grid: ADIGrid = ADIGrid(), verbose=True):
    """Same output format as heston_table.build_table (taus, log_m, vols, values (n_tau, 4, n_m, n_vol))."""
    from environment.heston_table import log_m_grid, vol_grid

    t0 = time.time()
    n_days = cfg.max_ttm
    kw = dict(kappa=cfg.kappa, xi=cfg.volvol, rho=cfg.rho, r=cfg.r, q=cfg.q, n_days=n_days,
              days_per_year=cfg.TradingDaysPerYear, grid=grid)
    h = cfg.vega_h
    sq = np.sqrt(cfg.theta)
    s, v, U = solve_surface(theta=cfg.theta, **kw)
    _, _, Up = solve_surface(theta=(sq + h) ** 2, **kw)
    _, _, Um = solve_surface(theta=(sq - h) ** 2, **kw)
    vols = vol_grid(cfg)
    taus = np.arange(1, n_days + 1)
    log_m = np.stack([log_m_grid(cfg, int(t)) for t in taus])
    values = np.empty((len(taus), 4, log_m.shape[1], len(vols)))
    for k, tau in enumerate(taus):
        x = np.exp(log_m[k])
        De, Ga = _delta_gamma(s, U[tau])
        for qq, F in enumerate((U[tau], De, Ga)):
            values[k, qq] = RectBivariateSpline(s, v, F, kx=3, ky=3, s=0).ev(
                np.repeat(x, len(vols)), np.tile(vols ** 2, len(x))).reshape(len(x), len(vols))
        Pp = RectBivariateSpline(s, v, Up[tau], kx=3, ky=3, s=0).ev(
            np.repeat(x, len(vols)), np.tile((vols + h) ** 2, len(x))).reshape(len(x), len(vols))
        Pm = RectBivariateSpline(s, v, Um[tau], kx=3, ky=3, s=0).ev(
            np.repeat(x, len(vols)), np.tile(np.maximum(vols - h, 0.0) ** 2, len(x))).reshape(len(x), len(vols))
        values[k, 3] = (Pp - Pm) / (2 * h) / 100.0
    data = {
        "taus": taus, "log_m": log_m, "vols": vols, "values": values,
        "config": json.dumps(dataclasses.asdict(cfg)),
        "adi_grid": json.dumps(dataclasses.asdict(grid)),
        "build_seconds": time.time() - t0,
    }
    if verbose:
        print(f"ADI Heston table {grid.ns}x{grid.nv}, {grid.steps_per_day} steps/day: {data['build_seconds']:.1f}s (3 sweeps)")
    return data
