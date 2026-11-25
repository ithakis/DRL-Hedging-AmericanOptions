"""Precomputed price / Greeks table for American puts under Heston.

Why
---
The environment prices every option on every path and day with its own QuantLib
FD solve (``Utils._American_Option_Heston``).  This takes about 2 hours per
8,192 paths.  An American put is homogeneous of degree 1 in (S, K):

    P(S, K, tau, v) = K * p(S/K, tau, v)

so one table over moneyness m = S/K, time to maturity tau (in trading days)
and volatility sqrt(v) is enough for all strikes.

How
---
* Every grid node is priced with exactly the same QuantLib engine as
  ``Utils.American_put_option`` (FdHestonVanillaEngine, Modified Craig-Sneyd,
  ``N_gridpoints`` = 50 for 'low' / 110 for 'high'); price, delta and gamma
  are the engine outputs.  Vega has the same meaning (parallel shift of
  sqrt(v0) and sqrt(theta), per 1 vol point) but by default uses a central
  0.5 vol-point bump: the original tiny forward bump is noisy (see vega_mode).
* tau is an integer number of trading days (the environment steps one day at a
  time), so the tau axis is exact: no interpolation in time.
* For each tau a bicubic spline in (log m, vol) is fitted for price, delta,
  gamma and vega separately.
* Points outside the moneyness grid get intrinsic value (deep ITM: price K-S,
  delta -1) or zero (deep OTM).  Volatility is clamped to the grid.  Both are
  counted in ``HestonAmericanPutTable.n_out_of_grid``.

All model parameters are in ``HestonTableConfig`` so a robustness bump gets its
own table (the file name contains a hash of the config).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

DEFAULT_TABLE_DIR = Path(__file__).resolve().parent.parent / "pricing_tables"


@dataclasses.dataclass(frozen=True)
class HestonTableConfig:
    # Heston (risk-neutral) parameters used by the pricer
    kappa: float = 1.0
    theta: float = 0.09
    volvol: float = 0.3
    rho: float = -0.7
    r: float = 0.0
    q: float = 0.0
    TradingDaysPerYear: int = 252
    # Grid
    max_ttm: int = 60  # trading days
    m_min: float = 0.5
    m_max: float = 2.0
    n_m: int = 81
    vol_min: float = 0.02
    vol_max: float = 1.0
    n_vol: int = 25
    # QuantLib engine: 50 = numerical_accuracy 'low', 110 = 'high'
    N_gridpoints: int = 50
    # Vega (per 1 vol point) = parallel shift of sqrt(v0) and sqrt(theta), as in Utils.American_put_option.
    # 'central': central difference with an absolute vol bump vega_h (robust, default).
    # 'env_forward': the original forward bump vol_eps = max((0.02*vol)^2, 2e-5).  This bump is so
    #   small that QuantLib FD mesh changes give large spikes (seen deep ITM and at some (tau, v0)).
    vega_mode: str = "central"
    vega_h: float = 0.005
    eps_iv: float = 2e-2

    def key(self) -> str:
        s = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:10]

    @classmethod
    def from_utils(cls, utils, **overrides):
        n = 50 if utils.numerical_accuracy == "low" else 110
        kw = dict(
            kappa=float(utils.kappa), theta=float(utils.theta), volvol=float(utils.volvol),
            rho=float(utils.rho), r=float(utils.r), q=float(utils.q),
            TradingDaysPerYear=int(utils.TradingDaysPerYear), N_gridpoints=n,
            max_ttm=int(max(max(utils.ttms), utils.init_ttm, utils.hed_ttm)),
        )
        kw.update(overrides)
        return cls(**kw)


# ----------------------------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------------------------
def log_m_grid(cfg: HestonTableConfig, tau_days: int) -> np.ndarray:
    """Log-moneyness grid for one tau, dense around ATM (sinh stretching).

    The core width scales with sqrt(tau) so short maturities (gamma spike) get
    fine spacing near the strike.  Two sinh branches cover the asymmetric range
    [log m_min, log m_max]."""
    T = tau_days / cfg.TradingDaysPerYear
    w = 0.15 * np.sqrt(T)  # core width
    lo, hi = np.log(cfg.m_min), np.log(cfg.m_max)
    n_lo = cfg.n_m // 2
    n_hi = cfg.n_m - n_lo - 1
    u_lo = np.linspace(-1.0, 0.0, n_lo + 1)[:-1]
    u_hi = np.linspace(0.0, 1.0, n_hi + 1)[1:]
    x_lo = w * np.sinh(u_lo * np.arcsinh(-lo / w))
    x_hi = w * np.sinh(u_hi * np.arcsinh(hi / w))
    return np.concatenate([x_lo, [0.0], x_hi])


def vol_grid(cfg: HestonTableConfig) -> np.ndarray:
    """Volatility grid, dense where the simulated paths live (about 0.1 to 0.6)."""
    u = np.linspace(0.0, 1.0, cfg.n_vol)
    # map u -> vol with more points in the middle
    mid, half = 0.33, 0.5
    z = np.sinh((u - 0.5) * 2 * np.arcsinh(1.0 / 0.6)) * 0.6  # in [-1, 1]
    vols = np.where(z < 0, mid + z * (mid - cfg.vol_min), mid + z * (cfg.vol_max - mid))
    del half
    return np.unique(np.round(vols, 6))


# ----------------------------------------------------------------------------------------------
# One grid node, identical to Utils.American_put_option (Heston branch)
# ----------------------------------------------------------------------------------------------
def heston_point(S, K, tau_days, vol, cfg: HestonTableConfig):
    """Price, delta, gamma, vega with the same engine and bump as the environment."""
    from environment.utils import Utils  # local import: QuantLib only needed here

    fT = max(tau_days, 1) / cfg.TradingDaysPerYear
    v0 = vol**2
    common = dict(S0=S, K=K, T=fT, r=cfg.r, q=cfg.q, kappa=cfg.kappa, volvol=cfg.volvol, rho=cfg.rho,
                  N_gridpoints=cfg.N_gridpoints, Type="put", exercise="American")
    price = lambda sig, sq_theta: Utils._American_Option_Heston(v0=sig**2, theta=sq_theta**2, **common)
    sq_theta = np.sqrt(cfg.theta)
    C0, delta, gamma = price(vol, sq_theta)
    if cfg.vega_mode == "env_forward":
        vol_eps = max((vol * cfg.eps_iv) ** 2, 0.00002)
        vega = (1 / 100) * (price(vol + vol_eps, sq_theta + vol_eps)[0] - C0) / vol_eps
    elif cfg.vega_mode == "central":
        h = cfg.vega_h
        vega = (1 / 100) * (price(vol + h, sq_theta + h)[0] - price(vol - h, sq_theta - h)[0]) / (2 * h)
    else:
        raise ValueError(cfg.vega_mode)
    return C0, delta, gamma, vega


def _price_tau(tau, cfg):
    x = log_m_grid(cfg, tau)
    vols = vol_grid(cfg)
    out = np.empty((4, len(x), len(vols)))
    for i, xi in enumerate(x):
        for j, s in enumerate(vols):
            out[:, i, j] = heston_point(float(np.exp(xi)), 1.0, tau, float(s), cfg)
    return tau, x, out


def build_table(cfg: HestonTableConfig, n_jobs: int = -1, verbose: bool = True) -> dict:
    """Price all grid nodes (K = 1).  Parallel over tau."""
    from joblib import Parallel, delayed

    t0 = time.time()
    taus = list(range(1, cfg.max_ttm + 1))
    res = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(delayed(_price_tau)(t, cfg) for t in taus)
    res.sort(key=lambda r: r[0])
    data = {
        "taus": np.array(taus),
        "log_m": np.stack([r[1] for r in res]),  # (n_tau, n_m)
        "vols": vol_grid(cfg),
        "values": np.stack([r[2] for r in res]),  # (n_tau, 4, n_m, n_vol)
        "config": json.dumps(dataclasses.asdict(cfg)),
        "build_seconds": time.time() - t0,
    }
    if verbose:
        print(f"Heston table built in {data['build_seconds']:.0f}s "
              f"({len(taus) * cfg.n_m * cfg.n_vol * (3 if cfg.vega_mode == 'central' else 2)} QuantLib solves)")
    return data


class HestonAmericanPutTable:
    """Interpolated American put price / delta / gamma / vega under Heston."""

    def __init__(self, data: dict):
        self.cfg = HestonTableConfig(**json.loads(str(data["config"])))
        self.taus = np.asarray(data["taus"])
        self.log_m = np.asarray(data["log_m"])
        self.vols = np.asarray(data["vols"])
        self.values = np.asarray(data["values"])
        self._splines = {}
        for k, tau in enumerate(self.taus):
            self._splines[int(tau)] = [
                RectBivariateSpline(self.log_m[k], self.vols, self.values[k, q], kx=3, ky=3, s=0)
                for q in range(4)
            ]
        self.n_out_of_grid = {"m_low": 0, "m_high": 0, "vol": 0}
        self._build_fast()

    # ------------------------------------------------------------------ fast evaluation
    # The same bicubic splines, evaluated for all points and all four outputs in a few vectorised
    # numpy calls (no Python loop over tau).  Uses the FITPACK knots and coefficients of the
    # RectBivariateSpline objects above, so the values equal spline.ev() up to rounding (~1e-15).
    fast = True  # class switch: False = the original per-tau loop over spline.ev()

    def _build_fast(self):
        n_tau = len(self.taus)
        assert np.array_equal(self.taus, np.arange(self.taus[0], self.taus[0] + n_tau)), "taus must be contiguous"
        tx, ty, C = [], None, []
        for k, tau in enumerate(self.taus):
            sp = self._splines[int(tau)]
            kx, ky = sp[0].get_knots()
            if ty is None:
                ty = ky
            assert np.array_equal(ky, ty), "vol knots must be the same for all tau"
            tx.append(kx)
            nx, ny = len(kx) - 4, len(ky) - 4
            C.append(np.stack([s.get_coeffs().reshape(nx, ny) for s in sp], axis=-1))  # (nx, ny, 4)
        self._tx = np.stack(tx)                    # (n_tau, Lx)
        self._ty = np.asarray(ty)                  # (Ly,)
        self._C = np.stack(C)                      # (n_tau, nx, ny, 4)
        Lx = self._tx.shape[1]
        # one sorted array for all tau: row k shifted by k * span, so a single searchsorted finds the
        # knot interval of every point in its own row
        self._span = float(self._tx.max() - self._tx.min()) + 10.0
        self._tx_flat = (self._tx + self._span * np.arange(n_tau)[:, None]).ravel()
        self._Lx = Lx

    @staticmethod
    def _basis3(knot, i, x):
        """Cubic B-spline basis values N_{i-3..i}(x) (de Boor), vectorised.  knot(o) -> t[i + o] per point."""
        left = [None] + [x - knot(1 - j) for j in (1, 2, 3)]
        right = [None] + [knot(j) - x for j in (1, 2, 3)]
        N = [np.ones_like(x)]
        for j in (1, 2, 3):
            saved = np.zeros_like(x)
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N.append(saved)
        return np.stack(N, axis=1)  # (P, 4)

    def _ev_fast(self, k, x, v):
        """All four outputs at points (tau index k, log m x, vol v), all inside the grid.  -> (4, P)."""
        Lx, Ly = self._Lx, len(self._ty)
        pos = np.searchsorted(self._tx_flat, x + self._span * k, side="right") - 1 - k * Lx
        i = np.clip(pos, 3, Lx - 5)
        j = np.clip(np.searchsorted(self._ty, v, side="right") - 1, 3, Ly - 5)
        txk = self._tx
        Nx = self._basis3(lambda o: txk[k, i + o], i, x)
        Ny = self._basis3(lambda o: self._ty[j + o], j, v)
        a = np.arange(-3, 1)
        G = self._C[k[:, None, None], (i[:, None] + a)[:, :, None], (j[:, None] + a)[:, None, :]]  # (P, 4, 4, 4)
        return np.einsum("pa,pb,pabq->qp", Nx, Ny, G)

    # ------------------------------------------------------------------ io
    @classmethod
    def load_or_build(cls, cfg: HestonTableConfig, table_dir=None, n_jobs=-1, verbose=True):
        table_dir = Path(table_dir) if table_dir else DEFAULT_TABLE_DIR
        table_dir.mkdir(parents=True, exist_ok=True)
        path = table_dir / f"heston_american_put_{cfg.key()}.npz"
        if path.exists():
            if verbose:
                print(f"Loading Heston pricing table `{path}`")
            data = dict(np.load(path, allow_pickle=False))
        else:
            if verbose:
                print(f"Building Heston pricing table `{path}` for {cfg}")
            data = build_table(cfg, n_jobs=n_jobs, verbose=verbose)
            np.savez_compressed(path, **data)
        return cls(data)

    @classmethod
    def load_or_build_adi(cls, cfg: HestonTableConfig, table_dir=None, quantlib_maturity=True, verbose=True):
        """Single-sweep ADI table (environment/heston_adi.py, 400 x 200, 8 steps per day), cached."""
        from environment.heston_adi import ADIGrid, build_table_adi

        g = ADIGrid(ns=400, nv=200, steps_per_day=8)
        table_dir = Path(table_dir) if table_dir else DEFAULT_TABLE_DIR
        table_dir.mkdir(parents=True, exist_ok=True)
        mode = "qldays" if quantlib_maturity else "trading"
        path = table_dir / f"heston_american_put_adi_{mode}_{cfg.key()}_{g.ns}x{g.nv}x{g.steps_per_day}.npz"
        if path.exists():
            if verbose:
                print(f"Loading ADI Heston table `{path}`")
            return cls(dict(np.load(path, allow_pickle=False)))
        data = build_table_adi(cfg, g, verbose=verbose, quantlib_maturity=quantlib_maturity)
        np.savez_compressed(path, **data)
        return cls(data)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, S, K, ttm_days, vol):
        """Vectorised price, delta, gamma, vega for strike K (any shape, broadcastable).

        ttm_days must be integers.  ttm == 0 gives the payoff, ttm < 0 gives zeros
        (same convention as Utils.American_put_option)."""
        S, K, ttm, vol = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (S, K, ttm_days, vol)))
        shape = S.shape
        S, K, ttm, vol = (a.ravel() for a in (S, K, ttm, vol))
        out = np.zeros((4, S.size))
        matured = ttm == 0
        out[0, matured] = np.maximum(K[matured] - S[matured], 0.0)

        active = ttm > 0
        if np.any(active):
            if np.any(np.abs(ttm[active] - np.round(ttm[active])) > 1e-9):
                raise ValueError("Heston table needs integer time to maturity (in days).")
            if ttm[active].max() > self.taus.max():
                raise ValueError(f"ttm {ttm[active].max()} is above the table max {self.taus.max()}.")
            idx = np.flatnonzero(active)
            x = np.log(S[idx] / K[idx])
            v = vol[idx]
            v_clamped = np.clip(v, self.vols[0], self.vols[-1])
            self.n_out_of_grid["vol"] += int(np.sum(v_clamped != v))
            tt = np.round(ttm[idx]).astype(int)
            if self.fast:
                kk = tt - int(self.taus[0])
                lo, hi = self.log_m[kk, 0], self.log_m[kk, -1]
                inside = (x >= lo) & (x <= hi)
                below = x < lo  # deep ITM put
                self.n_out_of_grid["m_low"] += int(below.sum())
                self.n_out_of_grid["m_high"] += int((x > hi).sum())
                if np.any(inside):
                    out[:, idx[inside]] = self._ev_fast(kk[inside], x[inside], v_clamped[inside])
                if np.any(below):
                    out[0, idx[below]] = 1.0 - np.exp(x[below])  # per unit strike
                    out[1, idx[below]] = -1.0
            for tau in (() if self.fast else np.unique(tt)):
                sel = tt == tau
                k = int(tau) - int(self.taus[0])
                xs, vs, ii = x[sel], v_clamped[sel], idx[sel]
                lo, hi = self.log_m[k, 0], self.log_m[k, -1]
                inside = (xs >= lo) & (xs <= hi)
                below = xs < lo  # deep ITM put
                self.n_out_of_grid["m_low"] += int(below.sum())
                self.n_out_of_grid["m_high"] += int((xs > hi).sum())
                spl = self._splines[int(tau)]
                if np.any(inside):
                    for q in range(4):
                        out[q, ii[inside]] = spl[q].ev(xs[inside], vs[inside])
                if np.any(below):
                    out[0, ii[below]] = 1.0 - np.exp(xs[below])  # per unit strike
                    out[1, ii[below]] = -1.0
            # homogeneity: back to strike K
            out[0, idx] *= K[idx]
            out[2, idx] /= K[idx]
            out[3, idx] *= K[idx]
        return tuple(o.reshape(shape) for o in out)
