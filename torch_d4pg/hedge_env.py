"""Vectorised version of environment/Environment.py (TradingEnv + MainPortfolio), numpy only.

It steps a batch of paths at once and reproduces TradingEnv exactly:

  state  s_t = [S_t, Gamma_t, gamma^h_t, (Vega_t, vega^h_t)]           (vega part if vega_obs)
           Gamma_t / Vega_t: liability + hedges bought before t; gamma^h_t / vega^h_t: the new ATM
           hedge option of date t (x contract size).  At the last date s[1:] = 0.
  action a in [0, 1]:   n_G = -Gamma_t / gamma^h_t (nan -> 0),   n_V = -Vega_t / vega^h_t
           low  = min(0, n_G[, n_V]),  high = max(0, n_G[, n_V]),   n_t = low + a (high - low)
  reward = -spread |n_t price^h_t| + liability P&L + hedge-option P&L + stock P&L,
           stock position = -(liability delta + hedge delta) after the trade (delta neutral).

tests/test_torch_d4pg.py checks it against MainPortfolio.step on the same market.
"""

import numpy as np


class HedgingEnv:
    def __init__(self, market, spread: float, vega_obs: bool):
        self.m = market
        self.spread = float(spread)
        self.vega_obs = bool(vega_obs)
        self.P, self.T1 = market["S"].shape
        self.T = self.T1 - 1  # number of hedging steps per episode
        self.obs_dim = 5 if vega_obs else 3

    # ------------------------------------------------------------------
    def reset(self, idx):
        idx = np.atleast_1d(np.asarray(idx))
        m = self.m
        # copy this batch of paths into memory (the full market may be a read-only memmap)
        self.b = {k: np.asarray(m[k][idx]) for k in m}
        self.B = len(idx)
        self.n = np.zeros((self.B, self.T1))
        self.t = 0
        self.pos_prev = np.zeros(self.B)
        return self._obs()

    def _port(self, t):
        b = self.b
        act = ~b["h_inactive"][:, t, :]
        G = b["l_gamma"][:, t] + np.sum(self.n * b["h_gamma"][:, t, :] * act, axis=1)
        V = b["l_vega"][:, t] + np.sum(self.n * b["h_vega"][:, t, :] * act, axis=1)
        return G, V

    def _obs(self):
        t, b = self.t, self.b
        G, V = self._port(t)
        cols = [b["S"][:, t], G, b["h_gamma"][:, t, t]]
        if self.vega_obs:
            cols += [V, b["h_vega"][:, t, t]]
        obs = np.stack(cols, axis=1)
        if t == self.T:
            obs[:, 1:] = 0.0
        return obs

    def bounds(self):
        t, b = self.t, self.b
        G, V = self._port(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_g = np.nan_to_num(-G / b["h_gamma"][:, t, t], nan=0.0, posinf=np.inf, neginf=-np.inf)
            lo, hi = np.minimum(0.0, n_g), np.maximum(0.0, n_g)
            if self.vega_obs:
                n_v = -V / b["h_vega"][:, t, t]
                lo, hi = np.minimum(lo, n_v), np.maximum(hi, n_v)
        return lo, hi

    def step(self, action):
        """action: (B,) hedge fraction in [0, 1]. Returns obs, reward (B,), done (bool), hed_share."""
        t, b = self.t, self.b
        a = np.clip(np.asarray(action, dtype=float).reshape(self.B), 0.0, 1.0)
        lo, hi = self.bounds()
        x = lo + a * (hi - lo)
        self.n[:, t] = x
        cost = -np.abs(self.spread * x * b["h_price"][:, t, t])
        act = ~b["h_inactive"][:, t, :]
        act_next = ~b["h_inactive"][:, t + 1, :]
        pos = -(b["l_delta"][:, t] + np.sum(self.n * b["h_delta"][:, t, :] * act, axis=1))
        pnl = b["l_next"][:, t] - b["l_price"][:, t]
        pnl = pnl + np.sum(self.n * (b["h_price"][:, t + 1, :] - b["h_price"][:, t, :]) * act_next, axis=1)
        pnl = pnl + pos * (b["S"][:, t + 1] - b["S"][:, t])
        self.t = t + 1
        done = self.t == self.T
        return self._obs(), cost + pnl, done, x
