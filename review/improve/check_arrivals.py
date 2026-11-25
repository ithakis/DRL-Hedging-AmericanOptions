"""Check of the arrival-averaged kernel target: (1) expanding the no-arrival node with the REALISED count reproduces
the original kernel's next state (all columns, incl. rule inputs); (2) Skellam weights sum to the node weight."""
import argparse
import numpy as np
import torch
import train_d4pg as T
from review.retrain.settings import BLOCKS
from torch_d4pg.heston_kernel import HestonKernel, book_arrivals, ArrivalExpander

a = T.get_args(BLOCKS["B"]["flags"] + ["--obj_func", "var", "--spread", "0.01", "--market_dir", "../DRL-Hedging-retrain/markets",
               "--n_train_paths", "40000", "--pricing_backend", "adi_trading", "--table_dir", "../DRL-Hedging-retrain/pricing_tables",
               "--extra_obs", "--kernel_mx", "3", "--kernel_atm_fix", "--warm_rule", "0.3,0.3,0.5", "--band", "0.5",
               "--rule_obs", "--rule_frac_obs", "--residual_scale", "0.5"])
m = T.get_market(a, a.market_seed, a.n_train_paths)
env = T.make_env(m, a.spread, a.vega_obs, a)
u = T.make_utils(argparse.Namespace(**{**vars(a), "spread": 0.0}), a.market_seed, a.n_train_paths)
u.table_dir = a.table_dir
_, _, c_book = book_arrivals(u)
K0 = HestonKernel(u, m, c_book, True, m_x=3, extra_obs=True, atm_fix=True)
K1 = HestonKernel(u, m, c_book, True, m_x=3, extra_obs=True, atm_fix=True, arrivals=True)
E = ArrivalExpander(1.0, 2, dict(rule_obs_col=7, rule_frac_col=9, rule=(0.3, 0.3, 0.5)))
p = np.arange(256); env.reset(p); rng = np.random.default_rng(0); worst = 0
while True:
    t = env.t
    _, r, done, x = env.step(rng.uniform(0, 1, len(p)))
    cost = -np.abs(a.spread * x * env.b["h_price"][:, t, t])
    r0, o0, d0, w0 = K0.transition(p, np.full(len(p), t), env.n.copy(), cost)
    r1, o1, d1, w1 = K1.transition(p, np.full(len(p), t), env.n.copy(), cost)
    ga, va = K1.last_arrival_greeks
    c1 = c_book[p, min(t + 1, c_book.shape[1] - 1)] * (t + 1 <= c_book.shape[1] - 1)
    ok = np.abs(c1) <= 2
    oe, re, we = E(torch.tensor(o1, dtype=torch.float32), torch.tensor(r1, dtype=torch.float32), torch.tensor(w1, dtype=torch.float32),
                   torch.tensor(ga, dtype=torch.float32), torch.tensor(va, dtype=torch.float32), torch.tensor(d1 * np.ones(len(p)), dtype=torch.float32))
    oe = oe.numpy().reshape(len(p), -1, 5, o1.shape[-1])
    sel = oe[np.arange(len(p)), :, (np.clip(c1, -2, 2) + 2).astype(int), :]   # (B, M, D)
    rel = np.abs(sel[ok] - o0[ok]) / (1 + np.abs(o0[ok]))
    worst = max(worst, rel.max())
    assert np.allclose(we.numpy().reshape(len(p), -1, 5).sum(-1), w1, atol=1e-6)
    assert np.allclose(r0, r1)
    if done:
        break
print("max rel diff of expanded (realised k) vs original kernel next state:", worst)
