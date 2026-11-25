"""Check: with --rule_frac_obs (and --rule_obs) the kernel's realised node reproduces the env's next state."""
import argparse
import numpy as np
import train_d4pg as T
from review.retrain.settings import BLOCKS
from torch_d4pg.hedge_env import HedgingEnv
from torch_d4pg.heston_kernel import HestonKernel, book_arrivals

a = T.get_args(BLOCKS["B"]["flags"] + ["--obj_func", "var", "--spread", "0.01", "--market_dir", "../DRL-Hedging-retrain/markets",
               "--n_train_paths", "40000", "--pricing_backend", "adi_trading", "--table_dir", "../DRL-Hedging-retrain/pricing_tables",
               "--extra_obs", "--kernel_mx", "3", "--kernel_atm_fix", "--warm_rule", "0.3,0.3,0.5", "--band", "0.5",
               "--rule_obs", "--rule_frac_obs", "--residual_scale", "0.5"])
m = T.get_market(a, a.market_seed, a.n_train_paths)
env = T.make_env(m, a.spread, a.vega_obs, a)
u = T.make_utils(argparse.Namespace(**{**vars(a), "spread": 0.0}), a.market_seed, a.n_train_paths)
u.table_dir = a.table_dir
_, _, c_book = book_arrivals(u)
K = HestonKernel(u, m, c_book, a.vega_obs, m_x=3, extra_obs=True, atm_fix=True)
p = np.arange(64)
obs = env.reset(p)
rng = np.random.default_rng(0)
worst = 0.0
while True:
    t = env.t
    nobs, r, done, x = env.step(rng.uniform(0, 1, len(p)))
    cost = -np.abs(a.spread * x * env.b["h_price"][:, t, t])
    rk, ok, _, _ = K.transition(p, np.full(len(p), t), env.n.copy(), cost, realised=True)
    d = np.abs(ok[:, 0, :] - nobs).max(0)
    worst = max(worst, d[-1])
    print(t, "obs dim", nobs.shape[1], "max |diff| per col", np.round(d, 10), "reward", np.abs(rk[:, 0] - r).max())
    if done:
        break
print("rule_frac column worst diff", worst)
