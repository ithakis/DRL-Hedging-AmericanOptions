"""Checks for the PyTorch D4PG-QR port.  Run: pytest -q tests/test_torch_d4pg.py

1. The vectorised environment (torch_d4pg/hedge_env.py) gives the same states and rewards as the
   original MainPortfolio / TradingEnv.step logic, GBM (3-dim state) and Heston (5-dim state).
2. The standalone numpy evaluator gives the same P&L as the torch evaluation of the same policy.
3. The fused quantile-Huber loss matches the direct port of agent/distributional.QuantileLoss.
4. Risk objectives on the quantile head match their definitions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from absl import flags

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

if "vega_obs" not in flags.FLAGS:
    flags.DEFINE_boolean("vega_obs", True, "")
flags.FLAGS(["test"])

from environment.Trading import MainPortfolio  # noqa: E402
from environment.market import build_market  # noqa: E402
from torch_d4pg.agent import D4PGQR, export_actor_npz  # noqa: E402
from torch_d4pg.distributional import RiskObjective, quantile_huber_loss, quantile_huber_loss_reference  # noqa: E402
from torch_d4pg.hedge_env import HedgingEnv  # noqa: E402
from torch_d4pg.networks import quantile_grid  # noqa: E402
import train_d4pg  # noqa: E402
import evaluate_policy_numpy as npe  # noqa: E402


def small_utils(process, n=8, seed=7, extra=()):
    a = train_d4pg.get_args(["--process", process, *extra])
    return train_d4pg.make_utils(a, seed, n)


def replay_original(utils, actions, spread, vega_obs):
    """TradingEnv.reset/step logic on MainPortfolio (copied from environment/Environment.py)."""
    flags.FLAGS.vega_obs = vega_obs
    utils.spread = spread
    np.random.seed(utils.seed)
    mp = MainPortfolio(utils)
    cs = utils.contract_size
    P, T1 = mp.a_price.shape
    states, rewards = np.zeros((P, T1, 5 if vega_obs else 3)), np.zeros((P, T1 - 1))

    class R:
        pass

    for ep in range(P):
        mp.reset(ep)
        mp.liab_port.add(ep, 0, utils.num_conts_to_add)
        states[ep, 0] = mp.get_state(0)
        for t in range(T1 - 1):
            g_b = -mp.get_gamma(t) / mp.hed_port.options[ep, t].gamma_path[t] / cs
            g_b = 0 if np.isnan(g_b) else g_b
            lo, hi = [0, g_b], [0, g_b]
            if vega_obs:
                v_b = -mp.get_vega(t) / mp.hed_port.options[ep, t].vega_path[t] / cs
                lo.append(v_b)
                hi.append(v_b)
            share = np.min(lo) + actions[ep, t] * (np.max(hi) - np.min(lo))
            rewards[ep, t], _ = mp.step(share, t, R())
            s = mp.get_state(t + 1)
            if t + 1 == T1 - 1:
                s[1:] = 0
            states[ep, t + 1] = s
    return states, rewards


# third case: paper P-measure setting (r=2%, mu=3%, lambda_v=1.5 -> kappa_Q=2.5, theta_Q=0.036),
# priced with QuantLib directly (no table built for it here), only 4 paths
P_SETTING = ("--r", "0.02", "--mu", "0.03", "--lambda_v", "1.5", "--kappa", "2.5", "--theta", "0.036",
             "--pricing_backend", "quantlib")


@pytest.mark.parametrize("process,vega_obs,extra,n", [("GBM", False, (), 8), ("Heston", True, (), 8),
                                                      ("Heston", True, P_SETTING, 4)])
def test_env_matches_mainportfolio(process, vega_obs, extra, n):
    u = small_utils(process, n=n, extra=extra)
    m = build_market(u)
    P, T1 = m["S"].shape
    actions = np.random.default_rng(0).uniform(0, 1, (P, T1 - 1))
    actions[:, ::5] = 0.0
    actions[:, 1::7] = 1.0
    s_ref, r_ref = replay_original(u, actions, 0.01, vega_obs)
    env = HedgingEnv(m, 0.01, vega_obs)
    obs = env.reset(np.arange(P))
    np.testing.assert_allclose(obs, s_ref[:, 0], rtol=1e-10, atol=1e-10)
    for t in range(T1 - 1):
        obs, rew, done, _ = env.step(actions[:, t])
        np.testing.assert_allclose(rew, r_ref[:, t], rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(obs, s_ref[:, t + 1], rtol=1e-9, atol=1e-8)
    assert done


def test_numpy_evaluator_matches_torch(tmp_path):
    u = small_utils("Heston", n=64)
    m = build_market(u)
    agent = D4PGQR(5, seed=3)
    with torch.no_grad():  # make the policy non-trivial
        agent.actor.head.weight.normal_(0, 0.5)
    pnl_t, cost_t = train_d4pg.evaluate(agent, m, 0.01, True)
    export_actor_npz(agent.actor, tmp_path / "p.npz", dict(spread=0.01, vega_obs=True))
    pol = npe.NumpyPolicy(tmp_path / "p.npz")
    pnl_n, cost_n, acts = npe.run_episodes(pol, m, 0.01, True, return_actions=True)
    assert acts.std() > 0.01
    np.testing.assert_allclose(pnl_n, pnl_t, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(cost_n, cost_t, rtol=1e-4, atol=1e-4)


def test_quantile_loss_matches_reference():
    torch.manual_seed(0)
    q, _ = quantile_grid()
    q = torch.tensor(q, dtype=torch.float32)
    z = (torch.randn(64, 99) * 3).requires_grad_()
    zn, r, d = torch.randn(64, 99) * 3, torch.randn(64, 1), (torch.rand(64, 1) > 0.3).float()
    l1 = quantile_huber_loss(z, q, r, d, zn)
    g1, = torch.autograd.grad(l1.mean(), z)
    l2 = quantile_huber_loss_reference(z, q, r, d, zn)
    g2, = torch.autograd.grad(l2.mean(), z)
    torch.testing.assert_close(l1, l2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-8)


def test_risk_objectives():
    q, p = quantile_grid()
    q, p = torch.tensor(q, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)
    z = torch.sort(torch.randn(4, 99), dim=-1).values
    assert torch.allclose(RiskObjective("var", q, p)(z), z[:, 4])                      # 5% quantile
    assert torch.allclose(RiskObjective("cvar", q, p)(z), 0.01 * z[:, :5].sum(-1))       # as in TF code
    ms = z.mean(-1) - 1.645 * (0.01 * (z - z.mean(-1, keepdim=True)).pow(2).sum(-1)).sqrt()
    assert torch.allclose(RiskObjective("meanstd", q, p)(z), ms)
