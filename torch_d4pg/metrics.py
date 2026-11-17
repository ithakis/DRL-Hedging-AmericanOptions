"""Risk metrics of episode P&L with bootstrap standard errors.

Same definitions as experiments/heston_rule_baselines.py (so RL and rule baselines are comparable):
loss = -P&L, VaR95 = 95% quantile of the loss, CVaR95 = mean loss at or beyond VaR95.
"""

import numpy as np


def var_cvar(pnl, alpha=0.95):
    loss = -np.asarray(pnl)
    v = np.quantile(loss, alpha, axis=-1)
    tail = loss >= v[..., None] if loss.ndim > 1 else loss >= v
    c = (loss * tail).sum(-1) / tail.sum(-1)
    return v, c


def objective_value(pnl, obj_func, threshold=0.95, std_coef=1.645):
    """Selection score, lower is better (VaR / CVaR of the loss, or -(mean - c*std) of the P&L)."""
    pnl = np.asarray(pnl)
    if obj_func == "meanstd":
        return -(pnl.mean(-1) - std_coef * pnl.std(-1))
    v, c = var_cvar(pnl, threshold)
    return v if obj_func == "var" else c


def summarize(pnl, costs=None, n_boot=1000, seed=1, alpha=0.95, std_coef=1.645):
    """Point values and bootstrap SEs (resampling paths)."""
    pnl = np.asarray(pnl, dtype=float)
    P = len(pnl)
    idx = np.random.default_rng(seed).integers(0, P, (n_boot, P))
    bs = pnl[idx]
    v, c = var_cvar(pnl, alpha)
    bv, bc = var_cvar(bs, alpha)
    ms = pnl.mean() - std_coef * pnl.std()
    bms = bs.mean(1) - std_coef * bs.std(1)
    out = dict(
        n_paths=P,
        mean_pnl=pnl.mean(), mean_pnl_se=pnl.std(ddof=1) / np.sqrt(P),
        std_pnl=pnl.std(), std_pnl_se=bs.std(1).std(ddof=1),
        VaR95=v, VaR95_se=bv.std(ddof=1),
        CVaR95=c, CVaR95_se=bc.std(ddof=1),
        meanstd=ms, meanstd_se=bms.std(ddof=1),
    )
    if costs is not None:
        costs = np.asarray(costs, dtype=float)
        out.update(mean_cost=costs.mean(), mean_cost_se=costs.std(ddof=1) / np.sqrt(P))
    return {k: float(x) for k, x in out.items()}
