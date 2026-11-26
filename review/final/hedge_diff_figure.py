"""Sven item 4: what the agent does differently from rule (b). Mean of (agent hedge fraction - rule (b) hedge fraction)
in the same state, by day of the episode and running P&L W_t, for the reported (selected) agent at each cost, on the
test paths (seed 4242). The agent plays f = clip(f_rule + 0.5 u, 0, 1) with the rule's band, so f - f_rule is exactly
its deviation from the rule. Dashed line: W_t = -b (the pre-commitment level). Cells with < 100 states are blank.
    PYTHONPATH=. python review/final/hedge_diff_figure.py"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("default")
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from review.retrain import analysis as A
import train_d4pg as T
from torch_d4pg.agent import D4PGQR
from torch_d4pg.hedge_env import HedgingEnv

LOGS = Path("../DRL-Hedging-AmericanOptions/logs_torch/final")
RUNS = {0.005: 4, 0.01: 1, 0.02: 8}  # selected seeds (final_table.md)
OUT = Path("review/results/final")
a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                       "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
m = A.T.get_market(a, 4242, 8192)
Wbins = np.arange(-40, 22.5, 2.5)
cmap = LinearSegmentedColormap.from_list("div", ["#2f62c9", "#f0efec", "#c9412f"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, constrained_layout=True, facecolor="white")
for ax_ in axes:
    ax_.set_facecolor("white")
summary = {}
for ax, (c, s) in zip(axes, RUNS.items()):
    d = LOGS / f"HestonP_var_spread{c}_seed{s}_FINAL"
    cfg = json.load(open(d / "config.json"))
    actor, _ = D4PGQR.load_actor(d / "best_actor.pt")
    HedgingEnv.extra_obs, HedgingEnv.pnl_obs, HedgingEnv.band = True, True, float(cfg["band"])
    env = T.make_env(m, c, True, argparse.Namespace(**cfg))
    obs = env.reset(np.arange(env.P))
    rows, done = [], False
    while not done:
        t, W, fr = env.t, env.W.copy(), env.rule_fraction()
        with torch.inference_mode():
            u = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()[:, 0]
        obs, _, done, _ = env.step((u + 1) / 2)
        rows.append(np.c_[np.full(env.P, t), W, env.last_frac - fr])
    HedgingEnv.extra_obs = HedgingEnv.pnl_obs = False; HedgingEnv.band = 0.0
    X = np.concatenate(rows)
    T_ = int(X[:, 0].max()) + 1
    wi = np.clip(np.digitize(X[:, 1], Wbins) - 1, 0, len(Wbins) - 2)
    S_ = np.zeros((len(Wbins) - 1, T_)); N_ = np.zeros_like(S_)
    np.add.at(S_, (wi, X[:, 0].astype(int)), X[:, 2]); np.add.at(N_, (wi, X[:, 0].astype(int)), 1)
    M = np.where(N_ >= 100, S_ / np.maximum(N_, 1), np.nan)
    im = ax.pcolormesh(np.arange(T_ + 1) + 0.5, Wbins, M, cmap=cmap, norm=TwoSlopeNorm(0, -0.3, 0.3), shading="flat")
    ax.axhline(-cfg["precommit_b"], color="#333", ls="--", lw=1)
    ax.text(T_ - 0.5, -cfg["precommit_b"] + 0.8, "W = -b", ha="right", fontsize=8, color="#333")
    ax.set_title(f"{c:.1%} cost (seed {s})", fontsize=10)
    ax.set_xlabel("Day of the episode (30 = horizon)")
    summary[c] = dict(mean_dev=float(X[:, 2].mean()), share_dev_gt_0_05=float(np.mean(np.abs(X[:, 2]) > 0.05)),
                      mean_dev_past_b=float(X[X[:, 1] < -cfg["precommit_b"], 2].mean()),
                      mean_dev_above_b=float(X[X[:, 1] >= -cfg["precommit_b"], 2].mean()))
axes[0].set_ylabel("Running P&L $W_t$")
fig.colorbar(im, ax=axes, shrink=0.9, label="agent hedge fraction − rule (b) hedge fraction")
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "hedge_diff.pdf", facecolor="white"); fig.savefig(OUT / "hedge_diff.png", dpi=160, facecolor="white")
json.dump(summary, open(OUT / "hedge_diff_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
