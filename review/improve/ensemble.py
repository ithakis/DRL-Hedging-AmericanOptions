"""Batch 1, idea E1: seed ensemble of the FINAL actors (validation only, no training).

The ensemble policy plays the average (or median) residual u of the 10 selected actors of one cost. All members use
the same env flags (pre-commitment state W_t, band, residual on rule (b)), so the ensemble is a Markov policy on the
same state. Evaluated on the validation market (seed 2025, 8,192 paths, adi_trading); the test sets stay closed.

    PYTHONPATH=. python review/improve/ensemble.py [LOGS_DIR] [TAG]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from review.retrain import analysis as A
from review.retrain.settings import COSTS

LOGS = Path(sys.argv[1] if len(sys.argv) > 1 else "../DRL-Hedging-AmericanOptions/logs_torch/final")
TAG = sys.argv[2] if len(sys.argv) > 2 else "FINAL"
OUT = Path("review/results/improve")


class Ensemble(torch.nn.Module):
    def __init__(self, actors, how="mean"):
        super().__init__()
        self.actors, self.how = actors, how

    def forward(self, obs):
        u = torch.stack([a(obs) for a in self.actors], 0)
        return u.mean(0) if self.how == "mean" else u.median(0).values


def main():
    import train_d4pg as T
    from torch_d4pg.agent import D4PGQR
    from torch_d4pg.hedge_env import HedgingEnv
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", "adi_trading",
                           "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
    vs = int(os.environ.get("VAL_SEED", "2025"))
    val = A.T.get_market(a, vs, 8192)
    P = val["S"].shape[0]
    idx = np.random.default_rng(7).integers(0, P, (500, P))
    rows = []
    for c in COSTS:
        runs = []
        for d in sorted(LOGS.glob(f"HestonP_var_spread{c}_seed*_{TAG}")):
            if (d / "results.json").exists():
                runs.append((json.load(open(d / "results.json"))["val_score"], d))
        runs.sort(key=lambda x: x[0])
        cfg = json.load(open(runs[0][1] / "config.json"))
        HedgingEnv.extra_obs = bool(cfg.get("extra_obs"))
        HedgingEnv.rule_obs = bool(cfg.get("rule_obs"))
        HedgingEnv.band = float(cfg.get("band") or 0.0)
        HedgingEnv.pnl_obs = cfg.get("precommit_b") is not None
        HedgingEnv.rule_frac = (tuple(float(x) for x in cfg["warm_rule"].split(",")) if cfg.get("rule_frac_obs")
                                else None)
        ns = argparse.Namespace(**cfg)
        actors = [D4PGQR.load_actor(d / "best_actor.pt")[0].eval() for _, d in runs]
        pols = {f"seed{json.load(open(d / 'config.json'))['seed']}": act for (_, d), act in zip(runs, actors)}
        pols.update({"ens_mean_all": Ensemble(actors), "ens_median_all": Ensemble(actors, "median"),
                     "ens_mean_top5": Ensemble(actors[:5]), "ens_mean_top3": Ensemble(actors[:3])})
        pn = {}
        for name, pol in pols.items():
            p, cost = T.evaluate(pol, val, c, cfg["vega_obs"], a=ns)
            pn[name] = p
            v, cv = A.var_cvar(p)
            rows.append(dict(cost=c, policy=name, val_VaR95=float(v), val_CVaR95=float(cv),
                             mean_cost=float(cost.mean()), mean_pnl=float(p.mean())))
        # paired SE of ensemble vs median member
        mem = sorted([k for k in pn if k.startswith("seed")], key=lambda k: A.var_cvar(pn[k])[0])
        med = mem[(len(mem) - 1) // 2]
        for r in rows:
            if r["cost"] == c and r["policy"].startswith("ens"):
                gb = A.var_cvar(pn[med][idx])[0] - A.var_cvar(pn[r["policy"]][idx])[0]
                r["gain_vs_median_member"] = float(A.var_cvar(pn[med])[0] - r["val_VaR95"])
                r["gain_se"] = float(gb.std(ddof=1))
        print(pd.DataFrame([r for r in rows if r["cost"] == c]).round(3).to_string(index=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"ensemble_{TAG}_val{vs}.csv", index=False)


if __name__ == "__main__":
    main()
