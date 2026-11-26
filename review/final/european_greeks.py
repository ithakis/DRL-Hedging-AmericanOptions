"""Sven item 5 / referee 'American is incidental': hedge the American book with EUROPEAN Greeks.

Same paths (identical S, V), same American prices for all P&L; only the Greeks used for the hedge decisions (Gamma,
Vega of the book and the hedge option) and for the delta hedge (Delta) come from a European Heston table
(single-sweep ADI without the early-exercise constraint, maturity tau/252; checked against the analytic Heston price,
review/final/check_european_adi.py). DG and rule (b); rule (b) with European Greeks is re-tuned on validation (seed
2025) on the same grid as the American one. Test: seed 4242. Paired bootstrap SE of (European - American) VaR95.
    PYTHONPATH=. python review/final/european_greeks.py"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review.retrain import analysis as A
from review.retrain.settings import COSTS

OUT = Path("review/results/final")
GRID = dict(a=np.round(np.arange(0, 1.01, 0.1), 2), b=np.round(np.arange(0, 1.01, 0.1), 2),
            w=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5])


def market(seed, backend):
    a = A.block_args("B", ["--market_dir", "../DRL-Hedging-retrain/markets", "--pricing_backend", backend,
                           "--table_dir", "../DRL-Hedging-retrain/pricing_tables"])
    return {k: np.asarray(v) for k, v in A.T.get_market(a, seed, 8192).items()}


def rollout(mp, mg, decide):
    """P&L with the American prices of mp; Greeks for decisions and the delta hedge from mg."""
    S = mp["S"]
    P, T1 = S.shape
    n = np.zeros((P, T1))
    pnl, ob = np.zeros(P), np.zeros(P)
    for t in range(T1 - 1):
        act = ~mp["h_inactive"][:, t, :]
        G = mg["l_gamma"][:, t] + np.sum(n * mg["h_gamma"][:, t, :] * act, axis=1)
        V = mg["l_vega"][:, t] + np.sum(n * mg["h_vega"][:, t, :] * act, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_g = np.nan_to_num(-G / mg["h_gamma"][:, t, t], nan=0.0, posinf=np.inf, neginf=-np.inf)
            n_v = np.nan_to_num(-V / mg["h_vega"][:, t, t], nan=0.0, posinf=np.inf, neginf=-np.inf)
        x = decide(n_g, n_v)
        n[:, t] = x
        ob += np.abs(x * mp["h_price"][:, t, t])
        act_next = ~mp["h_inactive"][:, t + 1, :]
        pos = -(mg["l_delta"][:, t] + np.sum(n * mg["h_delta"][:, t, :] * act, axis=1))
        pnl += mp["l_next"][:, t] - mp["l_price"][:, t]
        pnl += np.sum(n * (mp["h_price"][:, t + 1, :] - mp["h_price"][:, t, :]) * act_next, axis=1)
        pnl += pos * (S[:, t + 1] - S[:, t])
    return pnl, ob


def rule(a, b, w):
    def f(n_g, n_v):
        lo, hi = np.minimum(0, np.minimum(n_g, n_v)), np.maximum(0, np.maximum(n_g, n_v))
        x = np.clip(a * n_g + b * n_v, lo, hi)
        return np.where(np.abs(x) < w, 0.0, x)
    return f


DG = lambda n_g, n_v: n_g


def main():
    mk = {(s, be): market(s, be) for s in (2025, 4242) for be in ("adi_trading", "adi_trading_eu")}
    for s in (2025, 4242):
        assert np.array_equal(mk[(s, "adi_trading")]["S"], mk[(s, "adi_trading_eu")]["S"])
    va, ve = mk[(2025, "adi_trading")], mk[(2025, "adi_trading_eu")]
    ta, te = mk[(4242, "adi_trading")], mk[(4242, "adi_trading_eu")]
    P = ta["S"].shape[0]
    idx = np.random.default_rng(11).integers(0, P, (2000, P))
    rows, tuned = [], {}
    for c in COSTS:
        rb = json.load(open(f"review/results/warmstart/richer_rule_params_adi_trading_{c}.json"))[str(c)]["rule_b"]
        # re-tune rule (b) with European Greeks on validation (American P&L), same grid
        best = (np.inf, None)
        for a_, b_, w_ in itertools.product(GRID["a"], GRID["b"], GRID["w"]):
            p, ob = rollout(va, ve, rule(a_, b_, w_))
            v = A.var_cvar(p - c * ob)[0]
            if v < best[0]:
                best = (float(v), (float(a_), float(b_), float(w_)))
        tuned[c] = dict(a=best[1][0], b=best[1][1], w=best[1][2], val_VaR95=best[0])
        print(c, "rule (b) with European Greeks, tuned on validation:", tuned[c], "| American:", rb, flush=True)
        pols = {"DG": (DG, DG), "Rule (b)": (rule(rb["a"], rb["b"], rb["w"]), rule(*best[1]))}
        for name, (fa, fe) in pols.items():
            pa_, oa = rollout(ta, ta, fa)
            pe_, oe = rollout(ta, te, fe)
            na, ne = pa_ - c * oa, pe_ - c * oe
            va_, ca_ = A.var_cvar(na); ve_, ce_ = A.var_cvar(ne)
            d = A.var_cvar(ne[idx])[0] - A.var_cvar(na[idx])[0]
            dc = A.var_cvar(ne[idx])[1] - A.var_cvar(na[idx])[1]
            rows.append(dict(cost=c, policy=name, VaR95_american=float(va_), VaR95_european=float(ve_),
                             diff_VaR95=float(ve_ - va_), diff_VaR95_se=float(d.std(ddof=1)),
                             CVaR95_american=float(ca_), CVaR95_european=float(ce_), diff_CVaR95=float(ce_ - ca_),
                             diff_CVaR95_se=float(dc.std(ddof=1)),
                             cost_american=float(c * oa.mean()), cost_european=float(c * oe.mean())))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "european_greeks.csv", index=False)
    json.dump({str(k): v for k, v in tuned.items()}, open(OUT / "european_greeks_rule_params.json", "w"), indent=2)
    md = ["# Hedging the American book with European Greeks (test seed 4242, 8,192 paths)", "",
          "All P&L with American prices; European Greeks only for the hedge decisions and the delta hedge. "
          "Difference = European - American (positive = European Greeks worse), paired bootstrap SE.", "",
          "| Cost | Policy | VaR95 American Greeks | VaR95 European Greeks | Difference | CVaR95 American | CVaR95 European | Difference |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r.cost:.1%} | {r.policy} | {r.VaR95_american:.2f} | {r.VaR95_european:.2f} | "
                  f"{r.diff_VaR95:+.2f} ({r.diff_VaR95_se:.2f}) | {r.CVaR95_american:.2f} | {r.CVaR95_european:.2f} | "
                  f"{r.diff_CVaR95:+.2f} ({r.diff_CVaR95_se:.2f}) |")
    md += ["", "Rule (b) with European Greeks re-tuned on validation: " +
           "; ".join(f"{c:.1%}: a={v['a']}, b={v['b']}, w={v['w']}" for c, v in tuned.items())]
    (OUT / "european_greeks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
