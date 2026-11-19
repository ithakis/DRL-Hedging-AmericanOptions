"""Figure 1 of the revised paper: hedging dynamics of the selected Heston agent (r = 2%, 2% cost, VaR95).

    PYTHONPATH=. python review/retrain/make_figure1.py     # -> review/results/paper/figures/hedging_dynamics.pdf

Uses the cached block-B test market (seed 654, 8,192 paths), the selected agent of analysis.py
(pick(runs_of("B", 0.02))) and the tuned rule (b) of baselines_B.json. Only plotting: no training.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from review.retrain.analysis import OUT, markets, pick, policy_decider, rule_decider, runs_of  # noqa: E402

SPREAD = 0.02


def rollout_trace(m, decide):
    """Same steps as analysis.rollout, but keeps per-day series (paths x days)."""
    m = {k: np.asarray(v) for k, v in m.items()}
    S = m["S"]
    P, T1 = S.shape
    T = T1 - 1
    n = np.zeros((P, T1))
    keys = ["g_before", "g_after", "v_before", "v_after", "g_hed", "g_liab", "v_hed", "v_liab",
            "pnl_stock", "pnl_liab", "pnl_hed", "cost"]
    tr = {k: np.zeros((P, T)) for k in keys}
    for t in range(T):
        act = ~m["h_inactive"][:, t, :]
        Gh = np.sum(n * m["h_gamma"][:, t, :] * act, axis=1)
        Vh = np.sum(n * m["h_vega"][:, t, :] * act, axis=1)
        G = m["l_gamma"][:, t] + Gh
        V = m["l_vega"][:, t] + Vh
        g_h, v_h = m["h_gamma"][:, t, t], m["h_vega"][:, t, t]
        with np.errstate(divide="ignore", invalid="ignore"):
            n_g = np.nan_to_num(-G / g_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
            n_v = np.nan_to_num(-V / v_h, nan=0.0, posinf=np.inf, neginf=-np.inf)
        lo = np.minimum(np.minimum(0.0, n_g), n_v)
        hi = np.maximum(np.maximum(0.0, n_g), n_v)
        obs = np.stack([S[:, t], G, g_h, V, v_h], axis=1)
        x = decide(t, obs, n_g, n_v, lo, hi)
        n[:, t] = x
        tr["g_before"][:, t], tr["v_before"][:, t] = G, V
        tr["g_after"][:, t], tr["v_after"][:, t] = G + x * g_h, V + x * v_h
        tr["g_hed"][:, t], tr["v_hed"][:, t] = Gh + x * g_h, Vh + x * v_h
        tr["g_liab"][:, t], tr["v_liab"][:, t] = m["l_gamma"][:, t], m["l_vega"][:, t]
        tr["cost"][:, t] = SPREAD * np.abs(x * m["h_price"][:, t, t])
        act_next = ~m["h_inactive"][:, t + 1, :]
        pos = -(m["l_delta"][:, t] + np.sum(n * m["h_delta"][:, t, :] * act, axis=1))
        tr["pnl_liab"][:, t] = m["l_next"][:, t] - m["l_price"][:, t]
        tr["pnl_hed"][:, t] = np.sum(n * (m["h_price"][:, t + 1, :] - m["h_price"][:, t, :]) * act_next, axis=1)
        tr["pnl_stock"][:, t] = pos * (S[:, t + 1] - S[:, t])
    tr["step"] = tr["pnl_stock"] + tr["pnl_liab"] + tr["pnl_hed"] - tr["cost"]
    return tr


def band(ax, y, label, c):
    d = np.arange(y.shape[1])
    ax.plot(d, y.mean(0), c=c, label=label)
    lo, hi = np.quantile(y, [0.025, 0.975], axis=0)
    ax.fill_between(d, lo, hi, color=c, alpha=0.22)


def var_cvar_pnl(p):
    q = np.quantile(p, 0.05)
    return q, p[p <= q].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "review" / "results" / "paper" / "figures" / "hedging_dynamics.pdf"))
    a = ap.parse_args()
    m = markets("B", which=("test",))["test"]
    sel, _ = pick(runs_of("B", SPREAD))
    f_rl, _ = policy_decider(sel["policy"])
    rb = json.load(open(OUT / "baselines_B.json"))[f"{SPREAD}_0"]["rule_b_VaR95"]
    rl = rollout_trace(m, f_rl)
    dg = rollout_trace(m, rule_decider("DG"))
    rule = rollout_trace(m, rule_decider("frac", rb["a"], rb["b"], rb["w"]))

    plt.style.use("default")  # environment/utils.py sets dark_background on import
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axs = plt.subplots(4, 2, figsize=(10, 9.5))
    C1, C2, C3 = "tab:blue", "tab:green", "tab:orange"

    ax = axs[0, 0]; band(ax, rl["g_before"], "Before", C1); band(ax, rl["g_after"], "After", C2)
    ax.set_title("Portfolio Gamma Before/After Hedge"); ax.set_ylabel("Gamma")
    ax = axs[0, 1]; band(ax, rl["g_hed"], "Hedging options", C1); band(ax, -rl["g_liab"], "Liability (sign flipped)", C2)
    ax.set_title("Hedging vs Liability Portfolio Gamma"); ax.set_ylabel("Gamma")
    ax = axs[1, 0]; band(ax, rl["v_before"], "Before", C1); band(ax, rl["v_after"], "After", C2)
    ax.set_title("Portfolio Vega Before/After Hedge"); ax.set_ylabel("Vega")
    ax = axs[1, 1]; band(ax, rl["v_hed"], "Hedging options", C1); band(ax, -rl["v_liab"], "Liability (sign flipped)", C2)
    ax.set_title("Hedging vs Liability Portfolio Vega"); ax.set_ylabel("Vega")

    ax = axs[2, 0]
    d = np.arange(rl["step"].shape[1])
    for k, lab, c in (("pnl_stock", "Underlying", C1), ("pnl_liab", "Liability", C3), ("pnl_hed", "Hedging options", C2)):
        ax.plot(d, rl[k].mean(0), c=c, label=lab)
    ax.set_title("Portfolio Components PnL (Mean per Day)"); ax.set_ylabel("PnL")
    ax = axs[2, 1]
    for tr, lab, c in ((rl, "RL", "tab:red"), (dg, "Delta-Gamma", "tab:gray"), (rule, "Rule (b)", C2)):
        cc = np.cumsum(tr["cost"], axis=1); ax.plot(d, cc.mean(0), c=c, label=lab)
        if tr is rl:
            lo, hi = np.quantile(cc, [0.025, 0.975], axis=0); ax.fill_between(d, lo, hi, color=c, alpha=0.2)
    ax.set_title("Cumulative Option Transaction Costs"); ax.set_ylabel("Cost")

    ax = axs[3, 0]
    cum = np.cumsum(rl["step"], axis=1)
    band(ax, cum, "Mean cumulative PnL (RL)", C1)
    q = np.quantile(cum, 0.05, axis=0)
    cv = np.array([cum[cum[:, j] <= q[j], j].mean() for j in range(cum.shape[1])])
    ax.plot(d, q, c=C2, ls="--", label="5% quantile (= $-$VaR$_{95\\%}$)")
    ax.plot(d, cv, c=C3, ls=":", label="$-$CVaR$_{95\\%}$")
    ax.set_title("Cumulative (Total) Portfolio PnL (RL)"); ax.set_ylabel("PnL")

    ax = axs[3, 1]
    lo_x, hi_x = np.inf, -np.inf
    for tr, lab, c in ((rl, "RL", "tab:red"), (dg, "Delta-Gamma", "tab:gray"), (rule, "Rule (b)", C2)):
        p = tr["step"].sum(1)
        v, _ = var_cvar_pnl(p)
        ax.hist(p, bins=120, density=True, histtype="step", color=c, lw=1.2, label=f"{lab}: VaR$_{{95\\%}}$ {-v:.2f}")
        ax.axvline(v, color=c, ls="--", lw=1)
        lo_x, hi_x = min(lo_x, np.quantile(p, 0.005)), max(hi_x, np.quantile(p, 0.995))
        print(f"{lab}: VaR95 {-v:.2f} mean PnL {p.mean():.2f} E[cost] {tr['cost'].sum(1).mean():.2f}")
    ax.set_xlim(lo_x, hi_x)
    ax.set_title("Portfolio PnL at $T$ (Dashed: 5% Quantile)"); ax.set_ylabel("Density"); ax.set_xlabel("PnL")

    for ax in axs.flat:
        if ax is not axs[3, 1]:
            ax.set_xlabel("Days")
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print("selected run:", sel["policy"], "saved", a.out)


if __name__ == "__main__":
    main()
